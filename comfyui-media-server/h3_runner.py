# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import base64
import io
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np
import ttnn

from h3_config import H3Config
from utils.logger import setup_logger

_WARMUP_PROMPT = "A paper boat drifts across a still pond at dawn while birds call in the distance."


def _decode_image(b64: str):
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _parse_aspect_ratio(text: str) -> tuple:
    try:
        w, h = (int(x) for x in text.split(":"))
    except ValueError as e:
        raise ValueError(f"aspect_ratio must look like '16:9', got {text!r}") from e
    return (w, h)


class H3Runner:
    """Wrapper for tt-metal MiniMaxH3TurboPipeline (MiniMax-H3 audio+video).

    Same contract as LTXRunner: run_inference returns muxed MP4 bytes, since the
    clip carries audio. Clip geometry varies per request here, so the shape of
    the last clip is left in ``last_av_meta`` for the worker to report.

    The pipeline only ships Galaxy presets (4x8, 4x32). On a QuietBox 2 it runs
    the 1x4 mesh as TP=4 / SP=1 with every parallel setting passed explicitly,
    the text encoder and transformer taking turns on device (not co-resident).
    """

    def __init__(self, worker_id: int, config: H3Config):
        self.worker_id = worker_id
        self.config = config
        self.logger = setup_logger(f"H3Runner-{worker_id}")
        self.mesh_device = None
        self.pipeline = None
        self._fabric_config = None
        self.last_av_meta = None

    def initialize_device(self):
        rows, cols = self.config.device_mesh_shape
        self.logger.info(
            f"Initializing {rows}x{cols} mesh device for worker {self.worker_id} "
            f"on board {self.config.board.name.lower()}"
        )
        fabric_config = getattr(ttnn.FabricConfig, self.config.fabric_config_name)
        ttnn.set_fabric_config(
            fabric_config,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
        )
        self._fabric_config = fabric_config
        self.mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(rows, cols),
            l1_small_size=self.config.l1_small_size,
            dispatch_core_config=ttnn.DispatchCoreConfig(),
        )
        self.logger.info(
            f"Mesh device initialized: shape={tuple(self.mesh_device.shape)}, "
            f"fabric={self.config.fabric_config_name}, l1_small_size={self.config.l1_small_size}"
        )
        return self.mesh_device

    def _resolve_lora_path(self) -> str:
        if self.config.lora_path:
            return self.config.lora_path
        from huggingface_hub import hf_hub_download

        return hf_hub_download(self.config.lora_repo, self.config.lora_filename)

    def load_model(self, kernel_ready_queue=None):
        from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3 import MINIMAX_H3_BUCKET_LADDER_4X8
        from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3_turbo import MiniMaxH3TurboPipeline

        os.environ.setdefault("TT_DIT_CACHE_DIR", self.config.tt_dit_cache_dir)
        lora_path = self._resolve_lora_path()

        self.logger.info("Creating MiniMax-H3 Turbo pipeline...")
        self.logger.info(f"  weights: {self.config.weights_dir or '(HF cache)'}")
        self.logger.info(f"  turbo adapter: {lora_path}")
        self.logger.info(f"  steps: {self.config.num_inference_steps}, shifts: {self.config.video_shift}/{self.config.audio_shift}")
        self.logger.info(f"  tt_dit_cache: {os.environ['TT_DIT_CACHE_DIR']}")

        self.pipeline = MiniMaxH3TurboPipeline.create_pipeline(
            mesh_device=self.mesh_device,
            weights_dir=self.config.weights_dir,
            lora_path=lora_path,
            video_shift=self.config.video_shift,
            audio_shift=self.config.audio_shift,
            task="t2va",
            tp_axis=1,
            sp_axis=0,
            num_links=1,
            topology=ttnn.Topology.Linear,
            dit_fsdp=False,
            coresident=False,
            trace_denoise=False,
            trace_audio=False,
            bucket_denoise=True,
            # No 1x4 preset. The Galaxy rungs are 256-aligned, which also satisfies SP=1.
            bucket_ladder=MINIMAX_H3_BUCKET_LADDER_4X8,
            use_persistent_ccl_buffers=False,
            # yuv420 selects the device-stitched VAE decode, which needs every tile of a
            # chunk in one wave: 28 tiles at 768P against 4 devices.
            vae_output_type="float",
            # The pipeline's own warmup forces every bucket rung; one real generation
            # below compiles the default shape instead.
            warmup=False,
        )

        if kernel_ready_queue is not None:
            kernel_ready_queue.put(self.worker_id)

        if not self.config.dev_mode:
            self.logger.info("Running warmup generation at the default shape (compiles kernels)...")
            self.run_inference([{"prompt": _WARMUP_PROMPT, "seed": 0}])
            self.logger.info("Warmup complete")

    def run_inference(self, requests, on_event=None) -> list:
        """Generate one AV clip. Returns [mp4_bytes]."""
        from models.tt_dit.pipelines.minimax_h3.policy import get_num_frames

        request = requests[0]
        prompt = request.get("prompt") or ""
        if not prompt.strip():
            raise ValueError("H3 generation requires a non-empty prompt")
        seed = int(request.get("seed") or 0)
        duration = float(request.get("duration_s") or self.config.default_duration_s)
        if not 4 <= duration <= 15:
            raise ValueError(f"duration_s must be between 4 and 15 seconds, got {duration}")
        num_frames = get_num_frames(duration)
        aspect_ratio = _parse_aspect_ratio(request.get("aspect_ratio") or self.config.default_aspect_ratio)
        image = _decode_image(request["first_image_b64"]) if request.get("first_image_b64") else None
        last_image = _decode_image(request["last_image_b64"]) if request.get("last_image_b64") else None
        mode = "fl2va" if (image or last_image) else "t2va"

        self.logger.info(
            f"Running {mode}: prompt='{prompt[:80]}', {duration:g}s ({num_frames} frames), "
            f"aspect={aspect_ratio[0]}:{aspect_ratio[1]}, seed={seed}"
        )
        out = self.pipeline(
            prompt,
            image=image,
            last_image=last_image,
            num_frames=num_frames,
            aspect_ratio=aspect_ratio,
            seed=seed,
            num_inference_steps=self.config.num_inference_steps,
            on_event=on_event,
        )

        frames = out.video[0].permute(1, 2, 3, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy()
        data = self._mux(frames, out.audio, out.sampling_rate, out.fps)
        _, height, width, _ = frames.shape
        self.last_av_meta = {"num_frames": int(out.num_frames), "height": height, "width": width, "fps": int(out.fps)}
        self.logger.info(f"AV inference complete: {width}x{height}, {out.num_frames} frames, {len(data)} bytes of MP4")
        return [data]

    def _mux(self, frames: np.ndarray, audio, sampling_rate: int, fps: int) -> bytes:
        """Encode (F, H, W, 3) uint8 frames and (1, 2, samples) audio to an h264 + AAC MP4."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        num_frames, height, width, _ = frames.shape
        pcm = np.clip(np.ascontiguousarray(audio[0].float().cpu().numpy().T), -1.0, 1.0)
        with tempfile.TemporaryDirectory(prefix=f"h3_w{self.worker_id}_") as tmp:
            wav_path = os.path.join(tmp, "audio.wav")
            mp4_path = os.path.join(tmp, "clip.mp4")
            with wave.open(wav_path, "wb") as handle:
                handle.setnchannels(pcm.shape[1])
                handle.setsampwidth(2)
                handle.setframerate(sampling_rate)
                handle.writeframes((pcm * 32767.0).astype("<i2").tobytes())
            subprocess.run(
                [
                    ffmpeg, "-y", "-v", "error",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                    "-i", wav_path,
                    "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", mp4_path,
                ],
                input=frames.tobytes(),
                check=True,
                capture_output=True,
            )
            with open(mp4_path, "rb") as fh:
                return fh.read()

    def close_device(self):
        self.pipeline = None
        if self.mesh_device is not None:
            self.logger.info("Closing mesh device")
            try:
                ttnn.close_mesh_device(self.mesh_device)
            except Exception as e:
                self.logger.warning(f"close_mesh_device failed: {e}")
            self.mesh_device = None
        if self._fabric_config is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            self._fabric_config = None
