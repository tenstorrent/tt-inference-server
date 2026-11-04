# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Runners of the generations API. Each consumes its model's request from
domain/ directly, with the inputs the service resolved before queueing it:
decoded images, output size, seed and the prompt the model gets.

The tt_dit pipelines take the output size per call, so one deployment serves
every size its request schema allows. Warmup runs every size and input
combination the pipeline does not warm up itself, so none compiles after the
server reports ready.

The parameters the API leaves out (steps, guidance, sampler shift) are the
models' reference defaults.
"""

import os
import shutil
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import ttnn
from config.constants import (
    CANARY_DEEP_TASK_ID,
    LTX_FPS,
    LTX_HEIGHT,
    LTX_NUM_FRAMES,
    LTX_WIDTH,
    MINIMAX_H3_LIGHTX2V_NUM_INFERENCE_STEPS,
    WAN22_NUM_FRAMES,
    Resolution,
    generation_job_dir,
)
from config.settings import get_settings
from domain.flux2_klein_4b_request import KLEIN_STEPS, Flux2klein4bRequest
from domain.generation_request import GenerationRequest
from domain.ltx23_distilled_request import Ltx23DistilledRequest
from domain.minimax_h3_request import H3_SHORT_EDGE, _MiniMaxH3Request
from domain.wan22_i2v_a14b_request import Wan22i2vA14bRequest
from models.common.utility_functions import is_blackhole
from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3 import (
    MiniMaxH3Pipeline,
    resolve_mesh_preset,
)
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt, WanPipelineI2V
from PIL import Image
from telemetry.image_metrics import ImageStageRecorder, sampler_name
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.dit_runners import (
    TTDiTRunner,
    TTWan22I2VRunner,
    _wan22_dit_device_params,
    dit_runner_log_map,
)
from utils.decorators import log_execution_time
from utils.logger import log_exception_chain
from utils.media_probe import decode_image
from utils.video_manager import VideoManager

_WARMUP_PROMPT = "Sunrise on a beach"
# Warmup only compiles and traces, so it runs fewer steps where it can.
_WARMUP_STEPS = 2

WAN_STEPS = 40
WAN_GUIDANCE_SCALES = (3.5, 3.5)
WAN_SHIFT = 5.0

_inference_timer = log_execution_time(
    f"{dit_runner_log_map[get_settings().model_runner]} inference",
    TelemetryEvent.MODEL_INFERENCE,
    os.environ.get("TT_VISIBLE_DEVICES"),
)


def _warmup_request(
    request: GenerationRequest,
    width: int,
    height: int,
    images: dict[str, list[Image.Image]] | None = None,
) -> GenerationRequest:
    request._width, request._height = width, height
    request._images = images or {}
    request._prompt = _WARMUP_PROMPT
    return request


class _GenerationRunner(TTDiTRunner):
    def _steps(self, steps: int) -> int:
        return _WARMUP_STEPS if self._warming_up else steps

    def _image_stage_recorder(self) -> ImageStageRecorder | None:
        if self._warming_up:
            return None
        return ImageStageRecorder(
            model_type=self.settings.model_runner,
            device_id=self.device_id,
            sampler=sampler_name(self.pipeline),
            batch=1,
        )


class TTWan22I2VA14BRunner(_GenerationRunner, TTWan22I2VRunner):
    def create_pipeline(self):
        # The mesh's preset decides the sizes; the pipeline rejects the others
        # per call.
        try:
            return WanPipelineI2V.create_pipeline(
                mesh_device=self.ttnn_device, num_frames=WAN22_NUM_FRAMES
            )
        except Exception as e:
            log_exception_chain(
                self.logger, self.device_id, "Wan I2V pipeline creation failed", e
            )
            raise

    def warmup_requests(self) -> list[GenerationRequest]:
        # The pipeline's own warmup is untraced and first frame only. Each size
        # is traced here, with and without the last frame: conditioning it makes
        # the VAE encode all frames, which compiles anew.
        supported = set(self.pipeline.resolutions)
        return [
            _warmup_request(
                Wan22i2vA14bRequest.model_construct(
                    prompt=_WARMUP_PROMPT,
                    negative_prompt=_WARMUP_PROMPT,
                    last_frame="warmup" if last_frame else None,
                    frames=WAN22_NUM_FRAMES,
                ),
                width,
                height,
                {
                    name: [Image.new("RGB", (width, height))]
                    for name in ("first_frame", "last_frame")[: 1 + last_frame]
                },
            )
            for width, height in Wan22i2vA14bRequest.warmup_sizes()
            if (width, height) in supported
            for last_frame in (False, True)
        ]

    @_inference_timer
    def run(self, requests: list[Wan22i2vA14bRequest]):
        request = requests[0]
        # The pipeline center-crops each image to the output's aspect ratio and
        # resizes it.
        image_prompt = [
            ImagePrompt(image=request._images["first_frame"][0], frame_pos=0)
        ]
        if request.last_frame is not None:
            image_prompt.append(
                ImagePrompt(
                    image=request._images["last_frame"][0],
                    frame_pos=request.frames - 1,
                )
            )
        recorder = self._video_stage_recorder(
            Resolution(height=request._height, width=request._width)
        )
        frames = None
        try:
            frames = self.pipeline(
                prompts=[request._prompt],
                negative_prompts=[request.negative_prompt],
                image_prompt=image_prompt,
                num_inference_steps=self._steps(WAN_STEPS),
                guidance_scale=WAN_GUIDANCE_SCALES[0],
                guidance_scale_2=WAN_GUIDANCE_SCALES[1],
                flow_shift=WAN_SHIFT,
                seed=request._seed,
                height=request._height,
                width=request._width,
                traced=True,
                on_event=recorder,
            )
        finally:
            if recorder is not None:
                recorder.flush(frames)
        return frames


class TTLtx23DistilledRunner(_GenerationRunner):
    def create_pipeline(self):
        from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import (
            LTXDistilledPipeline,
        )
        from models.tt_dit.utils.ltx import default_ltx_gemma

        # The (4, 8) Blackhole Galaxy ring, as LTXPipeline.create_pipeline's
        # own 4x8 entry. The shape and fps are fixed here: fps sets the audio
        # length and the audio-video positions, which the traces bake in.
        try:
            return LTXDistilledPipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                checkpoint_name=self.settings.model_weights_path,
                gemma_path=default_ltx_gemma(),
                sp_axis=1,
                tp_axis=0,
                num_links=2,
                dynamic_load=False,
                topology=ttnn.Topology.Ring,
                is_fsdp=False,
                traced=True,
                num_frames=LTX_NUM_FRAMES,
                height=LTX_HEIGHT,
                width=LTX_WIDTH,
                fps=LTX_FPS,
                image_conditioning=False,
            )
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "LTX-2.3 distilled pipeline creation failed",
                e,
            )
            raise

    def get_pipeline_device_params(self):
        # The Wan2.2 ring and fabric defaults, plus L1_SMALL for the audio
        # vocoder (without it: "bank size is 0 B") and the region for the two
        # traced stages, from tt-metal's
        # models/tt_dit/tests/models/ltx/ltx_mesh_params.py.
        device_params = _wan22_dit_device_params(self.settings.device_mesh_shape)
        device_params["l1_small_size"] = 32768
        device_params["trace_region_size"] = 500_000_000
        return device_params

    def warmup_requests(self) -> list[GenerationRequest]:
        # The first traced call captures the traces of the one shape served.
        return [
            _warmup_request(Ltx23DistilledRequest.model_construct(), width, height)
            for width, height in Ltx23DistilledRequest.warmup_sizes()
        ]

    @_inference_timer
    def run(self, requests: list[Ltx23DistilledRequest]):
        request = requests[0]
        # Into the job's directory, so a file a cancelled or failed job leaves
        # behind is deleted with it.
        directory = Path(generation_job_dir(request._task_id))
        directory.mkdir(parents=True, exist_ok=True)
        try:
            # generate() writes an MP4 with the soundtrack and returns its path.
            path = self.pipeline.generate(
                request._prompt,
                output_path=str(directory / "ltx.mp4"),
                num_frames=request.frames,
                height=request._height,
                width=request._width,
                seed=request._seed,
                fps=request.fps,
            )
        finally:
            if self._warming_up:
                # No job owns warmup's video.
                shutil.rmtree(directory, ignore_errors=True)
        return [path]


MINIMAX_H3_TRACE_REGION_BYTES = 1_175_000_000
MINIMAX_H3_L1_SMALL_SIZE_DEFAULT_BH_GALAXY = 65536
MINIMAX_H3_L1_SMALL_SIZE_DEFAULT_WH_GALAXY = 32768
MINIMAX_H3_MAX_PAYLOAD_BYTES_BH_GALAXY = 8192
MINIMAX_H3_MAX_PAYLOAD_BYTES_WH_GALAXY = 4096
# The denoising steps of the base model; the distilled adapters fix their own.
MINIMAX_H3_STEPS = 50


class TTFlux2Klein4BRunner(_GenerationRunner):
    def create_pipeline(self):
        from models.tt_dit.pipelines.flux2.pipeline_flux2 import Flux2Pipeline

        try:
            return Flux2Pipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                checkpoint_name=self.settings.model_weights_path,
            )
        except Exception as e:
            log_exception_chain(
                self.logger, self.device_id, "FLUX.2 pipeline creation failed", e
            )
            raise

    def get_pipeline_device_params(self):
        # From tt-metal's models/tt_dit/tests/models/flux2/device_params.py: the
        # VAE's conv2d needs L1_SMALL. The pipeline captures a denoising trace
        # per size and reference image, plus the VAE traces, when it is created;
        # they take 42.2 MB on BH 2x2 (KLEIN_TRACE_REGION_SIZE).
        return {"l1_small_size": 65536, "trace_region_size": 50_000_000}

    def warmup_requests(self) -> list[GenerationRequest]:
        # The pipeline compiles and traces every size, with and without a
        # reference image, when it is created.
        return []

    @_inference_timer
    def run(self, requests: list[Flux2klein4bRequest]):
        request = requests[0]
        recorder = self._image_stage_recorder()
        images = self.pipeline(
            prompts=[request._prompt],
            images=request._images.get("images") or None,
            num_inference_steps=self._steps(KLEIN_STEPS),
            seed=request._seed,
            width=request._width,
            height=request._height,
            traced=True,
            on_event=recorder,
        )
        if recorder is not None:
            recorder.flush(images)
        return images


def _minimax_h3_env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    key = raw.strip().lower()
    if key in ("1", "true"):
        return True
    if key in ("0", "false"):
        return False
    raise ValueError(f"{name}={raw!r} must be 1/true or 0/false")


def _minimax_h3_device_params(mesh_shape: tuple, *, l1_small_size: int = None) -> dict:
    """Device params for MiniMax-H3, keyed on the mesh shape.

    Everything here is derived from the pipeline's per-shape preset so the two can't disagree:
    fabric follows topology (Ring -> FABRIC_1D_RING), and a `trace_region_size` is reserved when
    `trace_denoise` is on (preset: the 4x32 quad; override: MINIMAX_H3_TRACE_DENOISE) -- without it
    the pipeline's trace capture is fatal. The region is only reserved, so a shape that does not
    trace pays nothing but address space. `l1_small_size` is mandatory (a bare open fails as
    "bank size is 0 B"). Ref2VA uses 16384: the video VAE's taps=3 encoder clashes with a 65536
    pool.
    """
    preset = resolve_mesh_preset(mesh_shape)
    router_config = ttnn.FabricRouterConfig()
    router_config.max_packet_payload_size_bytes = (
        MINIMAX_H3_MAX_PAYLOAD_BYTES_BH_GALAXY
        if is_blackhole()
        else MINIMAX_H3_MAX_PAYLOAD_BYTES_WH_GALAXY
    )
    ring = preset["topology"] == ttnn.Topology.Ring
    if l1_small_size is None:
        l1_small_size = (
            MINIMAX_H3_L1_SMALL_SIZE_DEFAULT_BH_GALAXY
            if is_blackhole()
            else MINIMAX_H3_L1_SMALL_SIZE_DEFAULT_WH_GALAXY
        )
    params = {
        "fabric_config": (
            ttnn.FabricConfig.FABRIC_1D_RING if ring else ttnn.FabricConfig.FABRIC_1D
        ),
        "fabric_router_config": router_config,
        "l1_small_size": l1_small_size,
    }
    env_trace = _minimax_h3_env_bool("MINIMAX_H3_TRACE_DENOISE")
    trace = preset.get("trace_denoise", False) if env_trace is None else env_trace
    if trace:
        params["trace_region_size"] = MINIMAX_H3_TRACE_REGION_BYTES
    return params


class _MiniMaxH3Runner(_GenerationRunner):
    """MiniMax-H3: a video and its soundtrack, from text and the inputs of the
    task the deployment serves."""

    pipeline_task = "t2va"
    dit_fsdp = not is_blackhole()
    num_inference_steps = MINIMAX_H3_STEPS
    # Off on the other ranks of a multi-host mesh, whose output is discarded.
    export_in_runner = True

    def _weights_dir(self) -> str | None:
        """A local snapshot directory, or None to let the pipeline read its own env var.

        `settings.model_weights_path` defaults to the **HuggingFace repo id**
        (`MiniMaxAI/MiniMax-H3`), not a path, so passing it through unconditionally would send the
        pipeline looking for `MiniMaxAI/MiniMax-H3/transformer/config.json`. Only an actual
        directory is used; otherwise `MINIMAX_H3_DIFFUSERS_DIR` decides, which is what the model's
        own tests and docs use. The weights are ~62 GB per transformer partition and are always
        mounted, never baked into the image.
        """
        configured = self.settings.model_weights_path
        if configured and os.path.isdir(configured):
            return configured
        return None

    def create_pipeline(self):
        try:
            # No topology arg: the pipeline resolves it from its per-shape preset.
            return self._pipeline_factory().create_pipeline(
                mesh_device=self.ttnn_device,
                weights_dir=self._weights_dir(),
                task=self.pipeline_task,
                dit_fsdp=self.dit_fsdp,
                trace_denoise=_minimax_h3_env_bool("MINIMAX_H3_TRACE_DENOISE"),
                bucket_denoise=_minimax_h3_env_bool("MINIMAX_H3_BUCKET_DENOISE"),
                **self._create_pipeline_kwargs(),
            )
        except Exception as e:
            log_exception_chain(
                self.logger, self.device_id, "MiniMax-H3 pipeline creation failed", e
            )
            raise

    def get_pipeline_device_params(self):
        return _minimax_h3_device_params(self.settings.device_mesh_shape)

    def warmup_requests(self) -> list[GenerationRequest]:
        # The pipeline warms up every shape and input layout it serves when it
        # is created.
        return []

    def _build_warmup_video_request(self) -> GenerationRequest:
        """A multi-host deep canary's request: the shortest video the task
        takes."""
        request = _MiniMaxH3Request.model_construct(prompt=_WARMUP_PROMPT, duration=4)
        request._task_id = CANARY_DEEP_TASK_ID
        request._prompt = _WARMUP_PROMPT
        return request

    @_inference_timer
    def run(self, requests: list[GenerationRequest]):
        """Generates the video and writes it, with its soundtrack, as an MP4
        into the job's directory; returns its path, or None where no job keeps
        the video.

        Writing it here keeps the frames, hundreds of MB, out of the queue to
        the server process.
        """
        request = requests[0]
        try:
            inputs = self._inputs(request)
        except (ValueError, ImportError):
            raise
        except Exception as e:
            # A ValueError fails only the request; on a multi-host mesh,
            # anything else ends the job on every rank.
            raise ValueError(f"an input could not be decoded: {e}") from e
        output = self.pipeline(
            request._prompt,
            num_frames=request.frames,
            # With a keyframe, the pipeline takes the shape from it.
            **(
                {}
                if request.aspect_ratio == "adaptive"
                else {"aspect_ratio": tuple(map(int, request.aspect_ratio.split(":")))}
            ),
            num_inference_steps=self.num_inference_steps,
            seed=request._seed,
            **inputs,
        )
        if (
            self._warming_up
            or not self.export_in_runner
            or request._task_id == CANARY_DEEP_TASK_ID
        ):
            return [None]
        if getattr(output, "video_format", "rgb_float") == "yuv420":
            # Planar (F, H*3//2, W) uint8 straight off the device: already
            # ffmpeg's rawvideo yuv420p layout, with no host colour conversion.
            frames = np.ascontiguousarray(output.video)
            pixel_format = "yuv420p"
        else:
            # (1, 3, F, H, W) in [0, 1] -> (F, H, W, 3), as the encoder takes it.
            frames = output.video[0].permute(1, 2, 3, 0).contiguous().numpy()
            pixel_format = "rgb24"
        encoded = VideoManager().export_to_mp4_with_audio(
            frames,
            output.audio[0].numpy(),
            output.sampling_rate,
            fps=output.fps,
            pixel_format=pixel_format,
        )
        # Into the job's directory, so a file a cancelled or failed job leaves
        # behind is deleted with it.
        directory = Path(generation_job_dir(request._task_id))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "h3.mp4"
        shutil.move(encoded, path)
        return [str(path)]

    def _pipeline_factory(self):
        return MiniMaxH3Pipeline

    def _create_pipeline_kwargs(self) -> dict:
        """Extra ``create_pipeline(...)`` kwargs beyond mesh/weights/task/trace."""
        return {}

    def _inputs(self, request: GenerationRequest) -> dict:
        """The task's inputs, as ``pipeline(...)`` kwargs."""
        return {}

    @staticmethod
    def _images(request: GenerationRequest, name: str) -> list[Image.Image]:
        return [decode_image(data) for data in request._files.get(name, [])]


class _MiniMaxH3KeyframesRunner(_MiniMaxH3Runner):
    """Text, with an optional first and last keyframe."""

    def _inputs(self, request: GenerationRequest) -> dict:
        first = self._images(request, "first_frame")
        last = self._images(request, "last_frame")
        if not (first or last):
            return {}
        return {
            "image": first[0] if first else None,
            "last_image": last[0] if last else None,
        }


class _MiniMaxH3ReferencesRunner(_MiniMaxH3Runner):
    """Text with reference images, videos and audio clips, packed in that order.
    Loads ``transformer_ref/``."""

    pipeline_task = "ref2va"
    # transformer_ref is ~62 GB SP-replicated; without FSDP each 32 GB chip OOMs on 4x32 activations.
    dit_fsdp = True

    def get_pipeline_device_params(self):
        return _minimax_h3_device_params(
            self.settings.device_mesh_shape, l1_small_size=16384
        )

    def _build_warmup_video_request(self) -> GenerationRequest:
        request = super()._build_warmup_video_request()
        png = BytesIO()
        Image.new("RGB", (H3_SHORT_EDGE, H3_SHORT_EDGE), "gray").save(png, "PNG")
        request._files = {"images": [png.getvalue()]}
        return request

    def _inputs(self, request: GenerationRequest) -> dict:
        from models.tt_dit.pipelines.minimax_h3.packing_ref2va import (
            MiniMaxH3Reference,
            decode_reference_audio,
            reference_from_video_file,
        )

        references = [
            MiniMaxH3Reference(image=image) for image in self._images(request, "images")
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, data in enumerate(request._files.get("videos", [])):
                path = Path(directory) / f"video{index}.mp4"
                path.write_bytes(data)
                references.append(reference_from_video_file(str(path)))
            for index, data in enumerate(request._files.get("audios", [])):
                path = Path(directory) / f"audio{index}.wav"
                path.write_bytes(data)
                waveform, sample_rate = decode_reference_audio(str(path))
                references.append(
                    MiniMaxH3Reference(audio=waveform, sample_rate=sample_rate)
                )
        return {"references": references}


class _MiniMaxH3TurboMixin:
    """Binds a lightx2v 4-step Turbo adapter onto an H3 runner's transformer.

    The FL2VA and Ref2VA adapters are separate files (Ref2VA's targets ``transformer_ref/``), so one
    deployment serves one task with the adapter for that task in ``MINIMAX_H3_LORA_PATH``. The
    adapter's own ``alpha / rank`` is applied by the loader; ``MINIMAX_H3_LORA_STRENGTH`` multiplies
    it and defaults to 1.0.

    Each adapter was distilled at its own (video, audio) shift. A wrong shift still completes, over
    the wrong sigma grid, so the shift is pinned per runner to the adapter it serves.
    """

    num_inference_steps = MINIMAX_H3_LIGHTX2V_NUM_INFERENCE_STEPS
    video_shift: float
    audio_shift: float

    def _pipeline_factory(self):
        from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3_turbo import (
            MiniMaxH3TurboPipeline,
        )

        return MiniMaxH3TurboPipeline

    def _create_pipeline_kwargs(self) -> dict:
        lora_path = os.environ.get("MINIMAX_H3_LORA_PATH")
        if not lora_path or not os.path.isfile(lora_path):
            raise ValueError(
                f"{type(self).__name__} needs MINIMAX_H3_LORA_PATH set to the lightx2v Turbo "
                f"adapter's .safetensors file; got {lora_path!r}"
            )
        return {
            "lora_path": lora_path,
            "video_shift": float(
                os.environ.get("MINIMAX_H3_VIDEO_SHIFT", self.video_shift)
            ),
            "audio_shift": float(
                os.environ.get("MINIMAX_H3_AUDIO_SHIFT", self.audio_shift)
            ),
        }


class TTMiniMaxH3Runner(_MiniMaxH3KeyframesRunner):
    """MiniMax-H3, from text or keyframes (FL2VA)."""


class TTMiniMaxH3TurboRunner(_MiniMaxH3TurboMixin, TTMiniMaxH3Runner):
    """MiniMax-H3 FL2VA with the lightx2v FL2VA Turbo 4-step v1.x 768p adapter."""

    video_shift = 6.0
    audio_shift = 3.0


class TTMiniMaxH3Ref2VARunner(_MiniMaxH3ReferencesRunner):
    """MiniMax-H3 Ref2VA."""


class TTMiniMaxH3TurboRef2VARunner(_MiniMaxH3TurboMixin, TTMiniMaxH3Ref2VARunner):
    """MiniMax-H3 Ref2VA with the lightx2v Ref2VA Turbo 4-step v1.0 768p adapter."""

    video_shift = 12.0
    audio_shift = 3.0
