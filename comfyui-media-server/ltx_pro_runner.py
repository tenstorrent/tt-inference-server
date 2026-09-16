# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import tempfile

from ltx_pro_config import LTXProConfig
from ltx_runner import LTXRunner
from utils.logger import setup_logger


class LTXProRunner(LTXRunner):
    """Wrapper for tt-metal LTXOneStagePipeline (LTX-2.3 Pro audio+video).

    Subclasses LTXRunner because device setup and teardown are identical -- same
    2x2 mesh, same l1_small_size for the vocoder, same untraced run, same MP4
    return contract. Only model construction and the generate() call differ.

    Unlike the distilled runner, the negative prompt and the guidance settings
    are real inputs here: the one-stage pipeline runs classifier-free guidance,
    so it encodes the negative prompt and pushes against it.
    """

    def __init__(self, worker_id: int, config: LTXProConfig):
        super().__init__(worker_id, config)
        self.logger = setup_logger(f"LTXProRunner-{worker_id}")

    def load_model(self, kernel_ready_queue=None):
        from models.tt_dit.pipelines.ltx.pipeline_ltx_one_stage import LTXOneStagePipeline
        from models.tt_dit.utils.ltx import default_ltx_checkpoint, default_ltx_gemma

        # Read by _resolve_quant_config at pipeline construction.
        os.environ["LTX_QUANT"] = self.config.quant
        os.environ.setdefault("TT_DIT_CACHE_DIR", self.config.tt_dit_cache_dir)

        checkpoint = default_ltx_checkpoint(self.config.model_name)
        gemma = default_ltx_gemma()

        self.logger.info("Creating LTX-2.3 Pro (one-stage) pipeline...")
        self.logger.info(f"  checkpoint: {checkpoint}")
        self.logger.info(f"  gemma: {gemma}")
        self.logger.info(f"  size: {self.config.width}x{self.config.height}, frames: {self.config.num_frames}")
        self.logger.info(f"  quant: {self.config.quant}, use_trace: {self.config.use_trace}")
        self.logger.info(
            f"  steps: {self.config.num_inference_steps}, "
            f"cfg: video={self.config.video_cfg_scale} audio={self.config.audio_cfg_scale}, "
            f"stg: video={self.config.video_stg_scale} audio={self.config.audio_stg_scale} "
            f"block={self.config.stg_block}"
        )
        self.logger.info(f"  tt_dit_cache: {os.environ['TT_DIT_CACHE_DIR']}")

        # Geometry is fixed here for the same reason as the distilled pipeline, and
        # the parallelism kwargs are again omitted so create_pipeline resolves the
        # Blackhole (2,2) preset. run_warmup compiles call_av and the decoders up
        # front so the first request is not charged for JIT.
        self.pipeline = LTXOneStagePipeline.create_pipeline(
            mesh_device=self.mesh_device,
            checkpoint_name=checkpoint,
            gemma_path=gemma,
            run_warmup=True,
            traced=self.config.use_trace,
            num_frames=self.config.num_frames,
            height=self.config.height,
            width=self.config.width,
        )

        pc = self.pipeline.parallel_config
        self.logger.info(
            "LTX Pro pipeline created; resolved preset: "
            f"sp_axis={pc.sequence_parallel.mesh_axis} tp_axis={pc.tensor_parallel.mesh_axis} "
            f"sp={pc.sequence_parallel.factor} tp={pc.tensor_parallel.factor} "
            f"dynamic_load={getattr(self.pipeline, 'dynamic_load', None)} "
            f"is_fsdp={getattr(self.pipeline, 'is_fsdp', None)}"
        )

        if kernel_ready_queue is not None:
            kernel_ready_queue.put(self.worker_id)

    def run_inference(self, requests, on_event=None) -> list:
        """Generate one guided AV clip. Returns [mp4_bytes]. See LTXRunner."""
        request = requests[0]

        prompt = request["prompt"]
        if not prompt or not prompt.strip():
            raise ValueError("LTX generation requires a non-empty prompt")
        seed = request.get("seed")
        seed = int(seed) if seed is not None else 0

        self._reject_shape_mismatch(request)

        # Per-request guidance overrides, falling back to the server's defaults.
        def opt(name):
            got = request.get(name)
            return getattr(self.config, name) if got is None else got

        num_inference_steps = int(opt("num_inference_steps"))
        if num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1 (got {num_inference_steps})")

        # A negative prompt of None makes the pipeline substitute its own default
        # constant; an empty string would instead encode empty text, which is a
        # weaker uncond. Preserve that distinction.
        negative_prompt = request.get("negative_prompt") or None

        self.logger.info(
            f"Running Pro AV inference: prompt='{prompt[:80]}', "
            f"negative={'set' if negative_prompt else 'pipeline default'}, "
            f"size={self.config.width}x{self.config.height}, frames={self.config.num_frames}, "
            f"steps={num_inference_steps}, seed={seed}"
        )

        kwargs = dict(
            num_frames=self.config.num_frames,
            height=self.config.height,
            width=self.config.width,
            seed=seed,
            fps=self.config.fps,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            video_cfg_scale=float(opt("video_cfg_scale")),
            audio_cfg_scale=float(opt("audio_cfg_scale")),
            video_stg_scale=float(opt("video_stg_scale")),
            audio_stg_scale=float(opt("audio_stg_scale")),
            video_modality_scale=float(opt("video_modality_scale")),
            audio_modality_scale=float(opt("audio_modality_scale")),
            rescale_scale=float(opt("rescale_scale")),
            stg_block=int(opt("stg_block")),
        )
        if on_event is not None:
            kwargs["on_event"] = on_event

        tmp = tempfile.NamedTemporaryFile(prefix=f"ltxpro_w{self.worker_id}_", suffix=".mp4", delete=False)
        tmp.close()
        try:
            self.pipeline.generate(prompt, output_path=tmp.name, **kwargs)
            with open(tmp.name, "rb") as fh:
                data = fh.read()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        self.logger.info(f"Pro AV inference complete: {len(data)} bytes of MP4")
        return [data]
