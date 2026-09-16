# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import tempfile

import ttnn

from ltx_config import LTXConfig
from utils.logger import setup_logger


class LTXRunner:
    """Wrapper for tt-metal LTXDistilledPipeline (LTX-2.3 audio+video).

    Mirrors WanRunner in shape: initialize_device → load_model → run_inference →
    close_device.

    Departs from the other runners in what it returns. LTX-2.3 emits synchronized
    audio alongside video, and the runner contract elsewhere here is a bare
    ndarray of frames with no audio channel. Rather than return frames and drop
    the audio, run_inference returns the **muxed MP4 bytes** that the pipeline
    already writes. That is also far smaller on the wire: 241 frames at 576x1024
    is ~400MB of raw RGB (and ~290MB of base64 PNG at 1080p/145f), against a
    ~1MB MP4.
    """

    def __init__(self, worker_id: int, config: LTXConfig):
        self.worker_id = worker_id
        self.config = config
        self.logger = setup_logger(f"LTXRunner-{worker_id}")
        self.mesh_device = None
        self.parent_mesh = None
        self.pipeline = None
        self._fabric_config = None

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

        # No trace_region_size: this pipeline runs untraced.
        self.parent_mesh = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(rows, cols),
            l1_small_size=self.config.l1_small_size,
            dispatch_core_config=ttnn.DispatchCoreConfig(),
        )
        self.mesh_device = self.parent_mesh.create_submesh(ttnn.MeshShape(rows, cols))
        self.logger.info(
            f"Mesh device initialized: shape={tuple(self.mesh_device.shape)}, "
            f"fabric={self.config.fabric_config_name}, l1_small_size={self.config.l1_small_size}"
        )
        return self.mesh_device

    def load_model(self, kernel_ready_queue=None):
        from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
        from models.tt_dit.utils.ltx import default_ltx_checkpoint, default_ltx_gemma

        # Read by _resolve_quant_config at pipeline construction.
        os.environ["LTX_QUANT"] = self.config.quant
        os.environ.setdefault("TT_DIT_CACHE_DIR", self.config.tt_dit_cache_dir)

        checkpoint = default_ltx_checkpoint(self.config.model_name)
        gemma = default_ltx_gemma()

        self.logger.info("Creating LTX-2.3 distilled pipeline...")
        self.logger.info(f"  checkpoint: {checkpoint}")
        self.logger.info(f"  gemma: {gemma}")
        self.logger.info(f"  size: {self.config.width}x{self.config.height}, frames: {self.config.num_frames}")
        self.logger.info(f"  quant: {self.config.quant}, use_trace: {self.config.use_trace}")
        self.logger.info(f"  tt_dit_cache: {os.environ['TT_DIT_CACHE_DIR']}")

        # Geometry is fixed here (the upsampler pins its GroupNorm to T*H*W).
        # sp_axis/tp_axis/num_links/dynamic_load/topology/is_fsdp are deliberately
        # omitted so create_pipeline resolves them from the Blackhole (2,2) preset;
        # the resolved values are logged below to prove the preset fired.
        # run_warmup compiles kernels up front so the first request is not charged
        # for JIT compilation.
        self.pipeline = LTXDistilledPipeline.create_pipeline(
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
            "LTX pipeline created; resolved preset: "
            f"sp_axis={pc.sequence_parallel.mesh_axis} tp_axis={pc.tensor_parallel.mesh_axis} "
            f"sp={pc.sequence_parallel.factor} tp={pc.tensor_parallel.factor} "
            f"dynamic_load={getattr(self.pipeline, 'dynamic_load', None)} "
            f"is_fsdp={getattr(self.pipeline, 'is_fsdp', None)}"
        )
        # The distilled pipeline runs without CFG and encodes only the positive
        # prompt, so a negative prompt cannot influence it.
        self.logger.warning("negative_prompt is inert on the distilled pipeline (no CFG); it is accepted and ignored")

        if kernel_ready_queue is not None:
            kernel_ready_queue.put(self.worker_id)

    def run_inference(self, requests, on_event=None) -> list:
        """Generate one AV clip. Returns [mp4_bytes].

        Unlike the other runners this returns encoded MP4 bytes rather than an
        ndarray of frames, because the clip carries synchronized audio. See the
        class docstring.
        """
        request = requests[0]

        prompt = request["prompt"]
        if not prompt or not prompt.strip():
            raise ValueError("LTX generation requires a non-empty prompt")
        seed = request.get("seed")
        seed = int(seed) if seed is not None else 0

        if request.get("negative_prompt"):
            self.logger.info("negative_prompt supplied but ignored (distilled pipeline has no CFG)")

        # Reject a shape that does not match what the pipeline was built for, rather
        # than letting the upsampler's pinned GroupNorm assert mid-generation.
        for key, pinned in (
            ("num_frames", self.config.num_frames),
            ("height", self.config.height),
            ("width", self.config.width),
        ):
            got = request.get(key)
            if got is not None and int(got) != pinned:
                raise ValueError(
                    f"{key}={got} does not match this server's pinned shape "
                    f"({self.config.width}x{self.config.height}, {self.config.num_frames} frames). "
                    "Restart the server with the shape you want."
                )

        self.logger.info(
            f"Running AV inference: prompt='{prompt[:80]}', "
            f"size={self.config.width}x{self.config.height}, frames={self.config.num_frames}, seed={seed}"
        )

        kwargs = dict(
            output_type="rgb",
            num_frames=self.config.num_frames,
            height=self.config.height,
            width=self.config.width,
            seed=seed,
            fps=self.config.fps,
        )
        if on_event is not None:
            kwargs["on_event"] = on_event

        # generate(output_path=...) muxes video + audio and returns the path.
        # NamedTemporaryFile(delete=False) so the pipeline can reopen it by name.
        tmp = tempfile.NamedTemporaryFile(prefix=f"ltx_w{self.worker_id}_", suffix=".mp4", delete=False)
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

        self.logger.info(f"AV inference complete: {len(data)} bytes of MP4")
        return [data]

    def close_device(self):
        if self.pipeline is not None:
            try:
                self.pipeline.release_traces()
            except Exception as e:
                self.logger.warning(f"release_traces failed: {e}")
            self.pipeline = None

        for mesh in (self.mesh_device, self.parent_mesh):
            if mesh is not None:
                self.logger.info("Closing mesh device")
                try:
                    ttnn.close_mesh_device(mesh)
                except Exception as e:
                    self.logger.warning(f"close_mesh_device failed: {e}")
        self.mesh_device = None
        self.parent_mesh = None

        if self._fabric_config is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            self._fabric_config = None
