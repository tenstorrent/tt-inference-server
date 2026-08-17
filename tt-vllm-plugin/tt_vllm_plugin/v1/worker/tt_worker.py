# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from contextlib import suppress
from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.nn as nn
import ttnn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

from tt_vllm_plugin.v1.worker.tt_model_runner import TTModelRunner
from tt_vllm_plugin.v1.worker.tt_model_runner_pooling import TTModelRunnerPooling
from tt_vllm_plugin.model_loader.tt_loader import TTModelLoader
from tt_vllm_plugin.worker.tt_model_runner import TTModelInput
from tt_vllm_plugin.worker.tt_worker import (
    close_mesh_device,
    get_mesh_grid,
    get_num_available_blocks_tt,
    open_mesh_device,
)
from vllm.tasks import SupportedTask

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger("vllm.tt_vllm_plugin.v1.worker.tt_worker")
print("=== tt_worker.py module is being imported ===")
logger.info("=== tt_worker.py module is being imported ===")


class TTWorker(WorkerBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = True,
    ):
        logger.info("Initializing TT worker...")
        print("Initializing TT worker...")
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )

        # Initialized by init_device
        self.mesh_device = None
        self.model_config.override_tt_config = {}

        # Runtime weight-update (co-located RL) state. The transport is owned by
        # a WeightTransferEngine (device-socket bridge); the worker owns the
        # on-device apply + the authoritative weights-version counter. Both are
        # inert unless the co-located RL control plane calls the hooks below.
        self.weight_transfer_engine = None
        self._weights_version = 0

        # Whether to use ttnn tracing for model execution
        override_tt_config = self.model_config.override_tt_config
        trace_key = "trace_mode"
        self.trace_mode = True
        if override_tt_config and trace_key in override_tt_config:
            assert override_tt_config[trace_key] in [True, False], (
                f"Invalid {trace_key}: {override_tt_config[trace_key]}"
            )
            self.trace_mode = override_tt_config[trace_key]

    def init_device(self) -> None:
        logger.info("Initializing TT device...")
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        if dp_rank == 0:
            # Pass model_config to device_params_from_override_tt_config for BGE detection
            self.mesh_device = open_mesh_device(
                self.model_config.override_tt_config,
                self.trace_mode,
                dp_rank,
                self.model_config,
            )
            self.device_config.device = self.mesh_device
            assert self.mesh_device is not None
            self.device_config.num_devices = self.mesh_device.get_num_devices()
        else:
            mesh_grid = get_mesh_grid(dp_rank)
            self.mesh_device = None
            # Num devices is required for determining num blocks in KV cache.
            self.device_config.num_devices = mesh_grid[0] * mesh_grid[1]
        # Init ModelRunner here, so that we have access to self.mesh_device.
        # We'll determine the runner type after loading the model in load_model()
        # For now, create a placeholder that will be replaced
        self.model_runner: Optional[Union[TTModelRunner, TTModelRunnerPooling]] = None

    def load_model(self):
        # Only DP rank 0 loads the model
        if self.vllm_config.parallel_config.data_parallel_rank == 0:
            # First, load the model to determine its type
            loader = TTModelLoader(self.load_config)
            model = loader.load_model(
                vllm_config=self.vllm_config, model_config=self.model_config
            )

            # Detect if this is a pooling model
            # Check if model has forward() but not prefill_forward()/decode_forward()
            # This is a heuristic - pooling models typically only have forward()
            is_pooling = (
                hasattr(model, "forward")
                and not (
                    hasattr(model, "prefill_forward")
                    and hasattr(model, "decode_forward")
                )
                and hasattr(model, "get_embedding_dim")
            )

            # Also check model_config.runner_type if set
            runner_type = getattr(self.model_config, "runner_type", None)
            if runner_type == "pooling":
                is_pooling = True
            elif runner_type == "generate":
                is_pooling = False

            # Create the appropriate runner
            if is_pooling:
                logger.info("Detected pooling model, using TTModelRunnerPooling")
                self.model_runner = TTModelRunnerPooling(
                    vllm_config=self.vllm_config,
                    mesh_device=self.mesh_device,
                    trace_mode=self.trace_mode,
                )
                # Set the model directly (skip loader.load_model in runner)
                self.model_runner.model = model
            else:
                logger.info("Detected generation model, using TTModelRunner")
                self.model_runner = TTModelRunner(
                    vllm_config=self.vllm_config,
                    mesh_device=self.mesh_device,
                    trace_mode=self.trace_mode,
                )
                # Set the model directly (skip loader.load_model in runner)
                self.model_runner.model = model
                # Generation models still need KV cache initialization
                # This will be called later in initialize_from_config
        else:
            # For non-DP rank 0, we still need to create a runner placeholder
            # The actual model won't be loaded, but the runner structure is needed
            runner_type = getattr(self.model_config, "runner_type", "generate")
            if runner_type == "pooling":
                self.model_runner = TTModelRunnerPooling(
                    vllm_config=self.vllm_config,
                    mesh_device=self.mesh_device,
                    trace_mode=self.trace_mode,
                )
            else:
                self.model_runner = TTModelRunner(
                    vllm_config=self.vllm_config,
                    mesh_device=self.mesh_device,
                    trace_mode=self.trace_mode,
                )

    # ---- Runtime weight update over a device socket (RL rollouts) ----
    #
    # These hooks implement vLLM's *native* RL weight-sync worker contract
    # (``init_weight_transfer_engine`` / ``update_weights``), so the co-located
    # trainer can drive updates through vLLM's own endpoints
    # (``/init_weight_transfer_engine``, ``/update_weights``) + pause/resume
    # instead of a bespoke control plane. Both are dispatched to every worker in
    # the (DP) group via ``engine_client.collective_rpc(...)`` (which runs
    # between engine steps, so a swap never interleaves an ``execute_model``);
    # only the rank that owns the on-device model (DP rank 0 on TT) does work,
    # other ranks no-op so the collective returns uniformly.
    #
    # Transport is owned by a ``WeightTransferEngine`` -- here the TT
    # ``device_socket`` backend, which wraps tt-metal's ``WeightBridge`` over a
    # ``ttnn.MeshSocket`` (tt-train/.../grpo/utils/inference_bridge.py). The
    # trainer (tt-training-service) owns a *separate* mesh and is the bridge
    # *sender* (role="ttml"); this worker is the *receiver* (role="ttt").
    # ``recv_state()`` yields a HF-keyed dict of on-device ``ttnn.Tensor``
    # handles; this worker applies it via ``Generator.update_weights(hf_dict,
    # hf_rope=...)`` -- an in-place ``ttnn.copy`` per weight that preserves each
    # device buffer address so captured decode traces stay valid.
    #
    # Deployment prerequisites (established at launch, TT_COLOCATED_INFERENCE=1):
    #   * Trainer + inference share ONE MPI world (co-launched via ``tt-run``);
    #     the bridge pins TTML_RANK=0 (sender) / TTT_RANK=1 (receiver).
    #   * Fabric enabled (FABRIC_2D) before the mesh device is opened.
    #   * Weights replicated, DRAM-interleaved, TILE, bfloat16 (DDP-only on the
    #     trainer, no TP, for now).

    def _owns_model(self) -> bool:
        runner = getattr(self, "model_runner", None)
        return runner is not None and getattr(runner, "model", None) is not None

    def _colocated_rl_only(self, what: str) -> None:
        """Fast-fail off the co-located RL path.

        On a normal (non-colocated) server there is no trainer peer, so the
        device-socket rendezvous would block forever. The native routes are
        also only mounted on the RL path; this guards a stray/reachable call.
        """
        if os.getenv("TT_COLOCATED_INFERENCE") != "1":
            raise RuntimeError(
                f"{what} is only available on a co-located RL inference server "
                "(TT_COLOCATED_INFERENCE=1)."
            )

    def _ensure_weight_transfer_engine(self):
        """Lazily construct the TT ``device_socket`` weight-transfer engine.

        Prefer the engine already attached by vLLM (from
        ``--weight-transfer-config '{"backend": "device_socket"}'``); otherwise
        build one directly so the native ``/update_weights`` path works without
        threading a ``WeightTransferConfig`` through vLLM's CLI (whose
        ``backend`` is a closed ``Literal["nccl", "ipc"]``).
        """
        if self.weight_transfer_engine is not None:
            return self.weight_transfer_engine

        from tt_vllm_plugin.weight_transfer.tt_device_socket_engine import (
            TTDeviceSocketWeightTransferEngine,
        )

        config = getattr(self.vllm_config, "weight_transfer_config", None)
        self.weight_transfer_engine = TTDeviceSocketWeightTransferEngine(
            config, self.vllm_config.parallel_config
        )
        return self.weight_transfer_engine

    def init_weight_transfer_engine(self, init_info: dict) -> dict:
        """Native RL hook: stand up the device-socket transport.

        Called once by the trainer (via ``/init_weight_transfer_engine`` ->
        ``collective_rpc``) before the training loop. On TT this constructs +
        connects the receiver ``WeightBridge`` (the MPI handshake + MeshSocket
        descriptor exchange); it blocks until the trainer also reaches
        ``connect()``. Non-owning DP ranks have no device model / mesh, so they
        no-op and let the collective return uniformly.
        """
        self._colocated_rl_only("Weight transfer")
        rank = self.vllm_config.parallel_config.data_parallel_rank
        if not self._owns_model():
            return {"rank": rank, "initialized": False}

        engine = self._ensure_weight_transfer_engine()
        # Prefer the bridge dir threaded through additional_config (survives the
        # EngineCore's curated env, unlike a bare TT_WEIGHT_BRIDGE_DIR).
        override_tt_config = getattr(self.model_config, "override_tt_config", None) or {}
        engine.bind_runtime(
            device=self.mesh_device,
            bridge_dir=override_tt_config.get("tt_weight_bridge_dir"),
        )
        typed_init_info = engine.parse_init_info(init_info or {})
        engine.init_transfer_engine(typed_init_info)
        logger.info("Weight transfer engine initialized (rank %s)", rank)
        return {"rank": rank, "initialized": True}

    def update_weights(self, update_info: dict) -> dict:
        """Native RL hook: in-place replace on-device weights over the bridge.

        Signature matches vLLM's worker contract
        (``update_weights(update_info: dict)``, dispatched via
        ``collective_rpc``). Transport (``recv_state`` + barrier) is owned by
        the ``WeightTransferEngine``; this worker owns the on-device apply
        (``Generator.update_weights`` in-place ``ttnn.copy``), the decode-trace
        release, and the authoritative weights-version counter.

        Callers should quiesce inference first (vLLM ``/pause?mode=wait`` or the
        legacy admission gate) so no request spans the version boundary.
        """
        self._colocated_rl_only("Runtime weight update")
        rank = self.vllm_config.parallel_config.data_parallel_rank

        if not self._owns_model():
            # Non-owning DP ranks have no device model; nothing to do.
            return {"rank": rank, "updated": False, "version": self._weights_version}

        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "update_weights called before init_weight_transfer_engine(); "
                "the device-socket bridge has not been connected."
            )

        model = self.model_runner.get_model()
        if not hasattr(model, "update_weights"):
            raise NotImplementedError(
                f"Model {type(model).__name__} does not implement "
                "update_weights(hf_state_dict, hf_rope=...). Runtime weight "
                "update requires the tt-metal Generator.update_weights "
                "passthrough + Transformer.update_weights in-place API."
            )

        typed_update_info = self.weight_transfer_engine.parse_update_info(
            update_info or {}
        )

        # Drop any captured decode trace before the transfer: recv_state()
        # allocates the full state dict as fresh device buffers, and doing that
        # while a decode trace holds DRAM/L1 wedges the on-device CCL recv
        # (device timeout). tt-transformers re-captures the trace lazily on the
        # next generation. Guarded on hasattr for older Generators.
        ttnn.synchronize_device(self.mesh_device)
        if hasattr(model, "release_decode_traces"):
            model.release_decode_traces()

        # Apply callback handed to the engine's transport. Receives the HF-keyed
        # dict of on-device ttnn tensors from recv_state(). An empty dict is the
        # plumbing-test payload (SIM_PAYLOAD=empty): model.update_weights is
        # strict (KeyError on missing keys), so no-op the apply while still
        # bumping the version + running the barrier inside receive_weights().
        applied = {"weights": False}

        def _apply_ttnn_weights(hf_dict) -> None:
            if hf_dict:
                model.update_weights(hf_dict, hf_rope=typed_update_info.hf_rope)
                applied["weights"] = True
            else:
                logger.info(
                    "Received empty weight payload (plumbing test); skipping "
                    "model.update_weights and applying a no-op version bump."
                )

        self.weight_transfer_engine.receive_weights(
            typed_update_info,
            load_weights=_apply_ttnn_weights,
        )

        # The worker owns the version counter (single source of truth).
        self._weights_version += 1
        logger.info(
            "Weight update complete; weights_version=%s weights_applied=%s",
            self._weights_version,
            applied["weights"],
        )
        return {
            "rank": rank,
            "updated": True,
            "weights_applied": applied["weights"],
            "version": self._weights_version,
        }

    def get_weights_version(self) -> int:
        """Return the current on-device weights/policy version."""
        self._colocated_rl_only("Weights versioning")
        return self._weights_version

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        For the GPU/TPU backends, this method generates the KVCacheSpec by
        parsing the kv cache format from each Attention module in the static
        forward context (compilation_config.static_forward_context).
        core/kv_cache_utils.py uses the KVCacheSpec along with available
        memory info from a profiling run to determine num blocks.

        For the TT backend, the static forward context is not populated since
        the modelling code is independent so we currently skip creating a
        kv cache spec for each layer, similar to the Spyre/Neuron backends.
        Currently we also don't run profiling to determine available memory.

        Return a dummy single layer KVCacheSpec and in the
        determine_available_memory function override num blocks using
        self.cache_config.num_gpu_blocks_override.
        """

        # TODO: Once we're able to populate a static forward context,
        # generate separate specs per layer (e.g. also sliding window, local
        # attention).

        model_config = self.model_config
        parallel_config = self.parallel_config
        cache_config = self.cache_config

        # Excludes TP factor since that is handled on the model side for TT.
        total_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        dtype = (
            model_config.dtype
            if cache_config.cache_dtype == "auto"
            else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        )

        attn_spec = FullAttentionSpec(
            block_size=cache_config.block_size,
            num_kv_heads=total_num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            use_mla=model_config.use_mla,
            sliding_window=model_config.get_sliding_window(),
        )
        kv_cache_spec: dict[str, KVCacheSpec] = {"foo": attn_spec}
        return kv_cache_spec

    def determine_available_memory(self) -> int:

        num_tt_blocks = get_num_available_blocks_tt(self.vllm_config)
        self.cache_config.num_gpu_blocks_override = num_tt_blocks

        # page_size_bytes of the single dummy spec we hand vLLM in
        # get_kv_cache_spec(); this is exactly the per-block divisor vLLM uses.
        kv_cache_spec = self.get_kv_cache_spec()
        page_size_bytes = next(iter(kv_cache_spec.values())).page_size_bytes
        return num_tt_blocks * page_size_bytes

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate TT KV cache (only DP rank 0) and initialize persistent
        input batch (all DP ranks) with the specified kv_cache_config.
        Pooling models don't need KV cache, so skip initialization for them.
        """
        if isinstance(self.model_runner, TTModelRunnerPooling):
            # Pooling models don't need KV cache
            logger.info("Skipping KV cache initialization for pooling model")
            return
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        # Cache is already initialized in initialize_from_config.
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def compile_or_warm_up_model(self) -> None:
        # Currently skip and compile/capture-trace during the first execution.
        pass

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        assert self.is_driver_worker, "There should only be one Worker for TT"
        assert self.model_runner is not None, (
            "Model runner not initialized. Call load_model() first."
        )
        output = self.model_runner.execute_model(scheduler_output)
        return output

    def check_health(self) -> None:
        # Worker will always be healthy as long as it's running.
        return

    # ---- DP gather hooks called by DPEngineCoreProc in core.py ----

    def build_dp_model_input(
        self, scheduler_output: Optional["SchedulerOutput"]
    ) -> tuple[Optional[TTModelInput], int]:
        """Called by each DP rank to build model input from scheduler output.
        Pooling models don't use TTModelInput, so return None.
        """
        assert self.model_runner is not None, (
            "Model runner not initialized. Call load_model() first."
        )
        if isinstance(self.model_runner, TTModelRunnerPooling):
            # Pooling models handle input preparation internally
            return None, 0
        model_input = None
        if scheduler_output is not None:
            model_input = self.model_runner.build_model_input(scheduler_output)
        max_blocks = model_input.block_tables.shape[1] if model_input else 0
        return model_input, max_blocks

    def build_dp_decode_gather_input(
        self, model_input: Optional[TTModelInput], max_blocks_decode_batch: int
    ) -> dict[str, torch.Tensor]:
        assert self.model_runner is not None, (
            "Model runner not initialized. Call load_model() first."
        )
        if isinstance(self.model_runner, TTModelRunnerPooling):
            # Pooling models don't use decode gather input
            return {"int_inputs": torch.tensor([]), "float_inputs": torch.tensor([])}
        return self.model_runner.build_dp_decode_gather_input(
            model_input, max_blocks_decode_batch
        )

    def concat_and_execute_dp(
        self,
        inputs: Union[list[Optional[TTModelInput]], dict[str, torch.Tensor]],
        is_decode: bool,
        max_blocks_decode_batch: Optional[int],
    ) -> torch.Tensor:
        """Called only by DP rank 0 to concatenate DP-sized inputs and execute.
        Returns a stacked tensor [world, max_num_seqs, 1] of sampled ids.
        Each DP slice is right-padded with zeros to max_num_seqs; empty entries
        are zeros. Same behavior for both prefill and decode.

        For pooling models, this is not used as they don't support DP yet.
        """
        assert self.model_runner is not None, (
            "Model runner not initialized. Call load_model() first."
        )
        if isinstance(self.model_runner, TTModelRunnerPooling):
            # Pooling models don't support DP yet
            # Return empty tensor for now
            world = self.vllm_config.parallel_config.data_parallel_size
            B = int(self.model_runner.scheduler_config.max_num_seqs)
            return torch.zeros((world, B, 1), dtype=torch.int32)

        assert self.vllm_config.parallel_config.data_parallel_rank == 0, (
            "concat_and_execute_dp must run on DP rank 0"
        )
        assert self.is_driver_worker, "concat_and_execute_dp must run on driver"
        merged = self.model_runner.concat_dp_model_inputs(
            inputs, is_decode, max_blocks_decode_batch
        )
        sampled_token_ids_per_dp: list[torch.Tensor] = (
            self.model_runner.execute_with_model_input(merged)
        )

        # Pad each DP result to uniform shape for tensor all_gather.
        world = self.vllm_config.parallel_config.data_parallel_size
        assert len(sampled_token_ids_per_dp) == world
        B = int(self.model_runner.scheduler_config.max_num_seqs)
        for dp_rank in range(world):
            token_ids = sampled_token_ids_per_dp[dp_rank].to(torch.int32)
            if token_ids.numel() == 0:
                token_ids = torch.zeros((B, 1), dtype=torch.int32)
            else:
                assert token_ids.dim() == 2 and token_ids.shape[1] == 1, (
                    "Currently only supporting 1 output token per request"
                )
                pad_rows = B - token_ids.shape[0]
                if pad_rows > 0:
                    token_ids = torch.cat(
                        [
                            token_ids,
                            torch.zeros(
                                (pad_rows, token_ids.shape[1]), dtype=torch.int32
                            ),
                        ],
                        dim=0,
                    )
            sampled_token_ids_per_dp[dp_rank] = token_ids
        return torch.stack(sampled_token_ids_per_dp)  # [world, B, 1]

    def apply_dp_execution_result(
        self, sampled_token_ids: torch.Tensor
    ) -> ModelRunnerOutput:
        """Called by each DP rank to apply sampled tokens to internal caches.
        Pooling models don't use this method.
        """
        assert self.model_runner is not None, (
            "Model runner not initialized. Call load_model() first."
        )
        if isinstance(self.model_runner, TTModelRunnerPooling):
            # Pooling models don't support DP yet
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )
        # Trim to active local batch size to drop padding rows.
        num_reqs = self.model_runner.input_batch.num_reqs
        sampled_token_ids = sampled_token_ids[:num_reqs]
        return self.model_runner.generate_runner_output(sampled_token_ids)

    # ---- Destructor (used to close devices) ----

    def __del__(self):
        # Delete model runner first in case there are model artifacts
        with suppress(AttributeError):
            # attributes may be already torn down when destructor is called
            del self.model_runner

            if self.mesh_device:
                close_mesh_device(
                    self.mesh_device, self.model_config.override_tt_config
                )
                del self.mesh_device

        if hasattr(super(), "__del__"):
            super().__del__()  # type: ignore

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Get supported tasks by delegating to the model runner."""
        assert self.model_runner is not None, (
            "Model runner not initialized. Call load_model() first."
        )
        return self.model_runner.get_supported_tasks()

    def get_model(self) -> nn.Module:
        """Get the underlying model."""
        assert self.model_runner is not None, (
            "Model runner not initialized. Call load_model() first."
        )
        return self.model_runner.get_model()
