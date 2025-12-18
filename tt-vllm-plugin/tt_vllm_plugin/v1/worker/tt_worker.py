# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import suppress
from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.nn as nn

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
        """
        For the GPU/TPU backends, this method runs profiling to determine
        available memory for the KV cache. The available memory is then used
        in conjunction with the output of get_kv_cache_spec to determine
        the number of kv cache blocks (total memory / page_size / num layers).

        Currenly we just return a large dummy number of bytes similar to the
        Spyre/Neuron backends and override the number of kv cache blocks.
        """

        # TODO: Once we can run profiling, return real available memory
        # instead of overriding the number of blocks.
        num_tt_blocks = get_num_available_blocks_tt(self.vllm_config)
        self.cache_config.num_gpu_blocks_override = num_tt_blocks
        return 1 << 64

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
