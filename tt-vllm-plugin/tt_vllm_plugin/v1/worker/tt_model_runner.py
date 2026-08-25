# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING, Any, Optional

import torch
import ttnn
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalKwargsItems as MultiModalKwargs
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    LogprobsTensors,
    ModelRunnerOutput,
)

from tt_vllm_plugin.model_loader.tt_loader import TTModelLoader
from tt_vllm_plugin.platform import TTPlatform
from tt_vllm_plugin.v1.worker.tt_input_batch import CachedRequestState, InputBatch
from tt_vllm_plugin.worker.tt_model_runner import (
    TTModelInput,
    TTSamplingParams,
    sample_tokens,
)
from tt_vllm_plugin.worker.tt_worker import (
    effective_data_parallel,
    get_inprocess_data_parallel,
)

from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.model_executor.models.interfaces_base import (
    is_pooling_model,
    is_text_generation_model,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

import numpy as np

logger = init_logger("vllm.tt_vllm_plugin.v1.worker.tt_model_runner")


def compute_sampled_logprobs(
    logits: torch.Tensor,
    sampled_ids: torch.Tensor,
    num_logprobs: int,
) -> LogprobsTensors:
    """Log-softmax logprobs for the sampled tokens, plus the top ``num_logprobs``.

    ``logits`` must be the model's RAW logits, i.e. before the temperature /
    top-k / top-p warping ``sample_tokens`` applies. Those knobs shape which
    token gets *drawn*; they are deliberately not applied to the value the token
    is *scored* under. Async-RL importance sampling divides this behavior
    logprob by one the trainer recomputes from its own raw logits, so the two
    sides must share a basis or the ratio is biased.

    Column 0 of the returned tensors is always the sampled token, matching the
    vLLM convention that the sampled token precedes the top-k alternatives.
    """
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    sampled = sampled_ids.view(-1, 1).long()
    sampled_logprobs = logprobs.gather(-1, sampled)

    # 0-indexed rank of the sampled token: how many tokens are strictly likelier.
    ranks = (logprobs > sampled_logprobs).sum(dim=-1).to(torch.int32)

    if num_logprobs > 0:
        k = min(num_logprobs, logprobs.shape[-1])
        topk_logprobs, topk_ids = torch.topk(logprobs, k, dim=-1)
        token_ids = torch.cat([sampled.to(torch.int32), topk_ids.to(torch.int32)], -1)
        values = torch.cat([sampled_logprobs, topk_logprobs], dim=-1)
    else:
        token_ids = sampled.to(torch.int32)
        values = sampled_logprobs

    return LogprobsTensors(
        logprob_token_ids=token_ids,
        logprobs=values,
        selected_token_ranks=ranks,
    )


def empty_logprobs(num_rows: int, width: int) -> LogprobsTensors:
    """Zeroed logprob tensors to scatter per-replica results into."""
    return LogprobsTensors(
        logprob_token_ids=torch.zeros((num_rows, width), dtype=torch.int32),
        logprobs=torch.zeros((num_rows, width), dtype=torch.float32),
        selected_token_ranks=torch.zeros(num_rows, dtype=torch.int32),
    )


def scatter_logprobs(
    dst: LogprobsTensors, src: LogprobsTensors, rows: "list[int] | torch.Tensor"
) -> None:
    """Copy every row of ``src`` into the ``rows`` positions of ``dst``."""
    dst.logprob_token_ids[rows] = src.logprob_token_ids
    dst.logprobs[rows] = src.logprobs
    dst.selected_token_ranks[rows] = src.selected_token_ranks


def slice_logprobs(
    logprobs: Optional[LogprobsTensors], start: int, end: int
) -> Optional[LogprobsTensors]:
    """Row-slice logprob tensors, propagating None."""
    if logprobs is None:
        return None
    return LogprobsTensors(
        logprob_token_ids=logprobs.logprob_token_ids[start:end],
        logprobs=logprobs.logprobs[start:end],
        selected_token_ranks=logprobs.selected_token_ranks[start:end],
    )


class TTModelRunner:
    def __init__(
        self,
        vllm_config: VllmConfig,
        mesh_device: ttnn.MeshDevice,
        trace_mode: bool,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        # self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config

        # Because of multiprocessing, the config-dependent
        # class attributes might not have been set in this process,
        # so we need to call this again.
        TTPlatform.check_and_update_config(vllm_config)

        # Currently, TT model runner doesn't support chunked prefill.
        assert self.scheduler_config.chunked_prefill_enabled is False

        self.mesh_device = mesh_device
        self.trace_mode = trace_mode

        # Whether to sample on device
        self.sample_on_device_mode = TTPlatform.sample_on_device_mode

        logger.info(
            "TTModelRunner: trace_mode=%s, sample_on_device_mode=%s",
            self.trace_mode,
            self.sample_on_device_mode,
        )

        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}

        # Cached request states. Request states are tracked in the runner so
        # they don't need to be re-sent every scheduling step. For requests
        # that have been scheduled before, only the diff is received from
        # the scheduler output.
        self.requests: dict[str, CachedRequestState] = {}

        # ---- In-process (submesh) data-parallel ------------------------------
        # When enabled (override_tt_config["tt_data_parallel"] > 1), a single
        # EngineCore process owns all N submesh replicas and one scheduler; each
        # step's batch is fanned out across the replicas here in the runner.
        # Every other mode leaves _tt_dp == 1 and takes the unchanged code path.
        self._tt_dp = get_inprocess_data_parallel(vllm_config)
        if self._tt_dp > 1:
            assert self.parallel_config.data_parallel_size == 1, (
                "in-process submesh DP requires vLLM data_parallel_size == 1"
            )
            assert self.scheduler_config.max_num_seqs % self._tt_dp == 0, (
                f"max_num_seqs ({self.scheduler_config.max_num_seqs}) must be a "
                f"multiple of tt_data_parallel ({self._tt_dp}) so each submesh "
                "replica gets an equal per-replica batch width."
            )
            # Stable request -> submesh replica pinning. Assigned once when a
            # request first appears and held for its lifetime so its paged KV
            # (written on replica m during prefill) is always read back on the
            # same replica, independent of vLLM slot condensation.
            self._req_to_mesh: dict[str, int] = {}
        # Per-replica batch width. Equal to max_num_seqs when in-process DP is
        # off, so decode padding / stride references below are byte-identical
        # for the single-device and vLLM process-DP paths.
        self._per_mesh_max_seqs = self.scheduler_config.max_num_seqs // self._tt_dp
        self._decode_batch_width = self._per_mesh_max_seqs

    def load_model(self) -> None:
        logger.info("Loading TT model...")
        loader = TTModelLoader(self.load_config)
        self.model = loader.load_model(
            vllm_config=self.vllm_config, model_config=self.model_config
        )

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """

        kv_cache_groups = kv_cache_config.kv_cache_groups
        if len(kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not supported yet."
            )
        if isinstance(kv_cache_groups[0].kv_cache_spec, AttentionSpec):
            kv_cache_spec = kv_cache_groups[0].kv_cache_spec
        else:
            raise TypeError("Expected AttentionSpec")

        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            assert len(kv_cache_tensor.shared_by) == 1, (
                "KV cache shared by multiple layers is not supported for TT"
            )

        # Initialize persistent input batch with block size from kv_cache_spec.
        # The persistent batch optimization reduces overhead between steps
        # when consecutive batches contain mostly the same requests.
        max_num_reqs = self.scheduler_config.max_num_seqs
        max_model_len = self.model_config.max_model_len
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        self.input_batch = InputBatch(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            block_sizes=[kv_cache_spec.block_size],
        )

        # The block tables in the persistent input batch have
        # max_num_blocks_per_req = cdiv(max_model_len, block_size) but this
        # does not take into account num blocks in KV cache. Actual max is min
        # of these two. Used to slice block tables during input prep.
        self.max_num_blocks_per_req = min(
            cdiv(self.model_config.max_model_len, self.cache_config.block_size),
            kv_cache_config.num_blocks,
        )

        # Only DP rank 0 allocates KV cache
        if self.parallel_config.data_parallel_rank != 0:
            return

        # Make the assumption that we are tensor parallel by
        # min(number of devices, number of KV heads).
        # TODO: move this into model.allocate_kv_cache.
        model_config = self.model_config
        # Effective replica count so per-submesh device count / KV shape is
        # correct for both process DP and in-process submesh DP.
        data_parallel = effective_data_parallel(self.vllm_config)
        num_devices = self.device_config.num_devices // data_parallel
        total_kv_heads = kv_cache_spec.num_kv_heads
        num_kv_heads = total_kv_heads // min(num_devices, total_kv_heads)

        kv_cache_shape = (
            kv_cache_config.num_blocks,
            num_kv_heads,
            kv_cache_spec.block_size,
            kv_cache_spec.head_size,
        )
        dtype = kv_cache_spec.dtype
        num_layers = model_config.get_num_layers_by_block_type(
            self.parallel_config, "attention"
        )

        # Allocate KV cache tensors.
        self.kv_caches = self.model.allocate_kv_cache(kv_cache_shape, dtype, num_layers)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the
        scheduler output.
        The updated states are used in `_prepare_model_inputs` to create the
        input tensors for the model.
        Based on _update_states for GPU/TPU backends.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs. Upstream switched the encoder
        # cache to a flat mm_hash -> output map
        # (SchedulerOutput.free_encoder_mm_hashes); TT is text-only so this
        # is effectively a no-op, but must use the new field/key.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            # Generation models require sampling_params, pooling models don't
            if new_req_data.sampling_params is None:
                raise ValueError(
                    "Generation models require sampling_params. "
                    "For pooling models, use TTModelRunnerPooling instead."
                )
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                sampling_params=sampling_params,
                pooling_params=None,
                generator=None,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
                mm_features=getattr(new_req_data, "mm_features", []),
            )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                assert new_block_ids is not None
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

    def _validate_mm_input(self, mm_input: MultiModalKwargs) -> None:
        """Validate multi-modal input supports only single images."""
        if list(mm_input.modalities) != ["image"]:
            raise NotImplementedError("Only images are supported for now")
        assert mm_input.get_item_count("image") == 1, (
            "Request can contain multiple inputs, \
            but each input can contain only one image!"
        )

    def _gather_multi_modal_inputs(self, scheduler_output) -> dict:
        """
        Gather and batch multi-modal inputs from scheduled requests.
        #TODO: Currently only supports image inputs in the "pixel_values" field.

        Creates a list of pixel values for each request.
        Example:
        [
          None, # for requests without mm_inputs
          [pixel_values_1], # with single mm_input
          [pixel_values_2, pixel_values_3, ...], # with multiple mm_inputs
        ]
        """

        multi_modal_kwargs: MultiModalKwargs = {"pixel_values": []}

        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            req_state = self.requests[req_id]

            if not req_state.mm_inputs:
                multi_modal_kwargs["pixel_values"].append(None)
                continue

            pv_array = []
            for mm_input in req_state.mm_inputs:
                self._validate_mm_input(mm_input)
                pv_array.append(mm_input["pixel_values"])

            multi_modal_kwargs["pixel_values"].append(pv_array)

        return multi_modal_kwargs

    def _prepare_model_inputs(
        self, scheduler_output: "SchedulerOutput"
    ) -> TTModelInput:
        """
        Prepare model inputs, supporting mixed prefill/decode batches.
        """
        assert scheduler_output.total_num_scheduled_tokens > 0
        input_batch = self.input_batch
        num_reqs = input_batch.num_reqs
        assert num_reqs > 0
        assert len(input_batch.block_table.block_tables) == 1, (
            "Currently only supporting 1 KV cache group"
        )

        # Second dim of block table is (ceil(max_model_len / block_size)).
        # Slice to self.max_num_blocks_per_req which also takes into
        # account max num blocks in KV cache in case max KV blocks is smaller.
        # Constant shape is required for ttnn tracing to work.
        block_tables = input_batch.block_table[0].get_cpu_tensor()[
            :num_reqs, : self.max_num_blocks_per_req
        ]

        # DP optimization: don't send padding blocks if possible to reduce
        # overhead from gathering inputs to rank 0 and rely on DP concat
        # function to pad to global max blocks.
        if self.parallel_config.data_parallel_size > 1:
            max_tokens_in_batch = max(input_batch.num_tokens[:num_reqs])
            max_blocks_in_batch = cdiv(
                max_tokens_in_batch, self.cache_config.block_size
            )
            block_tables = block_tables[:, :max_blocks_in_batch]

        # Determine which requests are prefill vs decode
        # In v1, new requests are prefill, cached requests are decode
        scheduled_new_req_ids = {
            req.req_id for req in scheduler_output.scheduled_new_reqs
        }
        is_prefill_list = []

        for req_idx in range(num_reqs):
            req_id = input_batch.req_ids[req_idx]
            is_prefill = req_id in scheduled_new_req_ids
            is_prefill_list.append(is_prefill)

        is_prefill_tensor = torch.tensor(is_prefill_list, dtype=torch.bool)
        num_prefill = is_prefill_tensor.sum().item()
        num_decode = num_reqs - num_prefill

        # Check if batch is mixed
        is_mixed = num_prefill > 0 and num_decode > 0

        if is_mixed:
            # Mixed batch: prepare unified tensors for prefill and decode
            prefill_indices = torch.where(is_prefill_tensor)[0]
            decode_indices = torch.where(~is_prefill_tensor)[0]

            # Get max prompt tokens for prefill requests
            if len(prefill_indices) > 0:
                max_prompt_tokens = max(
                    input_batch.num_prompt_tokens[i] for i in prefill_indices
                )
            else:
                max_prompt_tokens = 1

            # Create unified input_tokens tensor
            # Shape: [num_reqs, max(max_prompt_tokens, 1)]
            # For prefill: fill in prompt tokens
            # For decode: fill in single token at position 0, rest is padding
            max_seq_len = max(max_prompt_tokens, 1)
            input_tokens = torch.zeros((num_reqs, max_seq_len), dtype=torch.int32)

            # Fill prefill tokens
            if len(prefill_indices) > 0:
                prefill_tokens = input_batch.token_ids_cpu_tensor[
                    prefill_indices, :max_prompt_tokens
                ]
                input_tokens[prefill_indices, :max_prompt_tokens] = prefill_tokens

            # Fill decode tokens (single token at position 0)
            # Create input_positions: 0 for prefill, actual position for decode
            input_positions = torch.zeros(num_reqs, dtype=torch.int32)
            if len(decode_indices) > 0:
                decode_indices_np = decode_indices.numpy()
                decode_positions = torch.from_numpy(
                    input_batch.num_tokens[decode_indices_np] - 1
                )
                # Extract the last token for each decode request
                decode_tokens = input_batch.token_ids_cpu_tensor[
                    decode_indices, decode_positions
                ].view(-1, 1)
                input_tokens[decode_indices, 0:1] = decode_tokens
                input_positions[decode_indices] = decode_positions

            # Create prompt_lens list: prefill has actual length, decode has -1
            prompt_lens = [-1] * num_reqs
            for idx in prefill_indices:
                prompt_lens[idx] = int(input_batch.num_prompt_tokens[idx])

            # For multimodal, gather inputs (only prefill requests have mm inputs)
            is_prompt_for_mm = len(prefill_indices) > 0
        else:
            # Pure batch (all prefill or all decode) - use existing logic
            is_prompt = num_prefill > 0
            if is_prompt:
                input_positions = 0
                max_prompt_tokens = max(input_batch.num_prompt_tokens[:num_reqs])
                input_tokens = input_batch.token_ids_cpu_tensor[
                    :num_reqs, :max_prompt_tokens
                ]
                prompt_lens = input_batch.num_prompt_tokens[:num_reqs].tolist()
            else:
                input_positions = torch.from_numpy(
                    input_batch.num_tokens[:num_reqs] - 1
                )
                input_tokens = input_batch.token_ids_cpu_tensor[
                    torch.arange(num_reqs), input_positions
                ].view(-1, 1)
                prompt_lens = None

                # TODO: Remove once TT models can support arbitrary batch sizes.
                # Pad batch to max_num_reqs.
                if input_tokens.shape[0] < input_batch.max_num_reqs:
                    batch_pad = input_batch.max_num_reqs - input_tokens.shape[0]
                    input_tokens = torch.cat(
                        [input_tokens, torch.zeros(batch_pad, 1, dtype=torch.int32)]
                    )
                    # Pad positions with -1 to indicate no position
                    input_positions = torch.cat(
                        [input_positions, torch.ones(batch_pad, dtype=torch.int32) * -1]
                    )
                    block_tables = torch.cat(
                        [
                            block_tables,
                            torch.zeros(
                                batch_pad, block_tables.shape[1], dtype=torch.int32
                            ),
                        ]
                    )

            is_prompt_for_mm = is_prompt

        # Sampling-related.
        temperature = input_batch.sampling.temperature_cpu[:num_reqs]
        top_p = input_batch.sampling.top_p_cpu[:num_reqs]
        top_k = input_batch.sampling.top_k_cpu[:num_reqs]
        if not np.all(temperature == temperature[0]):
            logger.warning(
                "Currently only supporting same temperature for all "
                "sequences in batch, falling back to first sequence's "
                "temperature (%s)",
                temperature[0],
            )
        if not np.all(top_k == top_k[0]):
            logger.warning(
                "Currently only supporting same top_k"
                "for all sequences in batch, "
                "falling back to first sequence's top_k (%s)",
                top_k[0],
            )
        if not np.all(top_p == top_p[0]):
            logger.warning(
                "Currently only supporting same top_p"
                "for all sequences in batch, "
                "falling back to first sequence's top_p (%s)",
                top_p[0],
            )
        tt_sampling_params = TTSamplingParams(
            temperature=temperature[0],
            top_k=top_k[0],
            top_p=top_p[0],
        )

        compat_sampling_used = False
        sampling_metadata = None

        if self.model_config.is_multimodal_model and is_prompt_for_mm:
            # Gather multimodal inputs for prefill requests
            prefill_mm_kwargs = self._gather_multi_modal_inputs(scheduler_output)

            if is_mixed:
                # For mixed batches, create multimodal kwargs for all requests
                # Prefill requests get their mm inputs, decode requests get None
                multi_modal_kwargs: MultiModalKwargs = {"pixel_values": []}
                prefill_mm_idx = 0
                for req_idx in range(num_reqs):
                    if is_prefill_list[req_idx]:
                        # Prefill request: use gathered mm input
                        multi_modal_kwargs["pixel_values"].append(
                            prefill_mm_kwargs["pixel_values"][prefill_mm_idx]
                        )
                        prefill_mm_idx += 1
                    else:
                        # Decode request: no mm input
                        multi_modal_kwargs["pixel_values"].append(None)
            else:
                # Pure prefill batch: use gathered mm kwargs directly
                multi_modal_kwargs = prefill_mm_kwargs
        else:
            multi_modal_kwargs = {}

        return TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=prompt_lens,
            seq_groups=None,  # Not used in V1
            block_tables=block_tables,
            unpadded_batch_size=num_reqs,
            perform_device_sampling=None,  # currently unused in v1
            tt_sampling_params=tt_sampling_params,
            compat_sampling_used=compat_sampling_used,
            sampling_metadata=sampling_metadata,
            multi_modal_kwargs=multi_modal_kwargs,
            cross_block_tables=None,  # Not yet supported in V1
        )

    def build_model_input(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[TTModelInput]:
        """
        Update internal state with the scheduler output and build
        TTModelInput without executing the model.
        Returns None if there is no scheduled work in this step.

        For data parallel, this function is called by each DP rank to build
        TTModelInput from it's own scheduler output.
        """
        # Update cached state
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            return None

        # Prepare model inputs only
        model_input = self._prepare_model_inputs(scheduler_output)
        return model_input

    def build_dp_decode_gather_input(
        self,
        model_input: Optional[TTModelInput],
        max_blocks_decode_batch: int,
    ) -> dict[str, torch.Tensor]:
        """
        Called by each DP rank to build tensorized gather input for decode.
        max_blocks_decode_batch is the max blocks in the global DP batch.
        Returns dict[str, torch.Tensor] with keys:
          - "int_inputs": flattened int tensor of constant size.
          - "float_inputs": flattened float tensor of constant size.
        """

        if model_input is None:
            max_batch = int(self._decode_batch_width)
            tokens = torch.zeros((max_batch, 1), dtype=torch.int32)
            positions = torch.full((max_batch,), -1, dtype=torch.int32)
            block_tables = torch.zeros(
                (max_batch, max_blocks_decode_batch), dtype=torch.int32
            )
            unpadded_batch_size = torch.tensor([0], dtype=torch.int32)
            temperature = torch.tensor([-1.0], dtype=torch.float32)
            top_k = torch.tensor([-1], dtype=torch.int32)
            top_p = torch.tensor([-1.0], dtype=torch.float32)
        else:
            tokens = model_input.input_tokens
            positions = model_input.input_positions
            block_tables = model_input.block_tables
            # Pad block tables to max_blocks_decode_batch
            if block_tables.shape[1] < max_blocks_decode_batch:
                pad_w = max_blocks_decode_batch - block_tables.shape[1]
                block_tables = torch.cat(
                    [
                        block_tables,
                        torch.zeros(
                            (block_tables.shape[0], pad_w), dtype=block_tables.dtype
                        ),
                    ],
                    dim=1,
                )
            unpadded_batch_size = torch.tensor(
                [int(model_input.unpadded_batch_size)], dtype=torch.int32
            )
            sp = model_input.tt_sampling_params
            temperature = torch.tensor([float(sp.temperature)], dtype=torch.float32)
            top_k = torch.tensor([int(sp.top_k)], dtype=torch.int32)
            top_p = torch.tensor([float(sp.top_p)], dtype=torch.float32)

        # Pack into flattened tensors to reduce number of collectives.
        # B = max batch size, W = max_num_blocks_per_req.
        int_inputs = torch.cat(
            [
                tokens.contiguous().view(-1),  # B
                positions.contiguous().view(-1),  # B
                block_tables.contiguous().view(-1),  # B*W
                unpadded_batch_size.contiguous().view(-1),  # 1
                top_k.contiguous().view(-1),  # 1
            ],
            dim=0,
        ).contiguous()
        float_inputs = torch.cat(
            [
                temperature.contiguous().view(-1),  # 1
                top_p.contiguous().view(-1),  # 1
            ],
            dim=0,
        ).contiguous()

        return {
            "int_inputs": int_inputs,
            "float_inputs": float_inputs,
        }

    def concat_dp_model_inputs(
        self, inputs, is_decode: bool, max_blocks_decode_batch: Optional[int]
    ) -> "TTModelInput":
        """
        Concatenate a DP-sized set of inputs into a single TTModelInput.
        inputs can be either:
        - For prefill: list[Optional[TTModelInput]]
        - For decode (optimized gather): dict[str, torch.Tensor] with keys:
          - "int_inputs": stacked int32 tensor of shape [world, -1]
          - "float_inputs": stacked float32 tensor of shape [world, -1]
        """

        input_tokens_list: list[torch.Tensor] = []
        block_tables_list: list[torch.Tensor] = []
        input_positions_list: list[torch.Tensor] = []  # (decode only)
        prompt_lens_list: list[np.ndarray] = []  # (prefill only)
        batch_size_per_dp: list[int] = []
        sampling_params_per_dp: list[Optional[TTSamplingParams]] = []

        # Need to pad block tables to global max num blocks for constant shape.
        def pad_block_tables(block_tables):
            max_bt_width = self.max_num_blocks_per_req
            if block_tables.shape[1] < max_bt_width:
                pad_w = max_bt_width - block_tables.shape[1]
                block_tables = torch.cat(
                    [
                        block_tables,
                        torch.zeros(
                            (block_tables.shape[0], pad_w), dtype=block_tables.dtype
                        ),
                    ],
                    dim=1,
                )
            return block_tables

        if is_decode:
            # For decode, given gathered flattened tensors from all DP ranks.
            # Ints: [toks(B), positions(B), block_tables(B*W), bs(1), top_k(1)]
            # Floats: [temperature(1), top_p(1)]
            assert max_blocks_decode_batch is not None, (
                "max_blocks_decode_batch must be provided for decode"
            )
            B = int(self._decode_batch_width)
            W = max_blocks_decode_batch
            for int_inputs, float_inputs in zip(
                inputs["int_inputs"], inputs["float_inputs"]
            ):
                # Slices
                off = 0
                stride = B
                tokens = int_inputs[off : off + stride].view(B, 1)
                off += stride
                stride = B
                positions = int_inputs[off : off + stride].view(B)
                off += stride
                stride = B * W
                block_tables = int_inputs[off : off + stride].view(B, W)
                off += stride
                batch_size = int(int_inputs[off].item())
                off += 1
                top_k = int(int_inputs[off].item())
                off += 1
                temperature = float(float_inputs[0].item())
                top_p = float(float_inputs[1].item())

                input_tokens_list.append(tokens)
                input_positions_list.append(positions)
                block_tables_list.append(pad_block_tables(block_tables))
                batch_size_per_dp.append(batch_size)
                if batch_size > 0:
                    sampling_params_per_dp.append(
                        TTSamplingParams(
                            temperature=temperature, top_k=top_k, top_p=top_p
                        )
                    )
                else:
                    sampling_params_per_dp.append(None)

            input_positions = torch.cat(input_positions_list, dim=0)
            prompt_lens = None
        else:
            active_inputs: list[TTModelInput] = [mi for mi in inputs if mi]
            if not active_inputs:
                raise ValueError("All inputs are None; nothing to concatenate")

            # Determine max token width across slots.
            max_tok_width = 0
            for mi in active_inputs:
                assert mi.input_tokens.dim() == 2, "Input tokens must be 2D"
                max_tok_width = max(max_tok_width, mi.input_tokens.shape[1])
            assert max_tok_width > 0, "At least one input must have tokens"

            # Iterate over DP inputs and build segments for concatenation.
            for mi in inputs:
                # Skip None slots entirely. Decode path reconstructs full
                # inputs, so None should not occur there anymore.
                if mi is not None:
                    # Right-pad tokens and block tables to max widths
                    toks = mi.input_tokens
                    if not is_decode and toks.shape[1] < max_tok_width:
                        pad_w = max_tok_width - toks.shape[1]
                        toks = torch.cat(
                            [
                                toks,
                                torch.zeros((toks.shape[0], pad_w), dtype=toks.dtype),
                            ],
                            dim=1,
                        )
                    input_tokens_list.append(toks)
                    prompt_lens_list.append(mi.prompt_lens)
                    block_tables_list.append(pad_block_tables(mi.block_tables))

                batch_size_per_dp.append(mi.unpadded_batch_size if mi else 0)
                sampling_params_per_dp.append(mi.tt_sampling_params if mi else None)

            input_positions = 0
            prompt_lens = np.concatenate(prompt_lens_list, axis=0)

        input_tokens = torch.cat(input_tokens_list, dim=0)
        block_tables = torch.cat(block_tables_list, dim=0)

        compat_sampling_used = False
        sampling_metadata = None

        if self.model_config.is_multimodal_model and not is_decode:
            # Gather multi-modal inputs from all DP ranks
            multi_modal_kwargs: MultiModalKwargs = {"pixel_values": []}
            for mi in inputs:
                multi_modal_kwargs["pixel_values"].append(
                    mi.multi_modal_kwargs["pixel_values"]
                )
        else:
            multi_modal_kwargs = {}

        if os.environ.get("DP_GATHER_DEBUG") == "1":
            logger.info("batch_size_per_dp=%s", batch_size_per_dp)
        merged = TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=prompt_lens,
            seq_groups=None,
            block_tables=block_tables,
            unpadded_batch_size=batch_size_per_dp,
            perform_device_sampling=None,  # currently unused in v1
            tt_sampling_params=sampling_params_per_dp,
            compat_sampling_used=compat_sampling_used,
            sampling_metadata=sampling_metadata,
            multi_modal_kwargs=multi_modal_kwargs,
            cross_block_tables=None,  # Not yet supported in V1
        )
        return merged

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        """Execution path for non-DP case.
        Execute the model with the given scheduler output."""

        # In-process submesh DP: one scheduler, N on-device replicas fed here.
        if self._tt_dp > 1:
            return self._execute_model_inprocess_dp(scheduler_output)

        # Update cached state and prepare model inputs
        model_input = self.build_model_input(scheduler_output)
        if model_input is None:
            return EMPTY_MODEL_RUNNER_OUTPUT

        # Only 1 DP rank here
        sampled_per_dp, logprobs_per_dp = self.execute_with_model_input(model_input)
        output = self.generate_runner_output(sampled_per_dp[0], logprobs_per_dp[0])
        return output

    def _execute_model_inprocess_dp(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput:
        # 1. Drop replica pins for finished requests (frees their per-replica
        #    capacity for future admissions).
        for req_id in scheduler_output.finished_req_ids:
            self._req_to_mesh.pop(req_id, None)

        # 2. Standard single-batch state update (unchanged vLLM bookkeeping).
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT

        input_batch = self.input_batch
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return EMPTY_MODEL_RUNNER_OUTPUT

        # 3. Group active slots by replica, classifying each as prefill or
        #    decode. Each request is pinned to the least-loaded replica on first
        #    sight (by req_id, stable across slot condensation) so its paged KV
        #    is always read back on the replica it was written to.
        scheduled_new_req_ids = {
            req.req_id for req in scheduler_output.scheduled_new_reqs
        }
        prefill_idx_per_mesh: list[list[int]] = [[] for _ in range(self._tt_dp)]
        decode_idx_per_mesh: list[list[int]] = [[] for _ in range(self._tt_dp)]
        active_load = [0] * self._tt_dp
        for req_idx in range(num_reqs):
            req_id = input_batch.req_ids[req_idx]
            mesh = self._req_to_mesh.get(req_id)
            if mesh is None:
                mesh = min(range(self._tt_dp), key=lambda m: active_load[m])
                self._req_to_mesh[req_id] = mesh
            active_load[mesh] += 1
            assert active_load[mesh] <= self._per_mesh_max_seqs, (
                f"replica {mesh} has {active_load[mesh]} active requests "
                f"(> per-replica width {self._per_mesh_max_seqs}); concurrency "
                "exceeded N*width or preemption skewed the pinning."
            )
            num_scheduled = scheduler_output.num_scheduled_tokens[req_id]
            num_computed = input_batch.num_computed_tokens_cpu[req_idx]
            is_full_hit = (
                req_id in scheduled_new_req_ids
                and num_scheduled == 1
                and num_computed >= input_batch.num_prompt_tokens[req_idx] - 1
            )
            is_prefill = (req_id in scheduled_new_req_ids) and not is_full_hit
            if is_prefill:
                prefill_idx_per_mesh[mesh].append(req_idx)
            else:
                decode_idx_per_mesh[mesh].append(req_idx)

        # Full page-table for the active batch, sliced to the constant width
        # used everywhere else (required for ttnn tracing).
        block_tables_full = input_batch.block_table[0].get_cpu_tensor()[
            :num_reqs, : self.max_num_blocks_per_req
        ]

        combined = torch.zeros((num_reqs, 1), dtype=torch.int32)
        combined_logprobs: Optional[LogprobsTensors] = None
        logprob_rows_filled = 0

        def absorb_logprobs(logprobs_per_mesh, idx_per_mesh) -> None:
            """Scatter each replica's logprobs back into batch order."""
            nonlocal combined_logprobs, logprob_rows_filled
            for mesh, idxs in enumerate(idx_per_mesh):
                lp = logprobs_per_mesh[mesh]
                if lp is None or not idxs:
                    continue
                if combined_logprobs is None:
                    combined_logprobs = empty_logprobs(
                        num_reqs, lp.logprobs.shape[-1]
                    )
                scatter_logprobs(combined_logprobs, lp, idxs)
                logprob_rows_filled += len(idxs)

        # 4a. PREFILL pass (replica-parallel over meshes with prefill work).
        if any(prefill_idx_per_mesh):
            prefill_inputs = [
                self._build_mesh_prefill_input(
                    input_batch, block_tables_full, idxs
                )
                for idxs in prefill_idx_per_mesh
            ]
            merged = self.concat_dp_model_inputs(
                prefill_inputs, is_decode=False, max_blocks_decode_batch=None
            )
            sampled_per_mesh, logprobs_per_mesh = self.execute_with_model_input(merged)
            for mesh, idxs in enumerate(prefill_idx_per_mesh):
                sampled = sampled_per_mesh[mesh]
                for local_j, req_idx in enumerate(idxs):
                    combined[req_idx] = sampled[local_j]
            absorb_logprobs(logprobs_per_mesh, prefill_idx_per_mesh)

        # 4b. DECODE pass (replica-parallel over meshes with decode work).
        if any(decode_idx_per_mesh):
            max_blocks = self.max_num_blocks_per_req
            int_inputs = []
            float_inputs = []
            for idxs in decode_idx_per_mesh:
                mi = self._build_mesh_decode_input(
                    input_batch, block_tables_full, idxs
                )
                gathered = self.build_dp_decode_gather_input(mi, max_blocks)
                int_inputs.append(gathered["int_inputs"])
                float_inputs.append(gathered["float_inputs"])
            gather = {
                "int_inputs": torch.stack(int_inputs),
                "float_inputs": torch.stack(float_inputs),
            }
            merged = self.concat_dp_model_inputs(
                gather, is_decode=True, max_blocks_decode_batch=max_blocks
            )
            sampled_per_mesh, logprobs_per_mesh = self.execute_with_model_input(merged)
            for mesh, idxs in enumerate(decode_idx_per_mesh):
                sampled = sampled_per_mesh[mesh]
                for local_j, req_idx in enumerate(idxs):
                    combined[req_idx] = sampled[local_j]
            absorb_logprobs(logprobs_per_mesh, decode_idx_per_mesh)

        # Every request lands in exactly one of the two passes, so a short count
        # means some replica sampled on device and produced no host logits. Emit
        # nothing rather than a half-populated tensor.
        if combined_logprobs is not None and logprob_rows_filled != num_reqs:
            combined_logprobs = None

        # 5. Standard single-batch output emission.
        return self.generate_runner_output(combined, combined_logprobs)

    def _mesh_sampling_params(self, input_batch, idxs: list[int]) -> TTSamplingParams:
        """Uniform sampling params for a replica's slots (first slot's values;
        execute_with_model_input asserts equality across active replicas)."""
        i0 = idxs[0]
        return TTSamplingParams(
            temperature=input_batch.sampling.temperature_cpu[i0],
            top_k=input_batch.sampling.top_k_cpu[i0],
            top_p=input_batch.sampling.top_p_cpu[i0],
        )

    def _build_mesh_prefill_input(
        self, input_batch, block_tables_full: torch.Tensor, idxs: list[int]
    ) -> Optional[TTModelInput]:
        """Pure-prefill TTModelInput for the given replica slots (variable width,
        packed contiguously). Mirrors the pure-prefill branch of
        _prepare_model_inputs. Returns None if the replica has no prefill."""
        if not idxs:
            return None
        idx_t = torch.tensor(idxs, dtype=torch.long)
        max_prompt_tokens = int(max(input_batch.num_prompt_tokens[i] for i in idxs))
        input_tokens = input_batch.token_ids_cpu_tensor[idx_t, :max_prompt_tokens]
        prompt_lens = input_batch.num_prompt_tokens[idxs]
        block_tables = block_tables_full[idx_t]
        return TTModelInput(
            input_tokens=input_tokens,
            input_positions=0,
            prompt_lens=prompt_lens,
            seq_groups=None,
            block_tables=block_tables,
            unpadded_batch_size=len(idxs),
            perform_device_sampling=None,
            tt_sampling_params=self._mesh_sampling_params(input_batch, idxs),
            compat_sampling_used=False,
            sampling_metadata=None,
            multi_modal_kwargs={},
            cross_block_tables=None,
        )

    def _build_mesh_decode_input(
        self, input_batch, block_tables_full: torch.Tensor, idxs: list[int]
    ) -> Optional[TTModelInput]:
        """Pure-decode TTModelInput for the given replica slots, padded to the
        per-replica width. Mirrors the pure-decode branch of
        _prepare_model_inputs. Returns None if the replica has no decode."""
        if not idxs:
            return None
        idx_t = torch.tensor(idxs, dtype=torch.long)
        positions = input_batch.num_tokens[idxs] - 1
        positions_t = torch.from_numpy(positions).to(torch.int32)
        input_tokens = input_batch.token_ids_cpu_tensor[idx_t, positions_t].view(-1, 1)
        block_tables = block_tables_full[idx_t]

        # Pad to the fixed per-replica width (TT models require a fixed batch).
        pad = self._per_mesh_max_seqs - len(idxs)
        if pad > 0:
            input_tokens = torch.cat(
                [input_tokens, torch.zeros(pad, 1, dtype=torch.int32)]
            )
            positions_t = torch.cat(
                [positions_t, torch.ones(pad, dtype=torch.int32) * -1]
            )
            block_tables = torch.cat(
                [
                    block_tables,
                    torch.zeros(pad, block_tables.shape[1], dtype=block_tables.dtype),
                ]
            )
        return TTModelInput(
            input_tokens=input_tokens,
            input_positions=positions_t,
            prompt_lens=None,
            seq_groups=None,
            block_tables=block_tables,
            unpadded_batch_size=len(idxs),
            perform_device_sampling=None,
            tt_sampling_params=self._mesh_sampling_params(input_batch, idxs),
            compat_sampling_used=False,
            sampling_metadata=None,
            multi_modal_kwargs={},
            cross_block_tables=None,
        )

    def execute_with_model_input(
        self,
        model_input: TTModelInput,
    ) -> tuple[list[torch.Tensor], list[Optional[LogprobsTensors]]]:
        """
        Execute with a prebuilt input, supporting mixed batches.
        Returns per-DP sampled ids plus per-DP sampled-token logprobs, without
        mutating internal state. A logprobs entry is None when the batch asked
        for none, or when sampling ran on device and the host never sees the
        logits to score against.
        In DP case, called by DP rank 0 to run merged batch.
        Note: currently does not support chunked prefill.
        """
        batch_size_per_dp = model_input.unpadded_batch_size
        if not isinstance(batch_size_per_dp, list):
            batch_size_per_dp = [batch_size_per_dp]
        if not any(bs > 0 for bs in batch_size_per_dp):
            num_dp = len(batch_size_per_dp)
            return [torch.tensor([], dtype=torch.int32)] * num_dp, [None] * num_dp

        # One uniform width for the batch (see InputBatch.max_num_logprobs).
        # None short-circuits every logprob computation below, so a run that
        # requests no logprobs behaves exactly as it did before.
        num_logprobs = self.input_batch.max_num_logprobs

        sampling_params_per_dp = model_input.tt_sampling_params
        if not isinstance(sampling_params_per_dp, list):
            sampling_params_per_dp = [sampling_params_per_dp]

        # Check if batch is mixed
        # Mixed batch: prompt_lens is a list with both positive (prefill) and -1 (decode) values
        is_mixed = False
        if isinstance(model_input.prompt_lens, list):
            prompt_lens_list = model_input.prompt_lens
            has_prefill = any(pl > 0 for pl in prompt_lens_list)
            has_decode = any(pl == -1 for pl in prompt_lens_list)
            is_mixed = has_prefill and has_decode

        if is_mixed:
            # Mixed batch: process prefill and decode separately
            prompt_lens_list = model_input.prompt_lens
            is_prefill = torch.tensor(
                [pl > 0 for pl in prompt_lens_list], dtype=torch.bool
            )
            prefill_indices = torch.where(is_prefill)[0]
            decode_indices = torch.where(~is_prefill)[0]

            # Process prefill requests
            prefill_tokens = model_input.input_tokens[prefill_indices]
            prefill_prompt_lens = [prompt_lens_list[i] for i in prefill_indices]
            prefill_block_tables = model_input.block_tables[prefill_indices]

            prefill_kwargs = {
                "tokens": prefill_tokens,
                "page_table": prefill_block_tables,
                "kv_cache": self.kv_caches,
                "prompt_lens": prefill_prompt_lens,
            }

            # Add multimodal kwargs for prefill (filter to only prefill requests)
            if (
                model_input.multi_modal_kwargs
                and "pixel_values" in model_input.multi_modal_kwargs
            ):
                prefill_mm_kwargs = {"pixel_values": []}
                for idx in prefill_indices:
                    prefill_mm_kwargs["pixel_values"].append(
                        model_input.multi_modal_kwargs["pixel_values"][idx]
                    )
                prefill_kwargs.update(prefill_mm_kwargs)

            # Handle empty_slots for DP if needed
            if len(batch_size_per_dp) > 1:
                # Calculate empty_slots for prefill requests only
                stride = int(self._decode_batch_width)
                empty_slots = []
                # For mixed batches, we need to map prefill indices to global indices
                # This is complex with DP, so we'll pass all slots for now
                # TODO: Refine this for proper DP support with mixed batches
                for dp_rank, sz in enumerate(batch_size_per_dp):
                    for i in range(int(sz)):
                        empty_slots.append(dp_rank * stride + i)
                prefill_kwargs["empty_slots"] = empty_slots

            prefill_output = self.model.prefill_forward(**prefill_kwargs)

            # prefill_output may be a tuple of (logits, logprobs); v1 currently
            # doesn't handle logprobs from TT models, so keep only the logits.
            if isinstance(prefill_output, tuple):
                prefill_output = prefill_output[0]

            # Process decode requests
            num_decode = len(decode_indices)
            # For decode, we only need the single token at position 0 (decode tokens are placed there in mixed batches)
            decode_tokens = model_input.input_tokens[
                decode_indices, 0:1
            ]  # Shape: [num_decode, 1]
            decode_positions = model_input.input_positions[decode_indices]
            decode_block_tables = model_input.block_tables[decode_indices]

            # Pad decode batch to max_num_reqs if needed (TT models require fixed batch size)
            # TODO: Remove once TT models can support arbitrary batch sizes.
            max_num_reqs = self._decode_batch_width
            if num_decode < max_num_reqs:
                batch_pad = max_num_reqs - num_decode
                # Pad tokens with zeros (shape should be [batch_pad, 1] to match decode_tokens)
                decode_tokens = torch.cat(
                    [decode_tokens, torch.zeros(batch_pad, 1, dtype=torch.int32)]
                )
                # Pad positions with -1 to indicate no position
                decode_positions = torch.cat(
                    [decode_positions, torch.ones(batch_pad, dtype=torch.int32) * -1]
                )
                # Pad block tables with zeros
                decode_block_tables = torch.cat(
                    [
                        decode_block_tables,
                        torch.zeros(
                            batch_pad, decode_block_tables.shape[1], dtype=torch.int32
                        ),
                    ]
                )

            decode_kwargs = {
                "tokens": decode_tokens,
                "start_pos": decode_positions,
                "page_table": decode_block_tables,
                "kv_cache": self.kv_caches,
                "enable_trace": self.trace_mode,
                "read_from_device": True,
            }

            # Handle sampling params for decode
            if self.sample_on_device_mode == "all" or (
                self.sample_on_device_mode == "decode_only"
            ):
                # Check that sampling params are the same for all DP ranks
                non_none_params = [
                    sp for sp in sampling_params_per_dp if sp is not None
                ]
                if non_none_params:
                    assert all(sp == non_none_params[0] for sp in non_none_params), (
                        "Sampling params must be the same for all active DP ranks"
                    )
                    decode_kwargs["sampling_params"] = non_none_params[0]

            # TODO: Add encoder-decoder support
            enc_dec_kwargs: dict[str, Any] = {}
            decode_output = self.model.decode_forward(**decode_kwargs, **enc_dec_kwargs)

            # decode_output may be a tuple of (logits, logprobs); v1 currently
            # doesn't handle logprobs from TT models, so keep only the logits.
            if isinstance(decode_output, tuple):
                decode_output = decode_output[0]

            # Combine outputs in original batch order
            # Prefill outputs: shape [num_prefill, seq_len, vocab_size] or [num_prefill] if sampled
            # Decode outputs: shape [num_decode, vocab_size] or [num_decode] if sampled
            total_batch_size = model_input.input_tokens.shape[0]

            # Handle prefill output based on sample_on_device_mode
            prefill_logprobs = None
            if self.sample_on_device_mode == "all":
                # Already sampled tokens
                prefill_sampled = prefill_output.view(-1, 1).to(torch.int32)
            else:
                # Logits: take last token and sample
                prefill_logits = prefill_output[:, -1, :]
                # Use first sampling param for prefill (assuming same for all)
                prefill_sp = (
                    sampling_params_per_dp[0] if sampling_params_per_dp[0] else None
                )
                if prefill_sp is None:
                    raise ValueError(
                        "Sampling params required for prefill in mixed batch"
                    )
                prefill_sampled = (
                    sample_tokens(prefill_logits, prefill_sp)
                    .view(-1, 1)
                    .to(torch.int32)
                )
                if num_logprobs is not None:
                    prefill_logprobs = compute_sampled_logprobs(
                        prefill_logits, prefill_sampled, num_logprobs
                    )

            # Handle decode output based on sample_on_device_mode and actual shape
            # Only use the first num_decode outputs (ignore padding)
            decode_output_actual = decode_output[:num_decode]

            # Check if output is already sampled (1D or 2D with last dim = 1) or logits
            is_sampled = decode_output_actual.dim() == 1 or (
                decode_output_actual.dim() == 2 and decode_output_actual.shape[-1] == 1
            )

            decode_logprobs = None
            if (
                self.sample_on_device_mode == "all"
                or self.sample_on_device_mode == "decode_only"
            ) or is_sampled:
                # Already sampled tokens
                decode_sampled = decode_output_actual.view(-1, 1).to(torch.int32)
            else:
                # Logits: sample
                # Handle 3D logits [batch, seq_len, vocab_size] - extract last token
                if decode_output_actual.dim() == 3:
                    # Extract last token's logits: [batch, seq_len, vocab_size] -> [batch, vocab_size]
                    decode_logits = decode_output_actual[:, -1, :]
                elif decode_output_actual.dim() == 2:
                    # Already 2D: [batch, vocab_size]
                    decode_logits = decode_output_actual
                else:
                    raise ValueError(
                        f"Unexpected decode output shape: {decode_output_actual.shape}, "
                        f"expected 2D [batch, vocab_size] or 3D [batch, seq_len, vocab_size]"
                    )

                decode_sp = (
                    sampling_params_per_dp[0] if sampling_params_per_dp[0] else None
                )
                if decode_sp is None:
                    raise ValueError(
                        "Sampling params required for decode in mixed batch"
                    )
                decode_sampled = (
                    sample_tokens(decode_logits, decode_sp).view(-1, 1).to(torch.int32)
                )
                if num_logprobs is not None:
                    decode_logprobs = compute_sampled_logprobs(
                        decode_logits, decode_sampled, num_logprobs
                    )

            # Combine in original order
            combined_output = torch.zeros((total_batch_size, 1), dtype=torch.int32)
            combined_output[prefill_indices] = prefill_sampled
            combined_output[decode_indices] = decode_sampled

            # Same scatter for logprobs. Only assembled when both halves produced
            # them; if either sampled on device there is no consistent set to
            # emit for the batch, so fall back to None.
            combined_logprobs = None
            if prefill_logprobs is not None and decode_logprobs is not None:
                combined_logprobs = empty_logprobs(
                    total_batch_size, prefill_logprobs.logprobs.shape[-1]
                )
                scatter_logprobs(combined_logprobs, prefill_logprobs, prefill_indices)
                scatter_logprobs(combined_logprobs, decode_logprobs, decode_indices)

            # Split by DP rank for return format
            # Batch is organized by DP rank: first batch_size_per_dp[0] requests are DP rank 0, etc.
            sampled_token_ids_per_dp = []
            logprobs_per_dp: list[Optional[LogprobsTensors]] = []
            start_idx = 0
            for dp_rank, sz in enumerate(batch_size_per_dp):
                if sz <= 0:
                    sampled_token_ids_per_dp.append(torch.tensor([], dtype=torch.int32))
                    logprobs_per_dp.append(None)
                else:
                    end_idx = start_idx + sz
                    sampled_token_ids_per_dp.append(combined_output[start_idx:end_idx])
                    logprobs_per_dp.append(
                        slice_logprobs(combined_logprobs, start_idx, end_idx)
                    )
                    start_idx = end_idx

            return sampled_token_ids_per_dp, logprobs_per_dp

        else:
            # Pure batch: use existing logic
            is_decode = model_input.prompt_lens is None

            kwargs = {
                "tokens": model_input.input_tokens,
                "page_table": model_input.block_tables,
                "kv_cache": self.kv_caches,
            }

            if not is_decode:
                kwargs["prompt_lens"] = model_input.prompt_lens
                kwargs.update(model_input.multi_modal_kwargs)
                if len(batch_size_per_dp) > 1:
                    # TODO: the model should only require DP ranks, but passing
                    # "global" user ids instead for backwards compatibility.
                    stride = int(self._decode_batch_width)
                    empty_slots = []
                    for dp_rank, sz in enumerate(batch_size_per_dp):
                        for i in range(int(sz)):
                            empty_slots.append(dp_rank * stride + i)
                    kwargs["empty_slots"] = empty_slots
            else:
                kwargs["start_pos"] = model_input.input_positions
            if self.sample_on_device_mode == "all" or (
                self.sample_on_device_mode == "decode_only" and is_decode
            ):
                # Check that sampling params are the same for all DP ranks.
                # TODO: Remove this restriction and concat sampling params in
                # concat_dp_model_inputs once models can support mixed params.
                non_none_params = [
                    sp for sp in sampling_params_per_dp if sp is not None
                ]
                assert all(sp == non_none_params[0] for sp in non_none_params), (
                    "Sampling params must be the same for all active DP ranks"
                )
                kwargs["sampling_params"] = non_none_params[0]

            # Execute model
            if not is_decode:
                tt_out = self.model.prefill_forward(**kwargs)
            else:
                # TODO: Add encoder-decoder support
                enc_dec_kwargs: dict[str, Any] = {}
                tt_out = self.model.decode_forward(
                    **kwargs,
                    **enc_dec_kwargs,
                    enable_trace=self.trace_mode,
                    read_from_device=True,
                )

            # tt_out is a tuple of (logits, logprobs)
            # v1 currently doesn't handle logprobs from TT models
            if isinstance(tt_out, tuple):
                tt_out = tt_out[0]

            sampled_token_ids_per_dp: list[torch.Tensor] = []
            logprobs_per_dp: list[Optional[LogprobsTensors]] = []
            # Decode ranks occupy fixed-width segments (rank r -> rows
            # [r*width, ...)) so address them positionally, robust to empty
            # ranks (sz==0). Prefill rows are packed contiguously.
            prefill_start = 0
            for dp_rank, sz in enumerate(batch_size_per_dp):
                if sz <= 0:
                    sampled_token_ids_per_dp.append(torch.tensor([], dtype=torch.int32))
                    logprobs_per_dp.append(None)
                    continue
                if is_decode:
                    seg_start = dp_rank * self._decode_batch_width
                else:
                    seg_start = prefill_start
                    prefill_start += sz
                rank_logprobs = None
                if not self.sample_on_device_mode or (
                    self.sample_on_device_mode == "decode_only" and not is_decode
                ):
                    logits = tt_out[seg_start : seg_start + sz, -1, :]
                    next_token_ids = sample_tokens(
                        logits, sampling_params_per_dp[dp_rank]
                    )
                    if num_logprobs is not None:
                        rank_logprobs = compute_sampled_logprobs(
                            logits, next_token_ids, num_logprobs
                        )
                else:
                    next_token_ids = tt_out[seg_start : seg_start + sz]
                sampled_token_ids_per_dp.append(next_token_ids.view(sz, 1))
                logprobs_per_dp.append(rank_logprobs)

            return sampled_token_ids_per_dp, logprobs_per_dp

    def generate_runner_output(
        self,
        sampled_token_ids: torch.Tensor,
        logprobs: Optional[LogprobsTensors] = None,
    ):
        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        assert sampled_token_ids.shape[0] == self.input_batch.num_reqs, (
            f"Number of request outputs {sampled_token_ids.shape[0]} != "
            f"number of requests in input batch {self.input_batch.num_reqs}"
        )
        num_out_tokens = sampled_token_ids.shape[1]
        assert num_out_tokens == 1, "Currently only supporting 1 output token"
        for req_idx, sampled_ids in enumerate(sampled_token_ids):
            start_idx = self.input_batch.num_tokens[req_idx]
            end_idx = start_idx + num_out_tokens
            assert end_idx <= self.model_config.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.model_config.max_model_len}"
            )

            # Update persistent batch
            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens[req_idx] = end_idx

            # Update request state
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # Empty prompt log probs
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}
        for req_id in self.input_batch.req_ids[: self.input_batch.num_reqs]:
            prompt_logprobs_dict[req_id] = None

        # Note: currently does not support speculative decoding, prompt logprobs,
        # or pooling. Sampled-token logprobs are supported for host-side sampling
        # (ModelRunnerOutput carries them as lists, hence tolists()).
        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=sampled_token_ids.tolist(),
            logprobs=logprobs.tolists() if logprobs is not None else None,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
        )

    def get_model(self) -> nn.Module:
        """Get the underlying model."""
        return self.model

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        """Get supported generation tasks for this model."""
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")
        else:
            # Fallback for TT models: Check if model has generation methods
            # All TT models (LlamaForCausalLM, QwenForCausalLM, etc.) have these methods
            if (
                hasattr(model, "prefill_forward")
                and hasattr(model, "decode_forward")
                and self.model_config.runner_type == "generate"
            ):
                supported_tasks.append("generate")
                logger.info(
                    f"TT model {model.__class__.__name__} recognized as generation model "
                    f"by method presence check"
                )

        # Add transcription support if the model supports it
        # (uncomment if needed)
        # from vllm.model_executor.models.interfaces import supports_transcription
        # if supports_transcription(model):
        #     if model.supports_transcription_only:
        #         return ["transcription"]
        #     supported_tasks.append("transcription")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        """Get supported pooling tasks for this model."""
        model = self.get_model()
        if not is_pooling_model(model):
            return []
        return list(model.pooler.get_supported_tasks())

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Get all supported tasks for this model runner."""
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)
