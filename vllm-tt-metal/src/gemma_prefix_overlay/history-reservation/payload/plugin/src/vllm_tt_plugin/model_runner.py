# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from __future__ import annotations

import gemma_history_reservation as _history_reservation
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, fields, replace
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import regex as re
import torch
import ttnn
from vllm.config import VllmConfig
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    DraftTokenIds,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
)
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_tt_plugin.async_decode import (
    AsyncTTModelRunnerOutput,
    CompletedDecodeStep,
    DeferredDecodeOutput,
    SubmittedStepContext,
    TTAsyncDecodeController,
    TTDecodeSubmission,
)
from vllm_tt_plugin.config import (
    get_tt_admitted_request_limit,
    get_tt_data_parallel_size,
    get_tt_max_batch_size,
    get_tt_output_tokens_per_step,
    get_tt_per_lane_max_num_seqs,
    get_tt_spec_plan,
    get_tt_supports_device_grammar,
    is_tt_adaptive_block_output_model,
    is_tt_block_output_model,
    is_tt_device_grammar_preload_eligible,
)
from vllm_tt_plugin.host_sampler import create_host_sampler
from vllm_tt_plugin.input_batch import (
    SEED_NONE_SENTINEL,
    CachedRequestState,
    InputBatch,
    TTLaneInputBatch,
    apply_cached_req_state_update,
    build_cached_request_state,
    clone_torch_generator,
)
from vllm_tt_plugin.lane_scheduler import get_tt_step_plan
from vllm_tt_plugin.loader import TTModelLoader
from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.logprobs import build_device_logprobs
from vllm_tt_plugin.model_input import (
    TTModelInput,
    TTSamplingParams,
    slice_tt_sampling_params,
)
from vllm_tt_plugin.platform import TTPlatform
from vllm_tt_plugin.spec_admission import method_requirements
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    HIDDEN_HANDOFF_ROUNDTRIP,
    HIDDEN_HANDOFFS,
    PLACEHOLDER_TOKEN_ID,
    SPEC_REQUIREMENT_DEVICE_PROPOSE,
    SPEC_REQUIREMENT_HIDDEN_FEED,
    SPEC_REQUIREMENTS,
    DraftOutput,
    accept_greedy_drafts,
    normalize_declared_values,
)
from vllm_tt_plugin.structured_output import (
    capture_structured_decode_request_ids,
    has_structured_outputs,
    reorder_grammar_bitmask_for_tt_batch,
    scheduled_structured_output_request_ids,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

import numpy as np

logger = init_tt_logger(__name__)

# Matches the upstream attention-layer naming convention used by registered
# vLLM models (e.g. "model.language_model.layers.5.self_attn") as well as
# bare "layers.5" forms used in TT spec hooks. The first capture group is
# the integer layer index.
_LAYER_NAME_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _parse_layer_index(layer_name: str) -> int:
    """Extract the integer layer index from a layer name.

    Used to map ``KVCacheGroupSpec.layer_names`` back to model-side layer
    indices when distributing per-group KV cache shapes across the
    layer-indexed allocator. The hook author is expected to follow the
    ``...layers.<idx>...`` convention.
    """
    match = _LAYER_NAME_RE.search(layer_name)
    if match is None:
        raise ValueError(
            f"Could not parse a layer index from layer name '{layer_name}'. "
            "TT spec hooks must use the '...layers.<idx>...' naming convention."
        )
    return int(match.group(1))


@dataclass(frozen=True)
class _SyncForward:
    """Materialized front-packed forward result awaiting sampling.

    Carries everything ``_get_output_tokens`` needs so the sampling tail can
    run in a later ``sample_tokens`` call rather than inside ``execute_model``.
    """

    tt_out: Any
    tt_log_probs: Any
    sampling_params: TTSamplingParams
    model_input: TTModelInput
    batch_size_per_dp: list[int]
    perform_device_sampling: bool
    is_decode: bool
    decode_submission: TTDecodeSubmission | None = None
    # The verify's hidden handle, for a model whose drafter consumes it.
    spec_hidden: Any | None = None


def _step_verifies(
    supports_narrow_decode: bool,
    num_valid_drafts: torch.Tensor,
    accepted_counts: torch.Tensor,
    num_rows: int,
) -> bool:
    """Whether a speculating launch sends the model a verify on this step.

    Two things make a step speculative. A row carrying drafts is the obvious
    one. The other is a row whose previous step committed more than one token:
    ``accepted_counts`` is how the model finds which candidate state slot that
    commit landed on, so the step after such a commit carries the count even
    when it drafts nothing. One step resolves it, because that step commits a
    single token and records a count of 1.

    A model that does not declare ``supports_narrow_decode`` implements one
    input shape, so every step it is sent is a verify.
    """
    if not supports_narrow_decode:
        return True
    if bool(num_valid_drafts.any()):
        return True
    # Only the live rows are read: a padding row sits at the post-prefill
    # default of 1 and owns no request, so it says nothing about state to
    # resolve.
    return any(int(accepted_counts[row]) > 1 for row in range(num_rows))


def _coerce_output_block(
    sampled_token_ids: torch.Tensor, num_reqs: int, width: int
) -> torch.Tensor:
    """Validate one step's sampled tokens against the output-width contract.

    Returns the tensor shaped ``[num_reqs, width]`` (reshaping the empty
    tensor) or raises when the model output violates the contract.
    """
    assert sampled_token_ids.shape[0] == num_reqs, (
        f"Number of request outputs {sampled_token_ids.shape[0]} != "
        f"number of requests in input batch {num_reqs}"
    )
    if num_reqs == 0 and sampled_token_ids.numel() == 0:
        sampled_token_ids = sampled_token_ids.reshape(0, width)
    if sampled_token_ids.dim() != 2 or sampled_token_ids.shape[1] != width:
        raise ValueError(
            "Model output width violates output_tokens_per_step: "
            f"got shape {tuple(sampled_token_ids.shape)}, expected "
            f"[num_requests, {width}]"
        )
    return sampled_token_ids


def _notify_model_slot_moves(runner, moves: dict[int, int]) -> None:
    """Tell the model that its per-slot state moved ``old -> new``.

    The model keys B=1 session ownership on the slot it was handed at prefill
    (``empty_slots``), and the decode gather permutes every slot. Two things have
    to stay true for a release to identify the right request:

    * the model's recorded owner slot must follow the gather -- otherwise a
      request prefilled at slot 1 and gathered to row 0 is released as 0 while
      the model still holds 1, and its session outlives the request;
    * the release must then use the request's CURRENT slot.

    Keying on the prefill slot instead (the first attempt at this) is not enough,
    because slots are REUSED: a cancelled request frees its slot, a survivor is
    gathered into it, and a new request is then prefilled into the survivor's old
    slot and arms a session there. Releasing the survivor by its prefill slot
    matches the newcomer's owner slot and clears a live session
    (vllm-tt-plugin#118 review, finding 1).

    The hook is optional -- models that hold no per-slot session do not define it.
    Reached through ``runner.__dict__``/``getattr`` so it also works on a
    partially initialised runner and on the ``SimpleNamespace`` stand-ins the
    state-slot tests pass as ``self`` when calling these methods unbound.
    """
    if not moves:
        return
    note = getattr(getattr(runner, "model", None), "note_state_slots_moved", None)
    if callable(note):
        note(dict(moves))


class TTModelRunner:
    def __init__(
        self,
        vllm_config: VllmConfig,
        mesh_device: ttnn.MeshDevice,
        trace_mode: str,
        enable_model_warmup: bool,
        num_devices: int,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config
        self._output_tokens_per_step = get_tt_output_tokens_per_step(vllm_config)
        self._is_block_output_model = is_tt_block_output_model(vllm_config)
        self._is_adaptive_block_output = is_tt_adaptive_block_output_model(vllm_config)
        self._persistent_capture_released = False
        self._device_grammar_poisoned = False

        # Resolved and validated by the platform's config-time admission;
        # ``None`` means the launch asked for no speculation. The draft length
        # is the plan's, not ``speculative_config.num_speculative_tokens``, for
        # a model that reduced it -- though the platform publishes the
        # reduction back, so the two agree.
        spec_plan = get_tt_spec_plan(vllm_config)
        self._num_speculative_tokens = spec_plan.effective_k if spec_plan else 0
        self._spec_method = (
            str(vllm_config.speculative_config.method) if spec_plan else None
        )
        # A model that also serves a narrow [B, 1] decode lets a step on which
        # no request carries drafts run as the ordinary decode it is, which is
        # what keeps batched baseline decoding overlapped inside a speculating
        # server. ``load_model`` narrows this further, once the model exists.
        self._spec_supports_narrow_decode = bool(
            spec_plan and spec_plan.supports_narrow_decode
        )
        # Who proposes: the model's own drafter on device, or a host drafter.
        # Read from the one requirements table rather than from a second list of
        # method names, and once, because that table is rebuilt per call.
        self._spec_drafts_from_model = bool(
            self._spec_method
            and SPEC_REQUIREMENT_DEVICE_PROPOSE
            in method_requirements(self._spec_method)
        )

        if self.model_config.is_encoder_decoder:
            raise ValueError("Encoder-decoder models aren't yet supported for TT")

        # Detect if the model has "mrope" rope_scaling type.
        # mrope requires keeping "rope_deltas" between prefill/decode phases.
        self.request_specific_rope = bool(self.model_config.uses_mrope)
        if self.request_specific_rope:
            self.previous_req_ids: set[str] = set()

        self.mesh_device = mesh_device
        self.trace_mode = trace_mode
        self.enable_model_warmup = enable_model_warmup
        self.supports_device_grammar = is_tt_device_grammar_preload_eligible(
            vllm_config,
            trace_mode=self.trace_mode,
            enable_model_warmup=self.enable_model_warmup,
        )
        if (
            not self.supports_device_grammar
            and get_tt_supports_device_grammar(vllm_config)
            and self.trace_mode != "none"
            and not self.enable_model_warmup
        ):
            logger.warning(
                "TT device grammar sampling requires model warmup before "
                "traced execution; retaining host grammar sampling because "
                "enable_model_warmup is false"
            )
            self.supports_device_grammar = False
        # Runtime-discovered physical device count, supplied by the worker.
        self.num_devices = num_devices
        # Whether to sample on device
        self.sample_on_device_mode = getattr(TTPlatform, "sample_on_device_mode", None)
        assert self.sample_on_device_mode in (None, "all", "decode_only")
        if self.supports_device_grammar and (
            self.sample_on_device_mode is None or self.scheduler_config.async_scheduling
        ):
            logger.info(
                "TT device grammar runtime activation is disabled "
                "(sample_on_device_mode=%s, async_scheduling=%s); retaining "
                "host grammar sampling",
                self.sample_on_device_mode,
                self.scheduler_config.async_scheduling,
            )
            self.supports_device_grammar = False
        # Whether the model supports top-K logprobs on device.
        # Detected from model_type (available to all DP ranks without
        # requiring the model to be loaded). Models like gpt-oss-120b
        # set use_topk_logprobs=True and return top-32 logprobs from device.
        # TODO: Update this check as more models add top-K logprobs support.
        # https://github.com/tenstorrent/tt-metal/issues/40810
        self.supports_topk_logprobs = (
            self.model_config.hf_config.model_type == "gpt_oss"
        )

        logger.info(
            "TTModelRunner: trace_mode=%s, "
            "sample_on_device_mode=%s, enable_model_warmup=%s",
            self.trace_mode,
            self.sample_on_device_mode,
            self.enable_model_warmup,
        )

        # mm_hash -> encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}

        # Cached request states. Request states are tracked in the runner so
        # they don't need to be re-sent every scheduling step. For requests
        # that have been scheduled before, only the diff is received from
        # the scheduler output.
        self.requests: dict[str, CachedRequestState] = {}

        # Cache the arange needed for unpacking structured output bitmask
        self.structured_output_arange = torch.arange(0, 32)
        self.vocab_size = self.model_config.get_vocab_size()
        self.bitmask_size = cdiv(self.vocab_size, 32)

        # For on-device decode sampling, we must signal if the padded decode
        # batch layout changed since the *previous decode step*. Layout can
        # change during prefill steps (e.g. new requests added), so we keep a
        # sticky flag and clear it only after a decode input consumes it.
        self._decode_layout_changed_since_last_decode: bool = True

        # Forward/sampling split: ``execute_model`` runs the device forward,
        # enqueues a deferred sampler here, and returns ``None``; the engine
        # then calls ``sample_tokens`` which pops and runs it. FIFO matches the
        # batch-queue's oldest-first finalization order, so a single queue stays
        # correct even with one forward in flight.
        self._pending_samples: deque[Any] = deque()

        # The platform has already disabled upstream async scheduling when the
        # registered model lacks async-decode capability. Contract negotiation
        # selects reload semantics, not async eligibility.
        self.async_decode_scheduling = self.scheduler_config.async_scheduling
        self._steady_decode_lock = threading.Lock()
        self._pending_async_steps: deque[DeferredDecodeOutput] = deque()
        self._pending_async_overlap_ok: deque[bool] = deque()
        self._completed_decode_steps: deque[CompletedDecodeStep] = deque()
        self.async_decode = TTAsyncDecodeController(self)
        self.tt_data_parallel_size = get_tt_data_parallel_size(vllm_config)
        self.tt_max_batch_size = get_tt_max_batch_size(vllm_config)
        self.tt_admitted_request_limit = get_tt_admitted_request_limit(vllm_config)
        self.tt_per_lane_max_num_seqs = get_tt_per_lane_max_num_seqs(vllm_config)

        # req_id -> device slot holding its per-slot state (GDN recurrent/conv, seed
        # RNG, decode trace buffers). Needed because evict/re-add and condense move a
        # request's ROW, not its state.
        self._req_state_slot: dict[str, int] = {}

        # req_id -> how many tokens its last step committed, for the same
        # reason: a running request the scheduler skips for one step is removed
        # from the persistent batch and added back later, and only an explicit
        # preemption releases the model's candidate state, so the count that
        # selects among those candidates has to survive the row going away.
        # Draft IDs are resolved separately by _drafts_to_verify according to
        # scheduling mode. An absent accepted-count entry defaults to 1.
        self._req_accepted_counts: dict[str, int] = {}

        # Pending proposal IDs. Synchronous execution hands them to the
        # scheduler through take_draft_token_ids. Asynchronous execution
        # consumes them in _drafts_to_verify within the scheduled lookahead
        # reservation. Each consumer takes a proposal only once.
        self._proposed_draft_token_ids: dict[str, list[int]] = {}
        # Rows a verify answered by argmax although their request is not
        # speculable. See _note_unspeculable_verify_rows; reported at shutdown.
        self._num_unspeculable_verify_rows = 0
        # Built on first use rather than here, because constructing it compiles
        # numba kernels and a launch that never speculates must not pay that.
        self._ngram_proposer: Any = None
        self._pending_state_slot_settle: dict[str, int] | None = None
        # Slot-level ``old -> new`` for the same pending gather, forwarded to the
        # model once the decode is accepted so its recorded session owner follows
        # the move (see ``_notify_model_slot_moves``).
        self._pending_state_slot_moves: dict[int, int] | None = None

        # Every standard-DP rank owns its own mesh and therefore its own host
        # sampler state. Single-process modes also instantiate exactly one.
        self.host_sampler = create_host_sampler()

        # Host-side logits processors (min_p, logit_bias, min_tokens, plus any
        # custom logits processors). Used by the host sampler when device
        # sampling isn't supported for a given batch.
        self._host_logitsprocs: LogitsProcessors = build_logitsprocs(
            vllm_config=vllm_config,
            device=torch.device("cpu"),
            is_pin_memory=False,
            is_pooling_model=False,
            custom_logitsprocs=(self.model_config.logits_processors or ()),
        )

    def shutdown(self) -> None:
        """Deterministically release optional model-lifetime captures.

        Called from ``TTWorker.shutdown`` while the mesh is still open. This
        is intentionally not wired to ``__del__``: the runner sits in
        reference cycles, so a destructor can fire during interpreter
        teardown after the devices closed, where a device-side release is
        worse than leaking the capture to process exit.
        """
        if getattr(self, "_persistent_capture_released", False):
            return
        unspeculable_rows = getattr(self, "_num_unspeculable_verify_rows", 0)
        if unspeculable_rows:
            logger.warning(
                "Speculative verify steps committed a greedy token for %d "
                "row(s) whose request is not speculable (non-zero temperature "
                "or a penalty).",
                unspeculable_rows,
            )
        release = getattr(
            getattr(self, "model", None), "release_persistent_capture", None
        )
        if callable(release):
            try:
                release()
            except Exception:
                logger.exception("Failed to release persistent model capture")
        self._persistent_capture_released = True

    @property
    def lane_batch(self) -> TTLaneInputBatch:
        """The persistent batch as its lane-DP type.

        Valid only in lane mode, where ``initialize_kv_cache`` builds a
        ``TTLaneInputBatch``. The lane step orchestration (``_execute_lane_step``)
        reads this instead of re-casting ``input_batch`` at every use site.
        """
        return cast(TTLaneInputBatch, self.input_batch)

    @property
    def _is_lane_mode(self) -> bool:
        """Single-process multi-lane (lane-DP) mode: one engine drives
        ``tt_data_parallel_size`` in-process DP replicas. In this mode the
        persistent batch is a ``TTLaneInputBatch`` that owns the lane layout
        (stable per-lane device slots, merged sampling) and the runner only
        orchestrates; single-process execution and standard multi-process DP
        ranks keep the plain ``InputBatch``, one per engine.

        Derived (rather than cached in ``__init__``) so it depends only on
        already-set runner state -- this mirrors ``uses_tt_lane_coordinator``:
        vLLM sees a single engine (``data_parallel_size == 1``) while the TT
        backend runs more than one lane.
        """
        return (
            self.parallel_config.data_parallel_size == 1
            and self.tt_data_parallel_size > 1
        )

    def load_model(self) -> None:
        loader = TTModelLoader(self.load_config)
        self.model = loader.load_model(
            vllm_config=self.vllm_config, model_config=self.model_config
        )
        if self.tt_admitted_request_limit != self.tt_max_batch_size:
            configure = getattr(self.model, "configure_admission_limit", None)
            if not callable(configure):
                raise RuntimeError("Explicit admission requires a supporting model")
            configure(self.tt_admitted_request_limit)
            logger.info("GEMMA_ADMISSION_CONFIG physical=%d admitted=%d model_hook=true", self.tt_max_batch_size, self.tt_admitted_request_limit)
        if self.supports_device_grammar:
            activate = getattr(self.model, "enable_device_grammar", None)
            if callable(activate):
                activate()
            runtime_support = bool(getattr(self.model, "device_grammar_enabled", False))
            has_sample_api = callable(
                getattr(self.model, "sample_deferred_decode", None)
            )
            if not runtime_support or not has_sample_api:
                logger.warning(
                    "TT model declares device grammar support but this runtime "
                    "layout has no compatible sampler; retaining host grammar "
                    "sampling (runtime_support=%s, sample_api=%s)",
                    runtime_support,
                    has_sample_api,
                )
                self.supports_device_grammar = False
        if self._spec_supports_narrow_decode:
            self._spec_supports_narrow_decode = self._narrow_steps_serve_the_drafter()

    def _poison_device_grammar(self) -> None:
        """Reject all later runner work after a deferred decode becomes unsafe."""
        self._device_grammar_poisoned = True

    def _raise_if_device_grammar_poisoned(self) -> None:
        if getattr(self, "_device_grammar_poisoned", False):
            raise RuntimeError(
                "TT device grammar sampling previously failed after decode "
                "submission; reconstruct the model runner before continuing"
            )

    def _narrow_steps_serve_the_drafter(self) -> bool:
        """Whether a step that verifies nothing can still feed the drafter.

        A step with nothing to verify returns no ``VerifyOutput``, so it
        produces no hidden handle. A drafter whose target hidden state reaches
        it through the runner would then be asked to draft from nothing, so for
        that one pairing the step stays a verify and the launch keeps the
        ordinary decode's overlap only where it can serve it. A drafter that
        declares no hidden feed, or one that keeps its own state on device, is
        unaffected.

        Read from what the model declares rather than from what the method
        requires, because this is a property of the drafter: the model-owned
        method demands only that the model propose, and what its drafter reads
        is the model's own statement.

        Decided here rather than at step time: the alternative is handing that
        drafter ``None`` and hoping, which is the silent degradation this
        contract refuses everywhere else.
        """
        if not self._spec_drafts_from_model:
            # A host proposer reads the request's own text and is handed no
            # hidden state, so nothing here constrains it.
            return True
        declared_requirements = normalize_declared_values(
            (getattr(type(self.model), "model_capabilities", None) or {}).get(
                "spec_requirements"
            ),
            SPEC_REQUIREMENTS,
            f"{type(self.model).__name__} model_capabilities['spec_requirements']",
        )
        if SPEC_REQUIREMENT_HIDDEN_FEED not in declared_requirements:
            return True
        declared = normalize_declared_values(
            (getattr(type(self.model), "model_capabilities", None) or {}).get(
                "spec_hidden_handoff"
            ),
            HIDDEN_HANDOFFS,
            f"{type(self.model).__name__} model_capabilities['spec_hidden_handoff']",
        )
        if HIDDEN_HANDOFF_ROUNDTRIP not in declared:
            return True
        logger.info(
            "TT speculative decoding: %s feeds its drafter the target hidden "
            "state through the runner, so every decode step stays a verify "
            "and none of them overlaps. A model that keeps its hidden state "
            "on device, or a drafter that needs none, decodes a draftless "
            "step as an ordinary overlapping decode.",
            type(self.model).__name__,
        )
        return False

    def _uses_async_scheduler(self) -> bool:
        """Whether upstream publishes outputs through placeholder accounting.

        Synchronously produced outputs such as final prefills still require
        lifecycle-deferred host state when the engine uses ``AsyncScheduler``.
        """
        return bool(
            getattr(getattr(self, "scheduler_config", None), "async_scheduling", False)
        )

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        # TT backend currently supports text generation only.
        # (No transcription support yet.)
        return ["generate"]

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        # TT backend does not support pooling/embedding tasks yet.
        return []

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on ``kv_cache_config``.

        Args:
            kv_cache_config: Configuration for the KV cache. May contain one
                group (uniform attention) or multiple groups (hybrid models
                like Gemma3/4 / GPT-OSS that mix sliding-window and full
                attention layers).
        """
        kv_cache_groups = kv_cache_config.kv_cache_groups
        self._validate_kv_cache_groups(kv_cache_groups)

        # Stash on the runner for downstream phases that need to walk the
        # group structure during input prep / forward.
        self.kv_cache_config = kv_cache_config

        # Upstream's hybrid kv cache manager equalises *page size*
        # (block_size × num_kv_heads × head_size × dtype_bytes) across
        # groups, not block_size itself: when groups have different
        # ``num_kv_heads × head_size`` (e.g. Gemma4's full layers use
        # head_dim=512 vs sliding head_dim=256), upstream's
        # ``unify_kv_cache_spec_page_size`` adjusts ``block_size`` per
        # spec instead. Use each group's own ``block_size`` here; the
        # input batch / MultiGroupBlockTable already takes a per-group
        # list. ``self.cache_config.block_size`` (the user-specified
        # value) is still used elsewhere for per-request bounds — that's
        # the smaller of the unified sizes, which conservatively
        # overestimates for the larger-block groups (extra block-table
        # rows allocated, never indexed).
        per_group_block_sizes = [g.kv_cache_spec.block_size for g in kv_cache_groups]

        max_num_reqs = self.tt_max_batch_size
        max_model_len = self.model_config.max_model_len
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        if self._is_lane_mode:
            # Lane-chunked persistent batch: rows are partitioned into
            # ``tt_data_parallel_size`` lanes of ``tt_per_lane_max_num_seqs``
            # slots, so a request's persistent row is its stable device slot.
            # ``num_lanes * per_lane == tt_max_batch_size`` (the global
            # max_num_seqs in lane mode), so the slot count matches the device
            # batch the merged step executes.
            self.input_batch: InputBatch = TTLaneInputBatch(
                num_lanes=self.tt_data_parallel_size,
                per_lane=self.tt_per_lane_max_num_seqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                vocab_size=self.vocab_size,
                block_sizes=per_group_block_sizes,
                kernel_block_sizes=per_group_block_sizes,
                logitsprocs=self._host_logitsprocs,
                disable_logprobs=self._is_block_output_model,
                output_tokens_per_step=self._output_tokens_per_step,
            )
        else:
            self.input_batch = InputBatch(
                max_num_reqs=max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                vocab_size=self.vocab_size,
                block_sizes=per_group_block_sizes,
                kernel_block_sizes=per_group_block_sizes,
                logitsprocs=self._host_logitsprocs,
                disable_logprobs=self._is_block_output_model,
                output_tokens_per_step=self._output_tokens_per_step,
            )

        # The block tables in the persistent input batch have
        # max_num_blocks_per_req = cdiv(max_model_len, block_size) but this
        # does not take into account num blocks in KV cache. Actual max is
        # min of these two. Used to slice block tables during input prep.
        self.max_num_blocks_per_req = min(
            cdiv(max_model_len, self.cache_config.block_size),
            kv_cache_config.num_blocks,
        )

        # Cache layer→group index mapping for hybrid models so submit_*
        # can expand ``block_tables_per_group`` into ``block_tables_per_layer``
        # without re-deriving vLLM's group construction order. Non-hybrid
        # configurations (single group) skip the expansion entirely. The
        # check is on ``len(kv_cache_groups)`` rather than the model class so
        # it works on every DP rank, including standard-DP subprocesses whose
        # local parallel config is collapsed to DP=1 by upstream before the TT
        # worker loads the model.
        self._layer_to_group_idx: list[int] | None = None
        if len(kv_cache_groups) > 1:
            num_layers = self.model_config.get_num_layers_by_block_type(
                self.parallel_config, "attention"
            )
            mapping: list[int | None] = [None] * num_layers
            for g_idx, group in enumerate(kv_cache_groups):
                for layer_name in group.layer_names:
                    idx = _parse_layer_index(layer_name)
                    mapping[idx] = g_idx
            missing = [i for i, g in enumerate(mapping) if g is None]
            if missing:
                raise ValueError(
                    f"No KVCacheGroupSpec covers layer indices {missing} "
                    f"on hybrid model; every attention layer must appear "
                    "in some group's layer_names."
                )
            self._layer_to_group_idx = mapping  # type: ignore[assignment]
            # A model whose verify trace reads one page table per KV-cache group
            # (rather than per layer) learns the grouping before it allocates
            # and warms up; layers of one group share a block table.
            set_groups = getattr(self.model, "set_kv_layer_groups", None)
            if callable(set_groups):
                set_groups(list(mapping))

        self.kv_caches = self._allocate_kv_caches(kv_cache_config)

    @staticmethod
    def _validate_kv_cache_groups(kv_cache_groups: list) -> None:
        if not kv_cache_groups:
            raise ValueError("kv_cache_config has no groups")
        for group in kv_cache_groups:
            # ``UniformTypeKVCacheSpecs`` wraps several same-type (e.g. all
            # FullAttentionSpec) layer specs that differ only in shape. vLLM
            # emits it when a hybrid model disables hybrid kv-cache groups and
            # exposes heterogeneous per-layer KV shapes (e.g. Gemma4 with
            # ``_HYBRID_KV_CACHE_GROUPS_ENABLED = False``: sliding layers 8x256,
            # full layers 1x512). We unwrap it to per-layer specs in
            # ``_build_per_layer_specs``, so accept it here too.
            if not isinstance(
                group.kv_cache_spec, (AttentionSpec, UniformTypeKVCacheSpecs)
            ):
                raise TypeError(
                    f"Expected AttentionSpec/UniformTypeKVCacheSpecs for group "
                    f"{group.layer_names}, got {type(group.kv_cache_spec).__name__}"
                )

    def _kv_cache_shape(
        self, spec: AttentionSpec, num_blocks: int
    ) -> tuple[int, int, int, int]:
        """Per-buffer shape ``(num_blocks, num_kv_heads, block_size,
        head_size)`` from a group's attention spec.

        TP factor is folded in here because it is handled on the model
        side for TT (caches are replicated per submesh and each device
        carries ``num_kv_heads // tp`` heads internally).
        """
        num_devices = self.num_devices // self.tt_data_parallel_size
        num_kv_heads = spec.num_kv_heads // min(num_devices, spec.num_kv_heads)
        return (num_blocks, num_kv_heads, spec.block_size, spec.head_size)

    def _allocate_kv_caches(self, kv_cache_config: KVCacheConfig) -> Any:
        """Allocate KV cache tensors, falling back to legacy uniform API.

        Builds a ``per_layer_specs`` list of ``(shape, dtype)`` tuples — one
        entry per attention layer in model layer-index order. Hybrid models
        opt in to per-layer allocation by exposing
        ``allocate_kv_cache_per_layer(per_layer_specs)``; legacy models keep
        the older ``allocate_kv_cache(shape, dtype, num_layers)`` signature
        and we adapt to it here, asserting the per-layer specs are uniform.
        """
        num_layers = self.model_config.get_num_layers_by_block_type(
            self.parallel_config, "attention"
        )
        per_layer_specs = self._build_per_layer_specs(kv_cache_config, num_layers)

        if hasattr(self.model, "allocate_kv_cache_per_layer"):
            return self.model.allocate_kv_cache_per_layer(per_layer_specs)

        # Legacy ``allocate_kv_cache(shape, dtype, num_layers)`` API: every
        # layer must have the same shape/dtype. The third tuple element is
        # the tensor index, which is irrelevant for the legacy uniform
        # path (each layer gets its own buffer there).
        shape, dtype, _ = per_layer_specs[0]
        for entry_shape, entry_dtype, _ in per_layer_specs[1:]:
            if (entry_shape, entry_dtype) != (shape, dtype):
                raise NotImplementedError(
                    f"{type(self.model).__name__} only implements legacy "
                    "allocate_kv_cache; hybrid attention models must "
                    "override allocate_kv_cache_per_layer."
                )
        return self.model.allocate_kv_cache(shape, dtype, len(per_layer_specs))

    def _block_tables_per_layer(
        self, block_tables_per_group: list[torch.Tensor]
    ) -> list[torch.Tensor] | None:
        """Expand per-group block tables to per-layer using the cached mapping.

        Returns None for non-hybrid models (the mapping is only populated
        when the model class exposes ``get_kv_cache_spec``). The output is
        a list of ``num_layers`` tensors, where entry ``i`` is the block
        table for layer ``i``'s containing kv_cache_group — what hybrid
        bridges hand to the underlying TT model so attention layer ``i``
        can index its own paged pool (full vs. sliding-window) without
        knowing how vLLM ordered the groups.
        """
        if self._layer_to_group_idx is None:
            return None
        result = [block_tables_per_group[g] for g in self._layer_to_group_idx]
        # Pad to the warmup shape ``[max_batch, max_num_blocks_per_req]``.
        # The model side (``Transformer._page_tables_to_ttnn``) allocates
        # *persistent* ttnn device tensors at this shape during warmup so
        # captured traces can be replayed against stable device addresses;
        # ``ttnn.copy_host_to_device_tensor`` then asserts
        # ``host_shape == device_shape`` when the runtime per-layer block
        # tables push their content into those buffers. Padding rows with
        # zeros is harmless — the kernel only reads up to each layer's
        # active block count.
        max_batch = self.tt_max_batch_size
        target_shape = (max_batch, self.max_num_blocks_per_req)
        padded = []
        for bt in result:
            if bt is None:
                padded.append(None)
                continue
            if bt.shape == target_shape:
                padded.append(bt)
                continue
            full = torch.zeros(target_shape, dtype=bt.dtype)
            rows = min(bt.shape[0], target_shape[0])
            cols = min(bt.shape[1], target_shape[1])
            full[:rows, :cols] = bt[:rows, :cols]
            padded.append(full)
        return padded

    def _build_per_layer_specs(
        self, kv_cache_config: KVCacheConfig, num_layers: int
    ) -> list[tuple[tuple[int, int, int, int], Any, int]]:
        """Resolve ``KVCacheConfig`` → list of ``(shape, dtype, tensor_idx)``
        per layer in model layer-index order.

        ``tensor_idx`` identifies the unique DRAM buffer that each layer's
        KV cache lives in. Multiple layers from different
        ``KVCacheGroupSpec``\\ s can carry the same ``tensor_idx`` — this
        is upstream's tensor-sharing model: with a 5:1 sliding/full split,
        a full-attention layer and several sliding-window layers all share
        one buffer, and they index disjoint slots within it via per-group
        block tables (vLLM's ``BlockPool`` allocates disjoint block IDs
        across groups, so the shared tensor is sized for the worst-case
        full-attention demand and the sliding-window layers fit within
        their window's worth of slots).

        Single-group (uniform-attention) models keep the previous behavior:
        every layer gets a unique buffer and ``tensor_idx == layer_idx``.
        """
        kv_cache_groups = kv_cache_config.kv_cache_groups

        if len(kv_cache_groups) == 1:
            spec = kv_cache_groups[0].kv_cache_spec
            if isinstance(spec, AttentionSpec):
                shape = self._kv_cache_shape(spec, kv_cache_config.num_blocks)
                return [(shape, spec.dtype, i) for i in range(num_layers)]
            # ``UniformTypeKVCacheSpecs``: one group / one block table, but the
            # wrapped per-layer specs have heterogeneous shapes (e.g. Gemma4
            # with hybrid groups disabled: sliding 8x256 vs full 1x512). Every
            # layer indexes the same ``num_blocks`` block table but gets its own
            # DRAM buffer sized to its own spec, so ``tensor_idx == layer_idx``
            # (matching the uniform single-group convention above).
            assert isinstance(spec, UniformTypeKVCacheSpecs)
            per_layer: list[tuple[tuple[int, int, int, int], Any, int] | None] = [
                None
            ] * num_layers
            for layer_name, layer_spec in spec.kv_cache_specs.items():
                assert isinstance(layer_spec, AttentionSpec)
                idx = _parse_layer_index(layer_name)
                if not 0 <= idx < num_layers:
                    raise ValueError(
                        f"Layer index {idx} parsed from '{layer_name}' is out "
                        f"of range for {num_layers} attention layers"
                    )
                shape = self._kv_cache_shape(layer_spec, kv_cache_config.num_blocks)
                per_layer[idx] = (shape, layer_spec.dtype, idx)
            missing = [i for i, e in enumerate(per_layer) if e is None]
            if missing:
                raise ValueError(
                    f"UniformTypeKVCacheSpecs missing per-layer specs for "
                    f"layer indices {missing}"
                )
            return per_layer  # type: ignore[return-value]

        # Multi-group: every group's KVCacheTensor starts at byte 0, so layers
        # whose regions start at the same byte share one buffer -- the i-th
        # layer of each group, as ``shared_by`` listed before vLLM 0.29.
        spec_by_layer_name: dict[str, AttentionSpec] = {}
        for group in kv_cache_groups:
            assert isinstance(group.kv_cache_spec, AttentionSpec)
            for layer_name in group.layer_names:
                spec_by_layer_name[layer_name] = group.kv_cache_spec

        per_layer: list[tuple[tuple[int, int, int, int], Any, int] | None] = [
            None
        ] * num_layers
        tensor_idx_by_start: dict[int, int] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            for position, layer_name in enumerate(kv_cache_tensor.layers):
                start = kv_cache_tensor.offset + position * kv_cache_tensor.layer_stride
                tensor_idx = tensor_idx_by_start.setdefault(
                    start, len(tensor_idx_by_start)
                )
                spec = spec_by_layer_name.get(layer_name)
                if spec is None:
                    raise ValueError(
                        f"KVCacheTensor.layers names layer '{layer_name}' "
                        "but it doesn't appear in any kv_cache_group"
                    )
                idx = _parse_layer_index(layer_name)
                if not 0 <= idx < num_layers:
                    raise ValueError(
                        f"Layer index {idx} parsed from '{layer_name}' is "
                        f"out of range for {num_layers} attention layers"
                    )
                if per_layer[idx] is not None:
                    raise ValueError(
                        f"Layer index {idx} (from '{layer_name}') is named "
                        "by more than one KVCacheTensor; each layer must "
                        "map to exactly one DRAM buffer"
                    )
                shape = self._kv_cache_shape(spec, kv_cache_config.num_blocks)
                per_layer[idx] = (shape, spec.dtype, tensor_idx)

        missing = [i for i, e in enumerate(per_layer) if e is None]
        if missing:
            raise ValueError(
                f"No KVCacheTensor covers layer indices {missing}; "
                "every attention layer must appear in some "
                "kv_cache_tensors[i].layers"
            )
        return per_layer  # type: ignore[return-value]

    def _release_model_request(self, req_id: str) -> None:
        """Release model-owned state while the slot mapping is still valid.

        Released by the request's CURRENT slot, which is the identity the model
        holds: it was handed the slot at prefill (``empty_slots``) and is told
        about every subsequent gather through ``note_state_slots_moved``, so the
        two agree at all times.

        Releasing by the PREFILL slot instead does not work, even though it
        survives the gather: slots are reused, so one request's prefill slot can
        be another live request's current owner slot, and the release then clears
        the newcomer's session (vllm-tt-plugin#118 review, finding 1). See
        ``_notify_model_slot_moves``.
        """
        slot = self._req_state_slot.get(req_id)
        release = getattr(getattr(self, "model", None), "release_request", None)
        if slot is not None and callable(release):
            release(slot)

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        """Update the cached states and the persistent batch with the
        scheduler output.
        The updated states are used in `_prepare_model_inputs` to create the
        input tensors for the model.
        Based on _update_states for GPU/TPU backends.
        """
        persistent_batch_layout_changed = False

        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        # Upstream input processing gives resubmissions fresh internal IDs, but
        # keep the update order robust for an internal caller that bypasses it:
        # a finished and newly scheduled duplicate is treated as two requests,
        # clearing the old cached state before adding the new one.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            self._release_model_request(req_id)
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)
                persistent_batch_layout_changed = True

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Explicit preemption invalidates model-owned request state. Temporary
        # unscheduling does not: a request may simply be hidden for this step.
        for req_id in scheduler_output.preempted_req_ids or ():
            self._release_model_request(req_id)

        # Only after the model released by slot: the release calls above read
        # the request's slot from the ownership map before it is dropped.
        self._release_dead_state_slots(scheduler_output)
        if self.tt_admitted_request_limit != self.tt_max_batch_size:
            owners = set(self._req_state_slot) | set(scheduler_output.num_scheduled_tokens)
            if len(owners) > getattr(self, "_gemma_admission_highwater", 0):
                self._gemma_admission_highwater = len(owners)
                logger.info("GEMMA_ADMISSION_HIGHWATER owners=%d admitted=%d physical=%d", len(owners), self.tt_admitted_request_limit, self.tt_max_batch_size)
            if len(owners) > self.tt_admitted_request_limit:
                raise RuntimeError("Gemma live plus scheduled request ownership exceeds admission")

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
            persistent_batch_layout_changed = True

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            self.requests[req_id] = build_cached_request_state(new_req_data)
            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids

            # Update the cached states.
            apply_cached_req_state_update(
                req_state, num_computed_tokens, new_block_ids, resumed_from_preemption
            )

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
            # Fill the empty index, or append to the end.
            req_index = removed_req_indices.pop() if removed_req_indices else None
            self.input_batch.add_request(req_state, req_index)
            persistent_batch_layout_changed = True

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)
            persistent_batch_layout_changed = True
        # Mark decode layout changed if persistent batch changed. This is
        # sticky across steps and will be consumed by the next decode batch.
        if persistent_batch_layout_changed:
            self._decode_layout_changed_since_last_decode = True

        # Refresh logits processors with batch state changes
        self.input_batch.refresh_logitsprocs()

    def _validate_mm_feature(self, mm_feature: MultiModalFeatureSpec) -> None:
        """Validate the multimodal feature is an image."""
        if mm_feature.modality != "image":
            raise NotImplementedError("Only images are supported for now")

    def _gather_multi_modal_inputs(
        self, req_indices: list[int] | None = None
    ) -> dict[str, Any]:
        """
        Gather and batch multi-modal inputs for the current persistent batch.

        Currently only supports image inputs in the "pixel_values" and
        "image_grid_thw" fields.

        Returns a dict with keys "pixel_values" and "image_grid_thw".
        Each value is a list aligned with the persistent batch order
        (`self.input_batch.req_ids[:num_reqs]`).

        For request i:
        - If it has no `mm_features`, the entry is None.
        - Otherwise the entry is a list aligned with that request's
          `mm_features` (currently only images), where each element is a
          tensor (or None if that feature has no data).

        Example (3 scheduled requests: text-only, 1 image, 2 images):
        {
          "pixel_values": [
            None,
            [pv_req1_img0],
            [pv_req2_img0, pv_req2_img1],
          ],
          "image_grid_thw": [
            None,
            [ig_req1_img0],
            [ig_req2_img0, ig_req2_img1],
          ],
        }
        """

        multi_modal_kwargs: dict[str, Any] = {
            "pixel_values": [],
            "image_grid_thw": [],
        }

        if req_indices is None:
            req_indices = list(range(self.input_batch.num_reqs))
        # The model input tensors are built in persistent batch order, so
        # multi-modal inputs must follow the same order (not just new reqs).
        for batch_index in req_indices:
            req_id = self.input_batch.req_ids[batch_index]
            req_state = self.requests[req_id]

            if not req_state.mm_features:
                multi_modal_kwargs["pixel_values"].append(None)
                multi_modal_kwargs["image_grid_thw"].append(None)
                continue

            pv_array: list[torch.Tensor | None] = []
            image_grid_thw_array: list[torch.Tensor | None] = []
            for mm_feature in req_state.mm_features:
                self._validate_mm_feature(mm_feature)
                item = mm_feature.data
                if item is None:
                    pv_array.append(None)
                    image_grid_thw_array.append(None)
                    continue
                pv_array.append(item["pixel_values"].data)
                image_grid_thw_array.append(
                    item["image_grid_thw"].data if "image_grid_thw" in item else None
                )

            multi_modal_kwargs["pixel_values"].append(pv_array)
            multi_modal_kwargs["image_grid_thw"].append(image_grid_thw_array)

        return multi_modal_kwargs

    def _sampling_params_for_padded_decode(
        self,
        sample_params,
        req_indices: list[int],
        pad_to: int,
    ) -> TTSamplingParams:
        """Build per-row sampling tensors padded to ``pad_to``.

        Decode inputs are padded to the wire capacity, so the sampling rows must
        be padded to match; the padding rows carry neutral defaults.
        """
        defaults = sample_params.create_default_tensors()

        def take(name: str) -> torch.Tensor:
            local = getattr(sample_params, name)[req_indices]
            if local.shape[0] >= pad_to:
                return local[:pad_to]
            pad_count = pad_to - local.shape[0]
            return torch.cat([local, defaults[name][:pad_count]])

        num_logprobs = take("num_logprobs")
        return TTSamplingParams(
            temperature=take("temperature"),
            top_k=take("top_k"),
            top_p=take("top_p"),
            presence_penalty=take("presence_penalty"),
            frequency_penalty=take("frequency_penalty"),
            repetition_penalty=take("repetition_penalty"),
            seed=take("seed"),
            num_logprobs=num_logprobs,
            enable_log_probs=num_logprobs >= 0,
        )

    # --- Per-request device state slots (see ``self._req_state_slot``) ---

    def _release_dead_state_slots(self, scheduler_output: SchedulerOutput) -> None:
        """Drop the slot claims of requests whose device state is no longer
        authoritative.

        The predicate is not "did the step schedule it": a RUNNING request the step
        left out (what every prefill step does to the whole decode batch) still owns
        live state and must keep its slot. Only two states release:

        - FINISHED: the request is gone.
        - PREEMPTED: ``Scheduler._preempt_request`` freed the KV blocks and reset
          ``num_computed_tokens``, so a resume re-prefills the prompt plus every
          generated token and writes the slot's final state itself. Nothing reads the
          old contents, and holding the slot only inflates ``held`` in
          ``_alloc_prefill_state_slots``, where a shortfall is fatal. The
          per-slot seed RNG does not need the hold either: the device seed is derived
          from the absolute decode position, not from slot residency.

        Both of ``_preempt_request``'s callers reach here: the scheduler's own
        running-loop, and the wholesale teardown in ``reset_prefix_cache`` when it
        is asked to reset running requests, whose preemptions surface because the
        scheduler keeps ``preempted_req_ids`` across the step boundary rather than
        rebuilding it per call.

        ``preempted_req_ids`` is typed optional, hence the ``or ()``.
        """
        for req_id in scheduler_output.finished_req_ids:
            self._req_state_slot.pop(req_id, None)
        for req_id in scheduler_output.preempted_req_ids or ():
            self._req_state_slot.pop(req_id, None)

    def _alloc_prefill_state_slots(self, row_req_ids: list[str]) -> list[int]:
        """Keep live prefill slots stable and allocate only new ownership.

        A continuing chunk may appear at a different compact batch row. Moving
        its slot here would leave model state behind: only decode gathers move
        state and notify the model. Reserve all existing owners before assigning
        a fresh request, then prefer its row when that slot is free.

        Exhaustion is unreachable: existing holders and fresh prefills both count
        against ``max_num_seqs``, which is ``n_slots``. Getting here means the map has
        stopped describing the device, and nothing on the host can recover it, so fail
        rather than guess and return plausible, wrong text.
        """
        n_slots = self.tt_per_lane_max_num_seqs
        if len(row_req_ids) > n_slots:
            raise RuntimeError(
                f"{len(row_req_ids)} prefill(s) exceed the {n_slots} device state "
                "slots; admission is the scheduler's job, not this function's"
            )
        prefilling = set(row_req_ids)
        existing = {
            req_id: slot
            for req_id, slot in self._req_state_slot.items()
            if req_id in prefilling or req_id in self.requests
        }
        held = set(existing.values())
        if len(held) != len(existing) or any(not 0 <= s < n_slots for s in held):
            raise RuntimeError(f"invalid device state slot ownership: {existing}")
        slots: list[int] = []
        for row, req_id in enumerate(row_req_ids):
            if req_id in existing:
                slots.append(existing[req_id])
                continue
            if row not in held:
                slot = row
            else:
                free = [s for s in range(n_slots) if s not in held]
                if not free:
                    # A raise, not an assert: the message is the only way to tell a
                    # capacity shortfall from a leak in the ownership map, and ``-O``
                    # would drop an assert and leave an opaque IndexError below.
                    raise RuntimeError(
                        f"no free device state slot for {len(row_req_ids)} prefill(s): "
                        f"held={sorted(held)}, capacity={n_slots}, "
                        f"map={self._req_state_slot}"
                    )
                slot = free[0]
            held.add(slot)
            self._req_state_slot[req_id] = slot
            slots.append(slot)
        return slots

    def _decode_state_slot_remap(self, row_req_ids: list[str]) -> torch.Tensor | None:
        """Gather permutation taking each request's state to its decode row: row
        ``i`` reads slot ``remap[i]``. Always full slot width (no OOB gather); None
        means identity, so skip it. The resulting ownership map remains pending
        until ``decode_forward`` accepts the submission."""
        n_slots = self.tt_per_lane_max_num_seqs
        # More decode rows than slots means the batch cannot be described at all, so
        # truncating would just drop a request's state silently.
        if len(row_req_ids) > n_slots:
            raise RuntimeError(
                f"{len(row_req_ids)} decode row(s) exceed the {n_slots} device state "
                "slots"
            )
        want: list[int] = []
        for req_id in row_req_ids:
            slot = self._req_state_slot.get(req_id)
            # Every decoding request got a slot at prefill. Inventing one here is how
            # a second request ends up recorded at an owned slot.
            if slot is None:
                raise RuntimeError(
                    f"decoding request {req_id!r} has no device state slot: "
                    f"map={self._req_state_slot}"
                )
            want.append(slot)
        if len(set(want)) != len(want) or any(not 0 <= s < n_slots for s in want):
            # Refusing the remap is not the safe option: no gather goes out for the
            # WHOLE batch, so every off-row request reads another's state.
            duplicates = sorted({s for s in want if want.count(s) > 1})
            raise RuntimeError(
                f"TT decode state slots are not a permutation: want={want}, "
                f"duplicated={duplicates}, capacity={n_slots}, "
                f"map={self._req_state_slot}"
            )
        taken = set(want)
        remap = want + [s for s in range(n_slots) if s not in taken]
        # The gather moves every slot, not just the batch rows: ownership must follow.
        # Build the new map whole and swap it in, so no read sees it half-written.
        by_slot: dict[int, list[str]] = {}
        for req_id, slot in self._req_state_slot.items():
            by_slot.setdefault(slot, []).append(req_id)
        moved = dict(self._req_state_slot)
        for row, slot in enumerate(remap):
            for req_id in by_slot.get(slot, ()):
                moved[req_id] = row
        self._pending_state_slot_settle = moved
        # Slot-level view of the SAME permutation, for the model's own per-slot
        # state: row ``i`` reads slot ``remap[i]``, so slot ``remap[i]`` becomes
        # slot ``i``. Handed over whole, not applied one pair at a time -- a
        # sequential walk over a permutation can move the same state twice.
        self._pending_state_slot_moves = {
            remap[i]: i for i in range(n_slots) if remap[i] != i
        }
        if all(remap[i] == i for i in range(n_slots)):
            self._pending_state_slot_settle = None
            self._pending_state_slot_moves = None
            return None
        return torch.tensor(remap, dtype=torch.int32)

    def note_decode_state_slots_settled(self) -> None:
        """Commit a remap only after the model accepted the decode.

        The model is told about the move here, for the same reason the commit
        happens here: a gather that the model refused never took effect on the
        device, so telling it earlier would leave its owner slot describing a
        permutation that did not happen.
        """
        if self._pending_state_slot_settle is not None:
            self._req_state_slot = self._pending_state_slot_settle
            self._pending_state_slot_settle = None
        moves = self.__dict__.get("_pending_state_slot_moves")
        if moves:
            _notify_model_slot_moves(self, moves)
        self._pending_state_slot_moves = None

    def note_decode_layout_consumed(self) -> None:
        """Retire the sticky layout transition after an accepted decode."""
        self._decode_layout_changed_since_last_decode = False

    @staticmethod
    def _build_host_generators(
        input_batch: InputBatch,
        req_indices: list[int],
        intermediate_prefill_mask: torch.Tensor | None,
    ) -> dict[int, torch.Generator]:
        """Re-key generators (batch row -> Generator) to this build's rows.

        The sampler advances the request's live generator. Preparing inputs
        must not consume random values before an output is sampled.

        An intermediate-prefill row's token is discarded, so it gets a clone
        and its request's real RNG state is left untouched.
        """
        intermediate_rows = (
            set(intermediate_prefill_mask.nonzero().view(-1).tolist())
            if intermediate_prefill_mask is not None
            else set()
        )
        generators: dict[int, torch.Generator] = {}
        for local_row, batch_row in enumerate(req_indices):
            generator = input_batch.sampling.generators.get(batch_row)
            if generator is None:
                continue
            if local_row in intermediate_rows:
                generators[local_row] = clone_torch_generator(generator)
            else:
                generators[local_row] = generator
        return generators

    def _drafts_to_verify(
        self,
        scheduler_output: SchedulerOutput,
        row_req_ids: list[str],
    ) -> dict[str, list[int]]:
        """The drafts this step verifies, by request.

        Synchronously, the scheduler's own list, and that is the whole story:
        a proposer reports through ``take_draft_token_ids``, upstream stores
        the ids on the request, budgets and grammar-checks them, and delivers
        what survived.

        Asynchronously, nothing arrives that way. ``EngineCore.post_step``
        skips ``take_draft_token_ids`` entirely when asynchronous scheduling is
        on, because a step's drafts are not known before it runs, and upstream
        substitutes its own ids inside the worker instead.
        ``AsyncScheduler`` still schedules the lookahead: it leaves each
        scheduled request holding ``[-1] * num_spec_tokens_to_schedule``, which
        is a reservation of that many positions rather than a proposal. So the
        placeholders are read as the reservation they are, and the proposal the
        runner published for that request is what gets verified, trimmed to
        what the scheduler reserved.

        The runner's entry is consumed here, keeping the rule the synchronous
        path gets from ``take_draft_token_ids``: a proposal is handed over
        once, so a row that proposed nothing this step verifies nothing next
        step rather than replaying a spent proposal.
        """
        scheduled = scheduler_output.scheduled_spec_decode_tokens
        if not self.async_decode_scheduling:
            return scheduled
        drafts: dict[str, list[int]] = {}
        for req_id in row_req_ids:
            offered = list(scheduled.get(req_id) or ())
            placeholders = sum(1 for token in offered if token == PLACEHOLDER_TOKEN_ID)
            if placeholders and placeholders != len(offered):
                raise RuntimeError(
                    f"request {req_id} was scheduled a mix of placeholder and "
                    f"real draft tokens: {offered}. A scheduled list is either "
                    "the asynchronous lookahead reservation or a proposal, and "
                    "the runner cannot tell which half to verify"
                )
            if not placeholders:
                # Real ids, or nothing scheduled at all. Either way this is
                # what the scheduler means to verify.
                if offered:
                    drafts[req_id] = offered
                self._proposed_draft_token_ids.pop(req_id, None)
                continue
            proposed = self._proposed_draft_token_ids.pop(req_id, None)
            if proposed:
                drafts[req_id] = list(proposed[:placeholders])
        return drafts

    @staticmethod
    def _spec_row_state(
        accepted_counts_by_req: dict[str, int],
        scheduled_drafts: dict[str, list[int]],
        row_req_ids: list[str],
        num_drafts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather ``(drafts, num_valid_drafts, accepted_counts)`` for these rows.

        The two halves come from different owners, deliberately.

        ``scheduled_drafts`` contains actual IDs resolved by
        ``_drafts_to_verify``. Synchronous execution receives scheduler-owned
        IDs after the proposal handoff. Asynchronous execution substitutes
        retained runner proposals within the scheduler's placeholder
        reservation. Both paths obey the scheduled per-request draft budget;
        structured speculative requests are not admitted.

        The **accepted count** is the runner's, because nothing upstream models
        it. It is keyed by request id rather than by row because a running
        request that one step does not schedule leaves the persistent batch and
        comes back, and only an explicit preemption releases the candidate
        state the count selects.

        A request absent from either mapping takes the post-prefill default: no
        drafts, and a count of 1.
        """
        drafts = torch.full(
            (len(row_req_ids), num_drafts), PLACEHOLDER_TOKEN_ID, dtype=torch.int32
        )
        num_valid = torch.zeros(len(row_req_ids), dtype=torch.int32)
        counts = torch.ones(len(row_req_ids), dtype=torch.int32)
        for row, req_id in enumerate(row_req_ids):
            counts[row] = accepted_counts_by_req.get(req_id, 1)
            row_drafts = scheduled_drafts.get(req_id) or ()
            valid = len(row_drafts)
            if valid > num_drafts:
                # The scheduler truncates to the lookahead it budgeted, which
                # the platform publishes as num_speculative_tokens, so a longer
                # list means the two disagree about the draft length. Raised
                # rather than truncated, because silently dropping the tail
                # would verify a prefix at positions the block was built for.
                raise RuntimeError(
                    f"request {req_id} was scheduled {valid} draft tokens, "
                    f"above the block's {num_drafts}: {list(row_drafts)}"
                )
            num_valid[row] = valid
            if valid:
                drafts[row, :valid] = torch.tensor(row_drafts, dtype=torch.int32)
        return drafts, num_valid, counts

    def _publish_draft(self, req_id: str, tokens) -> None:
        """Record this step's proposal for ``req_id``, or erase the last one.

        Every proposer publishes through here so the speculability gate cannot
        be bypassed by adding one: there is a single accept walk behind all of
        them, and it certifies by id equality only.

        Erasing on an empty offer is not optional. This map is what the
        scheduler is told to verify next, so an entry nothing rewrote would be
        verified against a token the request has already moved past.
        """
        if tokens and self._request_is_speculable(req_id):
            self._proposed_draft_token_ids[req_id] = list(tokens)
        else:
            self._proposed_draft_token_ids.pop(req_id, None)

    def _request_is_speculable(self, req_id: str) -> bool:
        """False when this request's sampling cannot be certified by id equality.

        ``accept_greedy_drafts`` compares token ids, so it can only certify a
        greedy continuation. Anything that makes the target distribution differ
        from argmax -- a non-zero temperature, or a penalty that reshapes the
        logits -- has to be decoded ordinarily. top_p/top_k need no entry: they
        only narrow a distribution that temperature 0 has already collapsed to a
        point, so a greedy request carrying them is still greedy.
        """
        batch = self.input_batch
        # A model declaring ``spec_sampled_verify`` returns, for every verify
        # row, a token sampled with that request's own temperature/top-k/top-p
        # and an independent per-position seed. Drafts here are deterministic,
        # so accepting while the sample equals the draft and committing the first
        # mismatching sample is exact speculative sampling: a sampled request is
        # speculable. Penalties still route sampling to the host and stay out.
        sampled_verify = bool(
            (getattr(type(self.model), "model_capabilities", None) or {}).get(
                "spec_sampled_verify", False
            )
        )
        # getattr rather than attribute access: the gate has to stay total for
        # any batch the runner is handed. A batch that tracks none of these
        # sets has no sampled request to withhold drafts from, so the answer is
        # the same as an empty set -- speculable.
        for attr in (
            *(() if sampled_verify else ("random_reqs",)),
            "presence_penalties_reqs",
            "frequency_penalties_reqs",
            "repetition_penalties_reqs",
        ):
            if req_id in getattr(batch, attr, ()):
                return False
        return True

    def _note_unspeculable_verify_rows(self, row_req_ids: list[str]) -> None:
        """Count the live rows of a verify whose request is not speculable.

        ``_publish_draft`` keeps drafts off such a request, but it cannot keep
        the request out of a verify: another row's drafts, an unresolved
        multi-token commit, or a model without ``supports_narrow_decode`` makes
        the whole step one. A verify in ``argmax_ids`` mode answers every row
        with the target's argmax, so the row commits that argmax, not a token
        drawn with its request's temperature and penalties.
        """
        rows = sum(
            1 for req_id in row_req_ids if not self._request_is_speculable(req_id)
        )
        if not rows:
            return
        self._num_unspeculable_verify_rows += rows
        logger.warning_once(
            "A speculative verify step included a request that is not "
            "speculable (non-zero temperature or a penalty). A verify returns "
            "the target argmax for every row, so on that step the request "
            "committed the argmax and its temperature and penalties were not "
            "applied. The total count is logged at shutdown."
        )

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        """Hand the drafts proposed since the last call to the engine.

        On the synchronous path, ``EngineCore`` collects proposals and passes
        them to ``Scheduler.update_draft_token_ids``. The supported async path
        instead consumes retained proposals in ``_drafts_to_verify`` within
        the scheduled lookahead reservation. Returning ``None`` reports that
        no pending proposal is available for handoff.
        """
        if not self._proposed_draft_token_ids:
            return None
        # Only requests the runner still holds. A request can finish in the
        # step that proposed for it, and the runner drops it from
        # ``self.requests`` before the engine collects; the scheduler tolerates
        # a draft for a request it has finished, but the runner has no business
        # reporting one for a request it has forgotten.
        proposed = self._proposed_draft_token_ids
        self._proposed_draft_token_ids = {}
        req_ids = [req_id for req_id in proposed if req_id in self.requests]
        if not req_ids:
            return None
        drafts = [proposed[req_id] for req_id in req_ids]
        return DraftTokenIds(req_ids, drafts)

    def _propose_ngram_drafts(self, committed: dict[str, list[int]]) -> None:
        """Propose the next step's drafts from the tokens just committed.

        The n-gram proposer searches the request's own text for a repeat of its
        most recent tokens and drafts what followed last time, so it reads the
        committed history and needs no model and no device. It runs after the
        commit for that reason: ``token_ids_cpu`` and ``num_tokens`` have to
        already hold this step's tokens.

        Rows are the persistent batch's, because that is what the proposer
        indexes; the result is re-keyed by request id, because that is what the
        scheduler indexes.
        """
        if self._spec_method != "ngram" or not committed:
            return
        if self._ngram_proposer is None:
            from vllm.v1.spec_decode.ngram_proposer import NgramProposer

            self._ngram_proposer = NgramProposer(self.vllm_config)

        num_reqs = self.input_batch.num_reqs
        row_req_ids = list(self.input_batch.req_ids[:num_reqs])
        # An empty list tells the proposer to skip that row, which is what a
        # request that committed nothing this step needs.
        sampled = [committed.get(req_id, []) for req_id in row_req_ids]
        drafts = self._ngram_proposer.propose(
            self._num_speculative_tokens,
            sampled,
            self.input_batch.num_tokens[:num_reqs],
            self.input_batch.token_ids_cpu,
        )
        for req_id, row_drafts in zip(row_req_ids, drafts):
            # A row the proposer found no repeat for drafts nothing this step,
            # and a request whose sampling the accept walk cannot certify is
            # not drafted for at all; _publish_draft handles both by erasing.
            self._publish_draft(req_id, row_drafts)

    def _propose_model_drafts(
        self,
        committed: torch.Tensor,
        counts: torch.Tensor,
        model_input: TTModelInput,
        hidden: Any,
        row_req_ids: list[str],
        *,
        skip_req_ids: set[str] | None = None,
    ) -> None:
        """Ask the model's own drafter for the next step's drafts.

        The drafter continues each row from the token that row just committed,
        so it is handed the committed block and each row's count rather than a
        single token: which entry of the block is that row's last is
        ``counts - 1``, the same arithmetic the verify uses to pick a candidate
        state slot. Rows are the verify's rows, padding included, because the
        drafter's state is indexed by row and a device graph has one shape.

        ``hidden`` is the verify's own handle, passed straight back. The runner
        does not interpret it, and a drafter that needs none is handed whatever
        its verify returned, which for a model that returns nothing is
        ``None``.
        """
        num_drafts = self._num_speculative_tokens
        width = num_drafts + 1
        rows = int(committed.shape[0])
        if int(committed.shape[1]) < width:
            # A narrow verify answers one column wide, because the model
            # declared it serves the plain decode's shapes on a step where no
            # row carries a draft. The drafter is a separate call with one
            # shape of its own, so the block is padded to the uniform 1+K,
            # which is what a row that committed less than the full width
            # already looks like after a wide step.
            committed = torch.cat(
                [
                    committed,
                    torch.full(
                        (rows, width - int(committed.shape[1])),
                        PLACEHOLDER_TOKEN_ID,
                        dtype=committed.dtype,
                    ),
                ],
                dim=1,
            )
        if model_input.spec_mode is None:
            positions = self._committed_positions_from_state(
                row_req_ids,
                counts,
                submitted_positions=model_input.input_positions,
                width=width,
            )
        else:
            # A verify has authoritative submitted positions. Its physical
            # acceptance count can exceed the prefix retained at max_model_len.
            positions = self._committed_positions(model_input.input_positions, width)
        history_kwargs = {}
        if (getattr(type(self.model), "model_capabilities", None) or {}).get(
            "spec_token_history", False
        ):
            # Each row's full token history (prompt, including prefix-cached
            # tokens, and every committed output token), as a host view of the
            # persistent batch's token buffer. Rows without a live request get
            # None. Read after the commit, so the newest tokens are present.
            histories = []
            batch = self.input_batch
            for row in range(rows):
                req_id = row_req_ids[row] if row < len(row_req_ids) else None
                batch_row = (
                    batch.req_id_to_index.get(req_id) if req_id is not None else None
                )
                if batch_row is None:
                    histories.append(None)
                else:
                    histories.append(
                        batch.token_ids_cpu[batch_row, : int(batch.num_tokens[batch_row])]
                    )
            history_kwargs["token_history"] = histories
        drafted = self.model.propose_draft_tokens(
            num_drafts,
            committed,
            positions,
            counts,
            hidden=hidden,
            **history_kwargs,
        )
        if not isinstance(drafted, DraftOutput):
            raise TypeError(
                f"TT model {type(self.model).__name__} declares "
                f"{SPEC_REQUIREMENT_DEVICE_PROPOSE!r} and its "
                "propose_draft_tokens returned "
                f"{type(drafted).__name__}; a drafter returns a DraftOutput "
                "from vllm_tt_plugin.spec_decode"
            )
        draft_token_ids = drafted.draft_token_ids
        if draft_token_ids.dtype != torch.int32:
            # Checked before the range test below, which a fractional value
            # passes: the ids are read out with ``int()``, so a float tensor
            # would have every draft silently truncated and the scheduler would
            # verify a token the drafter never proposed.
            raise ValueError(
                f"TT model {type(self.model).__name__} proposed drafts of "
                f"dtype {draft_token_ids.dtype}; propose_draft_tokens returns "
                "int32 token ids, as the candidate block and both side "
                "tensors do"
            )
        if tuple(draft_token_ids.shape) != (rows, num_drafts):
            raise ValueError(
                f"TT model {type(self.model).__name__} proposed drafts of "
                f"shape {tuple(draft_token_ids.shape)}; the contract is "
                f"[{rows}, {num_drafts}], one row per verified row"
            )
        offered = drafted.num_valid
        if offered is not None:
            if offered.dtype != torch.int32:
                raise ValueError(
                    f"TT model {type(self.model).__name__} returned "
                    f"num_valid of dtype {offered.dtype}; propose_draft_tokens "
                    "returns int32 counts, as both verify side tensors do"
                )
            if tuple(offered.shape) != (rows,):
                raise ValueError(
                    f"TT model {type(self.model).__name__} returned "
                    f"num_valid of shape {tuple(offered.shape)}; the contract "
                    f"is [{rows}], one count per verified row"
                )
            if bool(((offered < 0) | (offered > num_drafts)).any()):
                raise ValueError(
                    f"TT model {type(self.model).__name__} returned a "
                    f"num_valid outside [0, {num_drafts}]: "
                    f"{offered.tolist()}; a row offers at most the draft "
                    "length the plan resolved, and 0 to offer nothing"
                )
        # Range-checked before the ids reach the scheduler, because a draft it
        # stores is verified next step and committed if the model agrees with
        # it, and an id outside the vocabulary is not something any verify can
        # have chosen. Only the offered entries are checked: a row's unoffered
        # tail is never read, so a drafter may pad it with the placeholder or
        # with anything else.
        vocab = int(self.input_batch.vocab_size)
        if offered is None:
            out_of_range = (draft_token_ids < 0) | (draft_token_ids >= vocab)
        else:
            columns = torch.arange(num_drafts).unsqueeze(0)
            is_offered = columns < offered.to(torch.int64).unsqueeze(1)
            out_of_range = (
                (draft_token_ids < 0) | (draft_token_ids >= vocab)
            ) & is_offered
        if bool(out_of_range.any()):
            raise ValueError(
                f"TT model {type(self.model).__name__} proposed a draft token "
                f"id outside [0, {vocab}); a drafter offering nothing for a "
                "row says so with num_valid, and the ids in that row are then "
                "not read"
            )
        max_model_len = int(self.model_config.max_model_len)
        skipped = set(skip_req_ids or ())
        for row, req_id in enumerate(row_req_ids):
            # A row whose committed prefix was not applied has nothing to
            # continue: a forced prefix-cache reset discards those tokens, and
            # a draft of what followed them would be verified against the
            # history vLLM replays instead.
            if req_id in skipped:
                continue
            # A row that committed nothing this step proposes nothing: there is
            # no continuation to draft from.
            if int(counts[row]) < 1:
                continue
            # Two row spaces meet here. ``counts`` and ``draft_token_ids`` are
            # indexed by the step's rows, which is the order the model answered
            # in; the persistent batch is indexed by where the request sits
            # now, and a completion between this step's submission and its
            # commit makes ``condense`` slide a request into another row. A
            # request with no row has no current length for max-model-length
            # trimming, so this step records its retained accepted count but
            # does not publish a proposal.
            batch_row = self.input_batch.req_id_to_index.get(req_id)
            if batch_row is None:
                continue
            # A request whose sampling the accept walk cannot arbitrate is
            # served WITHOUT speculation rather than refused. The walk compares
            # token ids and never sees logits, so it can only certify a greedy
            # continuation; proposing for a sampled request and accepting on id
            # equality would silently return greedy text for a request that
            # asked to sample. Offering nothing instead routes the request down
            # the ordinary decode path, where the runner applies the full
            # sampling it already implements (temperature, top_p/top_k, the
            # penalties, seeds), so the caller gets what it asked for at
            # baseline speed. Lossless speculation for these requests needs the
            # target probabilities (rejection sampling), not this walk.

            # Trimmed to what the request can still hold, the way the host
            # proposer trims itself. Drafts past ``max_model_len`` would be
            # verified and then dropped at the commit.
            room = max_model_len - int(self.input_batch.num_tokens[batch_row])
            row_offered = num_drafts if offered is None else int(offered[row])
            usable = max(0, min(row_offered, room))
            self._publish_draft(
                req_id, [int(token) for token in draft_token_ids[row, :usable]]
            )

    @staticmethod
    def _committed_positions(input_positions: torch.Tensor, width: int) -> torch.Tensor:
        """Absolute positions of one step's committed block.

        The verify's input column 0 holds each row's last committed token at
        its own position, and the verify's return column ``j`` is the model's
        choice at candidate position ``j``, which follows that input. So the
        committed block starts one position past the input's first column and
        runs consecutively, whatever each row accepted.

        Padding rows carry an input position of -1 and come out negative here,
        which is what marks them as rows no request owns.
        """
        first = (
            input_positions if input_positions.dim() == 1 else input_positions[:, 0]
        ).to(torch.int32)
        offsets = torch.arange(1, width + 1, dtype=torch.int32)
        positions = first.unsqueeze(1) + offsets.unsqueeze(0)
        # A padding row's input position is -1, and the offsets would turn that
        # into 0, 1, 2 and so on, which is exactly where a request starting
        # from nothing would sit. The drafter is handed no live-row mask, so the
        # position is the only thing that marks a row as owned by no request.
        return positions.masked_fill((first < 0).unsqueeze(1), -1)

    def _committed_positions_from_state(
        self,
        row_req_ids: list[str],
        counts: torch.Tensor,
        *,
        submitted_positions: torch.Tensor,
        width: int,
    ) -> torch.Tensor:
        """Build committed positions from history after each applied result."""
        positions = self._committed_positions(submitted_positions, width)
        offsets = torch.arange(width, dtype=torch.int32)
        requests = getattr(self, "requests", {})
        for row, req_id in enumerate(row_req_ids):
            req_state = requests.get(req_id)
            count = int(counts[row])
            if req_state is None or count < 1:
                continue
            first = req_state.num_tokens - count
            positions[row] = first + offsets
        return positions

    @staticmethod
    def _spec_candidate_block(
        drafts: torch.Tensor,
        num_valid: torch.Tensor,
        input_tokens: torch.Tensor,
        input_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Widen a ``[B, 1]`` decode into the uniform ``[B, 1+K]`` block.

        Column 0 is the row's last committed token at its own position, which
        is the token a plain decode would have sent; columns 1..K are that
        row's pending drafts at the K positions that follow. A row with fewer
        than K valid drafts pads its tail columns, and a token there is
        ``PLACEHOLDER_TOKEN_ID`` while a position there is -1, which is the
        same no-position marker the unused rows of a padded decode carry. A
        padded column must therefore never be mistaken for a real candidate,
        which is what ``num_valid_drafts`` tells the model.
        """
        num_drafts = drafts.shape[1]
        columns = torch.arange(num_drafts, dtype=torch.int32)
        # True where a column lies past that row's valid draft count.
        padded_draft = columns.unsqueeze(0) >= num_valid.unsqueeze(1)
        # Column 0 is committed, so it is never padded.
        padded = torch.cat(
            [torch.zeros(drafts.shape[0], 1, dtype=torch.bool), padded_draft], dim=1
        )

        tokens = torch.cat([input_tokens, drafts], dim=1)
        tokens = torch.where(
            padded, torch.full_like(tokens, PLACEHOLDER_TOKEN_ID), tokens
        )
        offsets = torch.arange(1 + num_drafts, dtype=input_positions.dtype)
        positions = input_positions.unsqueeze(1) + offsets.unsqueeze(0)
        positions = torch.where(padded, torch.full_like(positions, -1), positions)
        return tokens, positions

    def _prepare_model_inputs(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: GrammarOutput | None,
    ) -> TTModelInput:
        """Build a ``TTModelInput`` for one prefill or decode step.

        Reads the current persistent ``self.input_batch`` and assembles the
        padded, fixed-shape tensors a TT model needs (constant shapes are
        required for ttnn tracing). This is the front-packed input builder for
        single-process execution and standard multi-process DP; each rank
        operates on its whole local
        ``input_batch``. Single-process lane mode does not use this builder --
        it builds its merged device input directly from ``TTLaneInputBatch``.

        Args:
            scheduler_output: Scheduler decisions for this step. Used to detect
                prefill vs. decode and (for grammar) which requests carry
                structured-output bitmasks.
            grammar_output: Structured-output bitmasks for this step, or
                ``None`` when no request uses guided decoding.

        Returns:
            A ``TTModelInput`` with tokens, positions, block tables, sampling
            params and host-only metadata, padded to the appropriate wire
            capacity. ``unpadded_batch_size`` records the real (pre-pad)
            request count.
        """
        assert scheduler_output.total_num_scheduled_tokens > 0
        input_batch = self.input_batch
        batch_num_reqs = input_batch.num_reqs
        assert batch_num_reqs > 0

        # The whole local batch.
        req_indices = list(range(batch_num_reqs))
        num_reqs = len(req_indices)
        row_req_ids = [input_batch.req_ids[i] for i in req_indices]
        if self._num_speculative_tokens:
            # Pruned here rather than where requests finish, because every
            # other way a request leaves the persistent batch is temporary and
            # has to keep its state. ``self.requests`` is what says a request
            # is gone for good: ``_update_states`` drops it there only on
            # finish. A preempted request keeps its entry and loses it below,
            # where the prefill it resumes with resets the count.
            for req_id in self._req_accepted_counts.keys() - self.requests.keys():
                self._req_accepted_counts.pop(req_id, None)

        # Pad decode to the per-rank wire capacity, which outside lane mode is
        # the whole engine capacity.
        decode_pad_to = self.tt_per_lane_max_num_seqs

        # Models that declare ``tt_supported_decode_batch_sizes`` (e.g. Gemma4)
        # may pad only to the nearest supported size >= num_reqs, so B=1 is not
        # forced through a B=max graph.
        #
        # A model must declare only the buckets it has actually captured a decode
        # trace for on this instance: padding to a bucket with no captured trace
        # leaves the device in an undefined state. There is deliberately no
        # separate "warmed" list to fall back from -- a bucket that is not warmed
        # is not supported.
        decode_buckets = getattr(
            getattr(self, "model", None), "tt_supported_decode_batch_sizes", None
        )
        if decode_buckets:
            decode_pad_to = next(
                (
                    int(b)
                    for b in sorted(int(x) for x in decode_buckets)
                    if int(b) >= num_reqs and int(b) <= self.tt_per_lane_max_num_seqs
                ),
                self.tt_per_lane_max_num_seqs,
            )

        # Second dim of each block table is (ceil(max_model_len / block_size)).
        # Slice/pad to ``self.max_num_blocks_per_req``: slicing handles
        # over-wide tables when the total KV-cache limit is tighter than
        # ``ceil(max_model_len / block_size)``, and padding handles
        # under-wide ones from hybrid kv-cache groups whose native
        # block-table widths differ after upstream page-size unification.
        target_width = self.max_num_blocks_per_req
        block_tables_per_group = input_batch.block_tables_for_rows(
            req_indices, target_width
        )

        # Group-0 view kept on TTModelInput.block_tables for back-compat with
        # the single-tensor consumers (decode_forward page_table arg). Hybrid
        # models additionally consume ``block_tables_per_group`` via the
        # ``page_tables_per_group`` kwarg in submit_prefill / submit_decode; the
        # legacy generator_vllm wrappers strip it on the way through and raise
        # loudly if the list has more than one entry.
        block_tables = block_tables_per_group[0]

        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        num_scheduled = scheduler_output.num_scheduled_tokens

        def _is_still_prefilling(req_id: str) -> bool:
            # Any uncomputed PROMPT token is prefill work, including a final
            # chunk of exactly one token: prompt work must dispatch through
            # the prefill path, which owns chunk bookkeeping (e.g. the
            # intermediate-prefill mask and vision rope state). Past the
            # prompt, a steady-state step legitimately has exactly one output
            # width outstanding (the sampled token for AR, one whole canvas
            # for block models); only MORE than that means uncomputed history
            # (preemption-resume replay). A plain ``computed < total``
            # comparison dispatched every post-first block step as prompt
            # work, re-encoding the entire session per canvas: quadratic
            # prefill and decode never ran.
            row = input_batch.req_id_to_index[req_id]
            # ``num_computed_tokens`` is the scheduler's pre-step snapshot.
            num_computed = input_batch.num_computed_tokens_cpu[row]
            return (
                num_computed < input_batch.num_prompt_tokens[row]
                or num_computed + self._output_tokens_per_step
                < input_batch.num_tokens[row]
            )

        # A "prefill" step can contain:
        # - brand new requests (scheduled_new_reqs), and/or
        # - resumed-from-preemption requests (scheduled_cached_reqs with
        #   resumed_req_ids set) that need to replay tokens to rebuild KV,
        #   and/or
        # - chunked-prefill continuations: cached requests that have not
        #   computed all their prompt tokens yet.
        has_chunked_continuation = any(
            _is_still_prefilling(req_id)
            for req_id in cached_reqs.req_ids
            if req_id not in cached_reqs.resumed_req_ids
        )
        is_prompt = (
            len(scheduler_output.scheduled_new_reqs) > 0
            or bool(cached_reqs.resumed_req_ids)
            or has_chunked_continuation
        )
        sample_params = input_batch.sampling
        intermediate_prefill_mask: torch.Tensor | None = None
        num_valid_drafts: torch.Tensor | None = None
        accepted_counts: torch.Tensor | None = None
        spec_drafts: torch.Tensor | None = None
        if is_prompt:
            # num_computed_tokens for each request is the input position
            # (=computed previously and cached)
            input_positions = input_batch.num_computed_tokens_cpu[req_indices]
            # The generator slices ``tokens[start_pos:prompt_lens]``, so
            # ``prompt_lens`` is the end position of the chunk scheduled this
            # step, not the sequence length:
            # - Full prefill: start_pos=0, chunk=prompt_len => prompt_len.
            # - APC hit: start_pos=cached, chunk=prompt_len-cached => prompt_len.
            # - Resumed from preemption: the chunk spans the generated output
            #   tokens too, so the full sequence is replayed to rebuild KV.
            # - Chunked continuation: start_pos=computed, chunk=budget =>
            #   computed + budget, i.e. this chunk's end.
            chunk_lens = np.array(
                [num_scheduled[input_batch.req_ids[i]] for i in req_indices],
                dtype=np.int64,
            )
            prompt_lens = input_positions + chunk_lens
            intermediate_prefill_mask = torch.from_numpy(
                prompt_lens < input_batch.num_tokens[req_indices]
            )
            max_prefill_tokens = int(prompt_lens.max())
            input_tokens = input_batch.token_ids_cpu_tensor[
                req_indices, :max_prefill_tokens
            ]
            decode_layout_changed = False
            if self._num_speculative_tokens:
                # A prefill commits one token, so its accepted count is 1,
                # and a request resumed from preemption has no candidate state
                # left to select anyway. Dropping the entry restores that
                # default, so the first speculative step after a prefill needs
                # no special case. The drafts a prefilling row may still carry
                # are the scheduler's to drop, and it does: update_draft_token_ids
                # clears spec_token_ids for a request in a prefill chunk.
                for req_id in row_req_ids:
                    self._req_accepted_counts.pop(req_id, None)
        else:
            positions_np = input_batch.num_tokens[req_indices] - 1
            input_positions = torch.from_numpy(positions_np)
            input_tokens = input_batch.token_ids_cpu_tensor[
                req_indices, positions_np
            ].view(-1, 1)
            prompt_lens = None
            # Record whether the padded decode layout changed since the previous
            # decode. The controller translates this lifecycle fact into either
            # explicit contract commands or the legacy ``reset_batch`` keyword.
            decode_layout_changed = self._decode_layout_changed_since_last_decode

            if self._num_speculative_tokens:
                spec_drafts, num_valid_drafts, accepted_counts = self._spec_row_state(
                    self._req_accepted_counts,
                    self._drafts_to_verify(scheduler_output, row_req_ids),
                    row_req_ids,
                    self._num_speculative_tokens,
                )
                if _step_verifies(
                    self._spec_supports_narrow_decode,
                    num_valid_drafts,
                    # A model declaring ``spec_stateless_verify`` keeps no
                    # per-candidate state, so a previous multi-token commit has
                    # nothing to resolve and does not force a verify.
                    torch.ones_like(accepted_counts)
                    if (getattr(type(self.model), "model_capabilities", None) or {}).get(
                        "spec_stateless_verify", False
                    )
                    else accepted_counts,
                    len(row_req_ids),
                ):
                    self._note_unspeculable_verify_rows(row_req_ids)
                    # Uniformly 1+K wide, so a model needs one verify shape
                    # rather than two.
                    input_tokens, input_positions = self._spec_candidate_block(
                        spec_drafts, num_valid_drafts, input_tokens, input_positions
                    )
                else:
                    # No row carries a draft and no row has a multi-token
                    # commit left to resolve, so this step has nothing to
                    # verify. It runs as the ordinary decode it is: no
                    # ``spec_mode``, no side tensors, the sampler's own tail,
                    # and eligible for asynchronous overlap, which a verify
                    # never is. That is what keeps batched baseline decoding
                    # overlapped inside a server with speculation configured.
                    # Only a model that declares ``supports_narrow_decode``
                    # reaches this: for any other, the step stays a wide verify
                    # whose rows all carry zero valid drafts, because that
                    # model implements one input shape.
                    spec_drafts = None
                    num_valid_drafts = None
                    accepted_counts = None

            # TODO: Remove once TT models can support arbitrary batch sizes.
            # Pad decode to the lane/rank wire capacity.
            if input_tokens.shape[0] < decode_pad_to:
                batch_pad = decode_pad_to - input_tokens.shape[0]
                # Width comes from the built block: 1 for a plain decode, 1+K
                # for a speculative one.
                input_tokens = torch.cat(
                    [
                        input_tokens,
                        torch.zeros(
                            batch_pad, input_tokens.shape[1], dtype=torch.int32
                        ),
                    ]
                )
                # Pad positions with -1 to indicate no position
                input_positions = torch.cat(
                    [
                        input_positions,
                        torch.full(
                            (batch_pad, *input_positions.shape[1:]),
                            -1,
                            dtype=input_positions.dtype,
                        ),
                    ]
                )
                if spec_drafts is not None:
                    # The accept walk reads the drafts against the verify's
                    # own row count, so they pad with the rows.
                    spec_drafts = torch.cat(
                        [
                            spec_drafts,
                            torch.full(
                                (batch_pad, spec_drafts.shape[1]),
                                PLACEHOLDER_TOKEN_ID,
                                dtype=torch.int32,
                            ),
                        ]
                    )
                if num_valid_drafts is not None:
                    # A padding row carries no draft and stands on its own
                    # input token, which is the count 1: ``accepted_counts`` is
                    # a count in [1, 1+K] and is never 0, on a padding row as
                    # much as on a real one.
                    num_valid_drafts = torch.cat(
                        [num_valid_drafts, torch.zeros(batch_pad, dtype=torch.int32)]
                    )
                    accepted_counts = torch.cat(
                        [accepted_counts, torch.ones(batch_pad, dtype=torch.int32)]
                    )
                # Pad each per-group block table to the same wire capacity so
                # the device sees a fixed shape regardless of how many users
                # are active. Keep ``block_tables`` aliased to the (now padded)
                # group-0 view, matching the alias set up where
                # ``block_tables_per_group`` is built.
                block_tables_per_group = [
                    torch.cat(
                        [bt, torch.zeros(batch_pad, bt.shape[1], dtype=bt.dtype)],
                        dim=0,
                    )
                    for bt in block_tables_per_group
                ]
                block_tables = block_tables_per_group[0]
                # Sampling parameters are intentionally NOT padded here. The
                # per-row wire tensors are built below by
                # ``_sampling_params_for_padded_decode``, which takes this
                # build's ``req_indices`` rows and right-pads with fresh
                # defaults. The persistent ``input_batch.sampling`` tail is
                # never read, so there is nothing to default in place.

        row_req_ids = [input_batch.req_ids[i] for i in req_indices]

        tt_sampling_params = slice_tt_sampling_params(sample_params, req_indices)
        if not is_prompt and input_tokens.shape[0] > len(req_indices):
            # Decode inputs are padded to the rank batch size; pad the sampling
            # params to match, right-filling padding rows with neutral defaults.
            tt_sampling_params = self._sampling_params_for_padded_decode(
                sample_params, req_indices, input_tokens.shape[0]
            )

        if self.model_config.is_multimodal_model and is_prompt:
            multi_modal_kwargs = self._gather_multi_modal_inputs(
                req_indices=req_indices
            )
        else:
            multi_modal_kwargs = {}

        # If we're not using structured outputs, grammar_bitmask is None.
        bitmask = grammar_output.grammar_bitmask if grammar_output is not None else None
        has_structured = has_structured_outputs(
            self.requests, scheduler_output, bitmask
        )
        scheduled_structured_req_ids = scheduled_structured_output_request_ids(
            self.requests,
            scheduler_output,
        )
        has_scheduled_structured = bool(scheduled_structured_req_ids)
        if is_prompt:
            structured_output_req_ids = frozenset(
                req_id
                for row, req_id in enumerate(row_req_ids)
                if req_id in scheduled_structured_req_ids
                and (
                    intermediate_prefill_mask is None
                    or not bool(intermediate_prefill_mask[row])
                )
            )
        else:
            structured_output_req_ids = capture_structured_decode_request_ids(
                scheduled_structured_req_ids,
                row_req_ids,
            )
        if bitmask is not None:
            # Using torch tensor instead of numpy array for consistency
            # because we need it as tensor for gather.
            bitmask = torch.from_numpy(bitmask)
            # unpadded for prefill, padded for decode
            batch_length = input_tokens.shape[0]
            # `structured_output_request_ids` comes from GrammarOutput as a list
            # of request IDs (bitmask rows are in this order). TT does not support
            # speculative decoding in this path, so we assume a single bitmask row
            # per request.
            structured_output_request_ids = (
                grammar_output.structured_output_request_ids
                if grammar_output is not None
                else []
            )
            bitmask = reorder_grammar_bitmask_for_tt_batch(
                bitmask=bitmask,
                structured_output_request_ids=structured_output_request_ids,
                row_req_ids=row_req_ids,
                batch_length=batch_length,
                expected_structured_output_request_ids=structured_output_req_ids,
            )

        perform_device_sampling = self.check_perform_device_sampling(
            is_decode=not is_prompt,
            has_structured_outputs=has_structured,
            sampling_rows=req_indices,
        )
        if has_structured and not has_scheduled_structured:
            perform_device_sampling = False
        if intermediate_prefill_mask is not None and intermediate_prefill_mask.any():
            # Device sampling advances device RNG state for every row it reads,
            # which an intermediate chunk must not do. Host sampling can hand
            # those rows a generator clone instead.
            perform_device_sampling = False
        defer_device_sampling = (
            perform_device_sampling and has_scheduled_structured and not is_prompt
        )

        # Populate prompt_tokens and output_tokens if penalties are needed
        # (decode only).
        prompt_tokens = None
        output_tokens = None
        if (not input_batch.no_penalties) and not is_prompt:
            # Restrict to this build's requests. For lane builds ``req_indices``
            # selects one lane's rows out of the merged batch; passing them
            # keeps penalty history attributed to the right requests instead of
            # the merged batch's leading rows. Non-lane builds pass
            # ``range(num_reqs)``, so this is a no-op there.
            prompt_tokens = input_batch.make_prompt_token_ids_tensor(req_indices)
            output_tokens = input_batch.make_output_token_ids_tensor(req_indices)

            # Pad to the persistent batch capacity. A lane build carries only
            # its own lane's rows, so padding to the global capacity there would
            # over-send.
            if (
                self.tt_data_parallel_size == 1
                and prompt_tokens.shape[0] < input_batch.max_num_reqs
            ):
                batch_pad = input_batch.max_num_reqs - prompt_tokens.shape[0]
                prompt_tokens = torch.cat(
                    [
                        prompt_tokens,
                        torch.full(
                            (batch_pad, prompt_tokens.shape[1]), -1, dtype=torch.int32
                        ),
                    ]
                )
                output_tokens = torch.cat(
                    [
                        output_tokens,
                        torch.full(
                            (batch_pad, output_tokens.shape[1]), -1, dtype=torch.int32
                        ),
                    ]
                )

        # Build host-only sampling params from input_batch. The host sampler
        # interprets every per-request key/index as a row in the batch it is
        # handed, so each of these must be reindexed to this build's rows
        # (lane-local 0..num_reqs-1). ``req_indices`` is ``range(num_reqs)`` for
        # non-lane builds, so the remaps below are the identity there and only
        # do real work for lane builds.
        allowed_token_ids_mask = None
        if (
            not input_batch.no_allowed_token_ids
            and input_batch.sampling.allowed_token_ids_mask is not None
        ):
            # Gather already reindexes to lane-local rows.
            allowed_token_ids_mask = input_batch.sampling.allowed_token_ids_mask[
                req_indices
            ].clone()

        # Re-key the bad-words dict (req_index -> token id lists) to lane-local
        # rows. ``condense`` keeps the source keys aligned with active rows.
        src_bad_words = input_batch.sampling.bad_words_token_ids
        bad_words_token_ids = {
            local: src_bad_words[g]
            for local, g in enumerate(req_indices)
            if g in src_bad_words
        }

        # Builtin/custom logits processors hold per-row state over the whole
        # local batch, so the host sampler reuses them as-is.
        logitsprocs = input_batch.sampling.logitsprocs

        generators: dict[int, torch.Generator] = {}
        if not perform_device_sampling:
            generators = self._build_host_generators(
                input_batch, req_indices, intermediate_prefill_mask
            )
            # NOTE: Our sampling paths are different between host and device.
            # Whether a request is sampled on device or host
            # depends also on other requests in the batch.
            # This means sampling is not perfectly deterministic
            # whenever device sampling is enabled.

        block_tables_per_group = [bt.contiguous() for bt in block_tables_per_group]
        block_tables = block_tables_per_group[0]
        # State follows the request, not the row (``self._req_state_slot``), which
        # subsumes the batch's condense-move remap.
        input_batch.reset_slot_remap()
        prefill_empty_slots = None
        slot_remap = None
        if is_prompt:
            prefill_empty_slots = self._alloc_prefill_state_slots(row_req_ids)
        else:
            # Advances the ownership map to the post-gather layout, so the returned
            # remap has to reach the device: dropping it would leave the map claiming
            # a move that never happened.
            slot_remap = self._decode_state_slot_remap(row_req_ids)

        return TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=prompt_lens,
            block_tables=block_tables,
            block_tables_per_group=block_tables_per_group,
            block_tables_per_layer=self._block_tables_per_layer(block_tables_per_group),
            unpadded_batch_size=num_reqs,
            row_req_ids=row_req_ids,
            tt_sampling_params=tt_sampling_params,
            multi_modal_kwargs=multi_modal_kwargs,
            perform_device_sampling=perform_device_sampling,
            grammar_bitmask=[bitmask],  # wrap to match DP case
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            decode_layout_changed=decode_layout_changed,
            slot_remap=slot_remap,
            # Host-only sampling params - wrapped in lists for DP compatibility
            allowed_token_ids_mask_list=[allowed_token_ids_mask],
            bad_words_token_ids_list=[bad_words_token_ids],
            max_num_logprobs=[input_batch.max_num_logprobs],
            logitsprocs_list=[logitsprocs],
            generators_list=[generators],
            # Destination state slot per row. Without it a stateful model falls
            # back to ``range(N)`` and a prefill overwrites a decoding request's
            # state. Stateless models ignore it.
            prefill_empty_slots=prefill_empty_slots,
            intermediate_prefill_mask=intermediate_prefill_mask,
            defer_device_sampling=defer_device_sampling,
            structured_output_req_ids=structured_output_req_ids,
            grammar_row_req_ids=tuple(row_req_ids)
            + (None,) * (input_tokens.shape[0] - len(row_req_ids)),
            # Already in post-remap row order: ``req_indices`` reads the
            # persistent batch after ``condense`` moved every surviving
            # request's speculative state down with it.
            num_valid_drafts=num_valid_drafts,
            accepted_counts=accepted_counts,
            draft_token_ids=spec_drafts,
            spec_mode=ACCEPT_MODE_ARGMAX_IDS if spec_drafts is not None else None,
        )

    def build_model_input(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: GrammarOutput | None,
    ) -> TTModelInput | None:
        """
        Update internal state with the scheduler output and build
        TTModelInput without executing the model.
        Returns None if there is no scheduled work in this step.

        For data parallel, this function is called by each DP rank to build
        TTModelInput from it's own scheduler output.
        """
        # Update scheduler-owned state before accepting any older async result.
        # Reject late results for request IDs already reported finished. The
        # result that finished the request was accepted earlier. Ordinary
        # preemption or resume keeps a completed token valid for replay. Only a
        # forced prefix-cache reset marks an in-flight result for discard.
        self._update_states(scheduler_output)
        if (
            self.async_decode.decode_input_update_contract_version() < 1
            and self._decode_layout_changed_since_last_decode
        ):
            # Preserve the legacy path's drain point: version-0 adapters still
            # own reload policy and receive reset_batch after batch mutation.
            self.async_decode.wait_for_all_pending_async_steps()
        self.async_decode.apply_completed_decode_steps_before_build(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            reservation = _history_reservation.context()
            if reservation is not None:
                reservation.prepare_runner(self, scheduler_output, None)
            return None

        # Prepare model inputs only
        model_input = self._prepare_model_inputs(scheduler_output, grammar_output)
        reservation = _history_reservation.context()
        if reservation is not None:
            reservation.prepare_runner(self, scheduler_output, model_input)
        return model_input

    # All lane-specific input/output shaping lives in ``TTLaneInputBatch``
    # (``apply_step_plan`` / ``build_model_input`` / ``extract_output``) and the
    # merge/redistribute in ``TTLaneCoordinator``; this orchestration only wires
    # the device submission, async decode, and deferred state application -- the
    # genuinely runner/model-owned parts.

    @torch.no_grad()
    def _execute_lane_step(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput | None:
        """Run the device forward for one merged lane-DP step.

        Invoked by ``execute_model`` when the lane scheduler attached a step
        plan. Returns ``None`` after enqueuing the lane sampler (the grammar
        bitmask is applied later in ``sample_tokens``); returns
        ``EMPTY_MODEL_RUNNER_OUTPUT`` only when there is nothing scheduled.
        """
        plan = get_tt_step_plan(scheduler_output)
        assert plan is not None, "lane step requires a scheduler-attached plan"

        # Lane-DP lays requests out at sparse stable slots. Request-specific
        # RoPE (mrope/vision models) instead assumes front-packed request rows,
        # so the two are incompatible until the RoPE delta mapping is made
        # slot-aware. No lane-DP model needs it.
        if self.request_specific_rope:
            raise NotImplementedError(
                "lane-DP does not support request-specific RoPE "
                "(mrope/vision models) yet"
            )

        lane_batch = self.lane_batch
        steady_decode_candidate = (
            self.async_decode.can_attempt_steady_lane_decode_from_scheduler(
                scheduler_output,
                decode_layout_changed=plan.decode_layout_changed,
            )
        )
        if self.async_decode.must_drain_pending_async_steps(
            steady_decode_candidate, scheduler_output
        ):
            self.async_decode.wait_for_all_pending_async_steps()

        layout_changed = lane_batch.apply_step_plan(
            scheduler_output, plan, self.requests, self.encoder_cache
        )
        assert layout_changed == plan.decode_layout_changed, (
            "lane scheduler and runner disagreed about decode layout change: "
            f"plan={plan.decode_layout_changed}, applied={layout_changed}"
        )
        if layout_changed:
            self._decode_layout_changed_since_last_decode = True
            if self.async_decode.decode_input_update_contract_version() < 1:
                self.async_decode.wait_for_all_pending_async_steps()
        self.async_decode.apply_completed_decode_steps_before_build(scheduler_output)

        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT
        scheduled_rows = list(plan.scheduled_rows)
        if not scheduled_rows:
            # Tokens were scheduled but no slot mapped to them; the engine still
            # calls sample_tokens, so enqueue an empty result to keep the
            # forward/sample calls balanced.
            self._pending_samples.append(lambda _grammar: EMPTY_MODEL_RUNNER_OUTPUT)
            return None

        # Grammar is applied at sample time, so the forward builds without it.
        model_input = lane_batch.build_model_input(self, scheduler_output, None, plan)
        if plan.is_decode:
            # ``slot_grammar_bitmask`` reorders against the full decode capacity.
            lane_total = plan.capacity
            if model_input.defer_device_sampling:
                self.async_decode.wait_for_all_pending_async_steps()
            if self.async_decode_scheduling and not model_input.defer_device_sampling:
                req_ids = [
                    model_input.grammar_row_req_ids[row] for row in scheduled_rows
                ]
                if any(req_id is None for req_id in req_ids):
                    raise RuntimeError(
                        "submitted lane row identity is missing scheduled request IDs: "
                        f"scheduled_rows={scheduled_rows}, row_req_ids={req_ids}"
                    )
                context = self.async_decode.capture_submitted_step_context(req_ids)
                wrapper = self.async_decode.submit_async_lane_decode(
                    model_input, context, scheduled_rows
                )
                self._pending_samples.append(
                    partial(
                        self._finish_async_decode,
                        wrapper=wrapper,
                        model_input=model_input,
                        lane_total=lane_total,
                    )
                )
                return None
            submission = self.async_decode.submit_decode(
                model_input,
                read_from_device=not model_input.defer_device_sampling,
                async_read=False,
                sampling_rows=scheduled_rows,
            )
            if model_input.defer_device_sampling:
                tt_out = submission.tt_out
                tt_log_probs = None
            else:
                finalized = self.async_decode.finalize_decode(
                    submission, sampling_rows=scheduled_rows
                )
                assert finalized is not None
                tt_out = finalized.tt_out
                tt_log_probs = finalized.tt_log_probs
            is_decode = True
        else:
            submission = None
            # Prefill reorders against the per-lane request capacity.
            lane_total = lane_batch.max_num_reqs
            tt_out = self.submit_prefill(model_input, model_input.unpadded_batch_size)
            tt_log_probs = None
            assert isinstance(
                model_input.tt_sampling_params.enable_log_probs, torch.Tensor
            )
            if (
                model_input.perform_device_sampling
                and model_input.tt_sampling_params.enable_log_probs.any()
            ):
                assert isinstance(tt_out, tuple) and len(tt_out) == 2
                tt_out, tt_log_probs = tt_out
            elif isinstance(tt_out, tuple):
                tt_out, _ = tt_out
            is_decode = False

        # Forward done; defer the lane sampling tail to ``sample_tokens``. The
        # step runs synchronously, so the lane batch layout is stable between
        # this enqueue and the deferred sample.
        self._pending_samples.append(
            partial(
                self._finish_lane_sync,
                tt_out=tt_out,
                tt_log_probs=tt_log_probs,
                model_input=model_input,
                scheduled_rows=scheduled_rows,
                is_decode=is_decode,
                lane_total=lane_total,
                decode_submission=submission,
            )
        )
        return None

    def _finish_lane_sync(
        self,
        grammar_output: GrammarOutput | None,
        *,
        tt_out: Any,
        tt_log_probs: Any,
        model_input: TTModelInput,
        scheduled_rows: list[int],
        is_decode: bool,
        lane_total: int,
        decode_submission: TTDecodeSubmission | None = None,
    ) -> ModelRunnerOutput:
        deferred = bool(getattr(model_input, "defer_device_sampling", False))
        try:
            return TTModelRunner._finish_lane_sync_impl(
                self,
                grammar_output,
                tt_out=tt_out,
                tt_log_probs=tt_log_probs,
                model_input=model_input,
                scheduled_rows=scheduled_rows,
                is_decode=is_decode,
                lane_total=lane_total,
                decode_submission=decode_submission,
            )
        except Exception:
            if deferred:
                self._poison_device_grammar()
            raise

    def _finish_lane_sync_impl(
        self,
        grammar_output: GrammarOutput | None,
        *,
        tt_out: Any,
        tt_log_probs: Any,
        model_input: TTModelInput,
        scheduled_rows: list[int],
        is_decode: bool,
        lane_total: int,
        decode_submission: TTDecodeSubmission | None = None,
    ) -> ModelRunnerOutput:
        """Sample and build the runner output for a deferred lane forward."""
        model_input = self._apply_grammar_to_input(
            model_input, grammar_output, lane_total=lane_total
        )
        if getattr(model_input, "defer_device_sampling", False):
            if decode_submission is None:
                raise RuntimeError(
                    "deferred lane decode is missing its device submission"
                )
            finalized = self.async_decode.complete_deferred_device_sampling(
                decode_submission,
                model_input,
            )
            tt_out = finalized.tt_out
            tt_log_probs = finalized.tt_log_probs
            model_input = replace(
                model_input,
                grammar_bitmask=[None],
                defer_device_sampling=False,
            )
        sampled, logprobs = self.lane_batch.extract_output(
            self, tt_out, tt_log_probs, model_input, scheduled_rows, is_decode=is_decode
        )
        # ``scheduled_rows`` are persistent slots; their req_ids in row order are
        # the canonical merged output order.
        submitted_row_ids = getattr(model_input, "grammar_row_req_ids", ())
        snapshot_covers_rows = bool(submitted_row_ids) and (
            not scheduled_rows or max(scheduled_rows) < len(submitted_row_ids)
        )
        if isinstance(model_input, TTModelInput) and not snapshot_covers_rows:
            raise RuntimeError(
                "submitted lane row identity does not cover scheduled rows: "
                f"scheduled_rows={scheduled_rows}, "
                f"row_identity_size={len(submitted_row_ids)}"
            )
        req_ids = (
            [submitted_row_ids[row] for row in scheduled_rows]
            if snapshot_covers_rows
            else [self.lane_batch.req_ids[row] for row in scheduled_rows]
        )
        if any(req_id is None for req_id in req_ids):
            raise RuntimeError(
                "submitted lane row identity is missing scheduled request IDs: "
                f"scheduled_rows={scheduled_rows}, row_req_ids={req_ids}"
            )

        if not is_decode and model_input.prompt_lens is not None:
            lane_batch = self.lane_batch
            prompt_lens = np.asarray(model_input.prompt_lens)
            num_tokens = np.array(
                [lane_batch.num_tokens[row] for row in scheduled_rows],
                dtype=np.int64,
            )
            intermediate_mask = prompt_lens < num_tokens
            if intermediate_mask.any():
                return self._build_chunked_prefill_output(
                    req_ids=req_ids,
                    sampled_token_ids=sampled,
                    logprobs=logprobs,
                    intermediate_mask=intermediate_mask,
                    defer_state_apply=TTModelRunner._uses_async_scheduler(self),
                )

        if TTModelRunner._uses_async_scheduler(self):
            return self.defer_state_apply_and_build_runner_output(
                sampled, logprobs, req_ids=req_ids
            )
        return self.apply_and_build_runner_output(sampled, logprobs, req_ids=req_ids)

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput | None:
        """Run this rank's front-packed or lane-DP device forward.

        Returns ``None`` after enqueuing a pending sampler; the engine computes
        the grammar bitmask while the forward runs and then calls
        ``sample_tokens``. Returns ``EMPTY_MODEL_RUNNER_OUTPUT`` only when
        nothing was scheduled (no sampler is enqueued). Standard multi-process
        DP reaches this entrypoint once per rank, each over its own mesh.
        """
        TTModelRunner._raise_if_device_grammar_poisoned(self)

        # Single-process lane-DP: the lane scheduler attaches a per-step plan to
        # the scheduler output. When present, the step runs over the merged lane
        # batch (``TTLaneInputBatch`` owns all the lane-specific input/output
        # shaping); the generic path below stays lane-agnostic.
        if get_tt_step_plan(scheduler_output) is not None:
            if _history_reservation.context() is not None:
                raise RuntimeError("Lane plan unsupported by history reservation")
            return self._execute_lane_step(scheduler_output)

        # Decide whether the next build can remain one step host-stale before
        # mutating the persistent batch. On a transition, finalize pending work
        # now but defer applying it until ``build_model_input`` has processed
        # lifecycle events and the forced-reset discard boundary.
        steady_decode_candidate = (
            self.async_decode.can_attempt_steady_decode_from_scheduler(scheduler_output)
        )
        if self.async_decode.must_drain_pending_async_steps(
            steady_decode_candidate, scheduler_output
        ):
            self.async_decode.wait_for_all_pending_async_steps()

        # Grammar is applied at sample time, so the forward builds without it.
        model_input = self.build_model_input(scheduler_output, None)
        if model_input is None:
            return EMPTY_MODEL_RUNNER_OUTPUT

        is_decode = model_input.prompt_lens is None
        if model_input.defer_device_sampling:
            self.async_decode.wait_for_all_pending_async_steps()
        if (
            self.async_decode_scheduling
            and is_decode
            and not model_input.defer_device_sampling
        ):
            steady_decode_fast_path = self.async_decode.can_use_steady_decode_fast_path(
                model_input
            )
            wrapper = self.async_decode.submit_async_decode(
                model_input,
                steady_decode_fast_path=steady_decode_fast_path,
            )
            self._pending_samples.append(
                partial(
                    self._finish_async_decode,
                    wrapper=wrapper,
                    model_input=model_input,
                    lane_total=None,
                )
            )
            return None

        # Synchronous path (prefill, or decode without async scheduling): run
        # the forward now and defer sampling to ``sample_tokens``.
        fwd = self._forward_with_model_input(model_input)
        self._pending_samples.append(partial(self._finish_front_packed_sync, fwd=fwd))
        return None

    def _reorder_grammar_bitmask(
        self,
        grammar_output: GrammarOutput | None,
        model_input: TTModelInput,
        *,
        lane_total: int | None,
        require_complete: bool = True,
    ) -> torch.Tensor | None:
        """Reorder a scheduler grammar bitmask into TT batch layout, or ``None``."""
        if grammar_output is None or grammar_output.grammar_bitmask is None:
            if require_complete and model_input.structured_output_req_ids:
                raise RuntimeError(
                    "sample-time grammar output is absent for TT batch request IDs: "
                    f"{sorted(model_input.structured_output_req_ids)}"
                )
            return None

        bitmask = torch.from_numpy(grammar_output.grammar_bitmask)
        row_req_ids = model_input.grammar_row_req_ids
        if not row_req_ids:
            raise RuntimeError(
                "sample-time grammar remapping is missing the submitted TT row identity"
            )
        return reorder_grammar_bitmask_for_tt_batch(
            bitmask=bitmask,
            structured_output_request_ids=grammar_output.structured_output_request_ids,
            row_req_ids=row_req_ids,
            batch_length=len(row_req_ids),
            expected_structured_output_request_ids=(
                model_input.structured_output_req_ids if require_complete else None
            ),
        )

    def _apply_grammar_to_input(
        self,
        model_input: TTModelInput,
        grammar_output: GrammarOutput | None,
        *,
        lane_total: int | None,
    ) -> TTModelInput:
        """Return ``model_input`` with the sample-time grammar bitmask attached."""
        bitmask = self._reorder_grammar_bitmask(
            grammar_output, model_input, lane_total=lane_total
        )
        if bitmask is None:
            return model_input
        return replace(model_input, grammar_bitmask=[bitmask])

    def _finish_async_decode(
        self,
        grammar_output: GrammarOutput | None,
        *,
        wrapper: AsyncTTModelRunnerOutput,
        model_input: TTModelInput,
        lane_total: int | None,
    ) -> AsyncTTModelRunnerOutput:
        """Attach the sample-time grammar bitmask to a deferred async decode.

        The reorder happens here on the engine thread (layout still matches the
        forward); the wrapper applies the bitmask when its read completes. A
        row finished by the preceding queued result may be absent from the new
        grammar output; it stays unrestricted here and late-output suppression
        discards its sampled token.
        """
        bitmask = self._reorder_grammar_bitmask(
            grammar_output,
            model_input,
            lane_total=lane_total,
            require_complete=False,
        )
        if bitmask is not None:
            wrapper.set_grammar_bitmask(bitmask)
        return wrapper

    def _finish_spec_decode(self, fwd: _SyncForward) -> ModelRunnerOutput:
        """Accept, commit and re-propose for one speculative decode step.

        The synchronous tail, where the forward has already completed: walk,
        commit and publish in one go. The asynchronous path splits the same
        three at the readback boundary, which is why the walk and the commit
        are separate methods rather than one.
        """
        committed, counts = self.walk_spec_acceptance(fwd.model_input, fwd.tt_out)
        prefixes = self.spec_committed_prefixes(fwd.model_input, committed, counts)
        self.commit_spec_acceptance(fwd.model_input, committed, counts, fwd.spec_hidden)
        return self.build_spec_runner_output(fwd.model_input.row_req_ids, prefixes)

    def walk_spec_acceptance(
        self, model_input: TTModelInput, argmax_ids: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy acceptance over one verify's return. Touches no host state.

        Separate from the commit because the asynchronous path runs this on
        whichever thread resolves the readback, where the runner's own state is
        one step behind and must not be read or written. Everything this needs
        is the step's own: the block it submitted and what came back.
        """
        row_req_ids = model_input.row_req_ids
        missing = [
            name
            for name in ("row_req_ids", "draft_token_ids", "num_valid_drafts")
            if getattr(model_input, name) is None
        ]
        if missing:
            raise RuntimeError(
                f"a speculative step in mode {model_input.spec_mode!r} reached "
                f"the accept walk without {missing}; the builder sets all three "
                "together whenever it sets spec_mode"
            )
        del row_req_ids

        if not isinstance(argmax_ids, torch.Tensor):
            raise TypeError(
                "a verify in argmax_ids mode must return a token id tensor, got "
                f"{type(argmax_ids).__name__}"
            )
        # The draft block is built at the full width K even on a narrow step,
        # where the model was handed one column because no row carried a draft.
        # The walk compares the two, so the drafts are trimmed to the width the
        # verify actually answered at.
        verified_drafts = int(argmax_ids.shape[1]) - 1
        # Walked over every row the verify answered for, padding rows included,
        # because a model-owned drafter is asked for the same rows the verify
        # ran on: its per-row state is indexed by row, and a device graph has
        # one shape. A padding row carries no draft, so the walk commits its
        # column 0 and counts 1, and only the live rows reach a request.
        return accept_greedy_drafts(
            argmax_ids,
            model_input.draft_token_ids[:, :verified_drafts],
            model_input.num_valid_drafts,
        )

    def spec_committed_prefixes(
        self,
        model_input: TTModelInput,
        committed: torch.Tensor,
        counts: torch.Tensor,
    ) -> list[list[int]]:
        """Each live row's accepted prefix, as the step's published output.

        Read off the step's own tensors and the immutable model config, never
        off runner state, so this is safe on a readback thread.

        The length cap is applied here rather than left to the commit, so both
        paths publish the same thing. It can be: the block's own first column
        sits at the position the row had reached when the block was built, and
        a speculative step is drained before the next one is built, so that is
        still the row's length when the commit runs. Publishing more than fits
        would report tokens the commit then refuses to write, and the request's
        own history and the engine's would disagree about its text.
        """
        rows = len(model_input.row_req_ids)
        max_model_len = int(self.model_config.max_model_len)
        positions = model_input.input_positions
        first = positions if positions.dim() == 1 else positions[:, 0]
        block = committed[:rows].tolist()
        prefixes: list[list[int]] = []
        for row, count in enumerate(counts[:rows].tolist()):
            # One past the row's last committed token is where this step writes.
            start = int(first[row]) + 1
            room = max(0, max_model_len - start)
            keep = min(int(count), room)
            prefixes.append([int(token) for token in block[row][:keep]])
        return prefixes

    def commit_spec_acceptance(
        self,
        model_input: TTModelInput,
        committed: torch.Tensor,
        counts: torch.Tensor,
        spec_hidden: Any,
        *,
        skip_req_ids: set[str] | None = None,
    ) -> dict[str, list[int]]:
        """Apply the accepted prefixes and publish the next proposal.

        Mutates runner state, so it runs on the engine thread only: on the
        synchronous path inside the sampling tail, and on the asynchronous path
        when the next step drains this one. ``skip_req_ids`` is how a forced
        prefix-cache reset drops a frame it has already accounted for.
        """
        row_req_ids = model_input.row_req_ids
        rows = len(row_req_ids)
        committed_by_req = self._apply_committed_spec_tokens_to_state(
            row_req_ids,
            committed[:rows],
            counts[:rows],
            skip_req_ids=skip_req_ids,
        )
        if self._spec_drafts_from_model:
            self._propose_model_drafts(
                committed,
                counts,
                model_input,
                spec_hidden,
                row_req_ids,
                skip_req_ids=skip_req_ids,
            )
        else:
            self._propose_ngram_drafts(committed_by_req)
        return committed_by_req

    def propose_after_plain_step(
        self,
        model_input: TTModelInput,
        sampled_token_ids: torch.Tensor,
        row_req_ids: list[str],
        *,
        skip_req_ids: set[str] | None = None,
    ) -> None:
        """Publish the next proposal after a step that verified nothing.

        A speculating launch runs a step with nothing to verify as the ordinary
        decode it is, which is what lets it overlap. The proposal still has to
        happen: the drafts for the next step come from a proposer that is asked
        once per step, so a launch that skipped it on a draftless step would
        never draft again and would stay in plain decoding for the rest of the
        server's life, however the batch changed.

        The block is one column wide, that column being the token this step
        committed, and every count is 1. ``_propose_model_drafts`` pads the
        block to the uniform ``1+K`` a drafter is called with, so a drafter
        sees one call shape whatever produced the tokens.

        ``hidden`` is ``None``: a step that returned no ``VerifyOutput``
        produced no handle. ``load_model`` keeps a drafter that is fed its
        hidden state through the runner off this path entirely.
        """
        skipped = set(skip_req_ids or ())
        live = len(row_req_ids)
        sampled = _coerce_output_block(sampled_token_ids, live, 1).to(torch.int32)
        # The drafter's rows are the step's rows, padding included, because its
        # state is indexed by row and a device graph has one shape. A padding
        # row's token is never read: what marks it as owned by no request is
        # its position, which ``_committed_positions`` leaves negative.
        rows = int(model_input.input_positions.shape[0])
        committed = torch.zeros((rows, 1), dtype=torch.int32)
        committed[:live] = sampled
        counts = torch.ones(rows, dtype=torch.int32)
        if self._spec_drafts_from_model:
            self._propose_model_drafts(
                committed,
                counts,
                model_input,
                None,
                row_req_ids,
                skip_req_ids=skipped,
            )
            return
        sampled_np = sampled.numpy()
        self._propose_ngram_drafts(
            {
                req_id: [int(sampled_np[row, 0])]
                for row, req_id in enumerate(row_req_ids)
                if req_id not in skipped
            }
        )

    def build_spec_runner_output(
        self, row_req_ids: list[str], prefixes: list[list[int]]
    ) -> ModelRunnerOutput:
        """One speculative step's output, with its per-row committed prefixes.

        Not routed through ``_build_runner_output``: that coerces the sampled
        ids to one fixed width for every row, and a speculative step commits a
        width that differs per row.
        """
        return ModelRunnerOutput(
            req_ids=list(row_req_ids),
            req_id_to_index={req: idx for idx, req in enumerate(row_req_ids)},
            sampled_token_ids=[list(prefix) for prefix in prefixes],
            logprobs=None,
            prompt_logprobs_dict=dict.fromkeys(row_req_ids, None),
            pooler_output=[],
        )

    def _apply_committed_spec_tokens_to_state(
        self,
        row_req_ids: list[str],
        committed: torch.Tensor,
        counts: torch.Tensor,
        *,
        skip_req_ids: set[str] | None = None,
    ) -> dict[str, list[int]]:
        """Commit each row's accepted prefix and record its count.

        Separate from ``_apply_sampled_tokens_to_state`` rather than widening
        it. That function commits one fixed width for every row of every step
        of every model, twice over in two code paths, and a speculative step
        commits a length that differs per row. Sharing it would put a variable
        width through the one path that every non-speculating model depends on.

        Returns the committed ids per request, which the proposer needs and
        which become the step's output.
        """
        max_model_len = self.model_config.max_model_len
        committed_np = committed.numpy()
        skipped = set(skip_req_ids or ())
        committed_by_req: dict[str, list[int]] = {}
        for row, req_id in enumerate(row_req_ids):
            if req_id in skipped:
                # A forced prefix-cache reset already accounted for this frame.
                # Writing it would replay tokens the scheduler has discarded,
                # and proposing from it would draft a continuation of them.
                continue
            count = int(counts[row])
            block = [int(token) for token in committed_np[row, :count]]
            batch_row = self.input_batch.req_id_to_index.get(req_id)
            if batch_row is not None:
                start = int(self.input_batch.num_tokens[batch_row])
                end = min(start + len(block), max_model_len)
                block = block[: end - start]
                self.input_batch.token_ids_cpu[batch_row, start:end] = block
                self.input_batch.num_tokens[batch_row] = end
            # Exactly one extend: ``InputBatch.add_request`` stores the
            # request's own output list in ``req_output_token_ids``, so the two
            # are the same object while the request holds a row and extending
            # both would commit every token twice.
            req_state = self.requests.get(req_id)
            if batch_row is not None:
                output_token_ids = self.input_batch.req_output_token_ids[batch_row]
                if output_token_ids is None:
                    raise RuntimeError(
                        f"request {req_id} holds row {batch_row} of the "
                        "persistent batch but that row has no output token "
                        f"list, so its {len(block)} committed token(s) have "
                        "nowhere to go"
                    )
                output_token_ids.extend(block)
            elif req_state is not None:
                req_state.output_token_ids.extend(block)
            if batch_row is None and req_id not in self._req_state_slot:
                # Explicit preemption releases the request's model state slot.
                # The tokens remain valid for replay, but the accepted count
                # would select candidate state the model no longer holds.
                continue
            # The count the next verify reads, which is what selects the
            # candidate state this step left the model holding. Length-capped
            # rows record what they actually committed, never the full count.
            self._req_accepted_counts[req_id] = max(len(block), 1)
            committed_by_req[req_id] = block
        return committed_by_req

    def _finish_front_packed_sync(
        self, grammar_output: GrammarOutput | None, *, fwd: _SyncForward | None
    ) -> ModelRunnerOutput:
        deferred = bool(fwd is not None and fwd.model_input.defer_device_sampling)
        try:
            return TTModelRunner._finish_front_packed_sync_impl(
                self,
                grammar_output,
                fwd=fwd,
            )
        except Exception:
            if deferred:
                self._poison_device_grammar()
            raise

    def _finish_front_packed_sync_impl(
        self, grammar_output: GrammarOutput | None, *, fwd: _SyncForward | None
    ) -> ModelRunnerOutput:
        """Sample and build output for a deferred front-packed forward."""
        if fwd is None:
            return self.apply_and_build_runner_output(
                torch.tensor([], dtype=torch.int32), None
            )
        fwd = replace(
            fwd,
            model_input=self._apply_grammar_to_input(
                fwd.model_input, grammar_output, lane_total=None
            ),
        )
        if fwd.model_input.spec_mode is not None:
            return self._finish_spec_decode(fwd)

        sampled_token_ids_per_dp, logprobs_per_dp = self._sample_sync_forward(fwd)
        sampled_token_ids = sampled_token_ids_per_dp[0]
        logprobs_tensors = logprobs_per_dp[0] if logprobs_per_dp else None
        logprobs = logprobs_tensors.tolists() if logprobs_tensors else None

        row_req_ids = fwd.model_input.row_req_ids

        assert row_req_ids is not None

        is_prefill = not fwd.is_decode and fwd.model_input.prompt_lens is not None

        if is_prefill:
            intermediate_mask = fwd.model_input.intermediate_prefill_mask

            assert intermediate_mask is not None
            if intermediate_mask.any():
                return self._build_chunked_prefill_output(
                    req_ids=row_req_ids,
                    sampled_token_ids=sampled_token_ids,
                    logprobs=logprobs,
                    intermediate_mask=intermediate_mask.numpy(),
                    defer_state_apply=TTModelRunner._uses_async_scheduler(self),
                )

        if TTModelRunner._uses_async_scheduler(self):
            return self.defer_state_apply_and_build_runner_output(
                sampled_token_ids,
                logprobs,
                req_ids=(
                    row_req_ids
                    if is_prefill
                    else list(self.input_batch.req_ids[: self.input_batch.num_reqs])
                ),
            )
        output = self.apply_and_build_runner_output(
            sampled_token_ids,
            logprobs,
            # Only a prefill build can filter rows. Decode rows are the front-packed
            # batch rows, so it keeps the vectorized state write.
            req_ids=row_req_ids if is_prefill else None,
        )
        if not is_prefill and self._num_speculative_tokens:
            self.propose_after_plain_step(
                fwd.model_input, sampled_token_ids, row_req_ids
            )
        return output

    def _build_chunked_prefill_output(
        self,
        req_ids: list[str],
        sampled_token_ids: torch.Tensor,
        logprobs: LogprobsLists | None,
        intermediate_mask: np.ndarray,
        req_id_to_index: dict[str, int] | None = None,
        defer_state_apply: bool = False,
    ) -> ModelRunnerOutput:
        """Build a prefill output with intermediate rows suppressed.

        A request mid-prompt gets ``[]`` so the engine advances its computed
        tokens without appending output. Only final rows emit a token and reach
        runner state.
        """
        assert self._output_tokens_per_step == 1, (
            "Chunked-prefill output suppression assumes one sampled token per "
            "request; block-output models must disable chunked prefill"
        )
        emit_mask = np.asarray(
            [not bool(intermediate_mask[i]) for i in range(len(req_ids))],
            dtype=bool,
        )
        final_idx_np = np.where(emit_mask)[0]
        final_tokens = None
        final_req_ids: list[str] = []
        if final_idx_np.shape[0] > 0:
            final_idx_tensor = torch.from_numpy(final_idx_np.astype(np.int64))
            final_tokens = sampled_token_ids[final_idx_tensor]
            final_req_ids = [req_ids[int(i)] for i in final_idx_np]
            if not defer_state_apply:
                self._apply_sampled_tokens_to_state(final_tokens, req_ids=final_req_ids)

        num_reqs = len(req_ids)
        sampled_token_ids_np = sampled_token_ids.view(num_reqs).numpy()
        if sampled_token_ids_np.dtype != np.int32:
            sampled_token_ids_np = sampled_token_ids_np.astype(np.int32, copy=False)
        sampled_token_id_lists = [
            [int(sampled_token_ids_np[i])] if emit_mask[i] else []
            for i in range(num_reqs)
        ]

        runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=(
                dict(req_id_to_index)
                if req_id_to_index is not None
                else {req_id: idx for idx, req_id in enumerate(req_ids)}
            ),
            sampled_token_ids=sampled_token_id_lists,
            logprobs=logprobs,
            prompt_logprobs_dict=dict.fromkeys(req_ids, None),
            pooler_output=[],
        )
        if defer_state_apply and final_tokens is not None:
            self._enqueue_deferred_state_apply(
                final_tokens, final_req_ids, runner_output
            )
        return runner_output

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | AsyncTTModelRunnerOutput | None:
        """Sample the forward deferred by a preceding ``execute_model``.

        Pops the oldest pending forward (FIFO, matching the engine's
        oldest-first finalization), applies the now-ready grammar bitmask, and
        produces its output (or an async wrapper for overlapped decode). The
        engine calls this exactly once per ``execute_model`` that returned
        ``None``.

        If the preceding ``execute_model`` failed it enqueued no pending
        forward, so return ``None`` instead of raising ``IndexError`` on an
        empty deque. The engine treats ``None`` here as "the original
        execute_model failed" and re-raises that real exception (see
        ``EngineCore``'s ``if model_output is None`` path), so the true cause
        surfaces instead of being masked by the empty-popleft error.
        """
        TTModelRunner._raise_if_device_grammar_poisoned(self)
        if not self._pending_samples:
            return None
        finish = self._pending_samples.popleft()
        return finish(grammar_output)

    def check_perform_device_sampling(
        self,
        is_decode: bool,
        has_structured_outputs: bool,
        *,
        sampling_rows: list[int] | None = None,
    ) -> bool:
        want_device_sampling = self.sample_on_device_mode == "all" or (
            self.sample_on_device_mode == "decode_only" and is_decode
        )
        if not want_device_sampling:
            return False

        # Calculate number of devices per DP rank
        num_devices = self.num_devices // self.tt_data_parallel_size

        # Always host-only sampling params: min_p, bad_words, logit_bias,
        # allowed_token_ids, min_tokens require host sampling.
        input_batch = self.input_batch
        max_top_k = self.model.model_capabilities.get("max_device_top_k")
        if max_top_k is not None:
            sampling = input_batch.sampling
            # Prefill can submit only a subset of the resident lane requests.
            # An unscheduled decode request must not change its sampling route.
            # Steady decode omits the selection because it submits all occupants.
            rows = (
                list(input_batch.req_id_to_index.values())
                if sampling_rows is None
                else sampling_rows
            )
            temperature = sampling.temperature[rows]
            top_k = sampling.top_k[rows]
            needs_unbounded_or_larger_k = (temperature != 0) & (
                (top_k < 1) | (top_k > max_top_k)
            )
            if needs_unbounded_or_larger_k.any():
                return False
        if not input_batch.no_penalties and not self.model.model_capabilities.get(
            "supports_device_penalties", True
        ):
            return False
        has_always_host_only_sampling_params = (
            not input_batch.no_allowed_token_ids  # allowed_token_ids set
            or input_batch.sampling.bad_words_token_ids  # bad_words set
            or input_batch.sampling.has_active_logitsprocs()  # min_p, logit_bias,
            # min_tokens
            or bool(self.model_config.logits_processors)  # custom logitsprocs
        )
        if has_always_host_only_sampling_params:
            return False

        # The initial model contract covers decode only. Prefill keeps the
        # established host path so its first token remains grammar-safe.
        # Upstream async scheduling can queue another forward before the
        # current sample-time grammar arrives; the traced TT logits buffer is
        # single-owner, so structured decode stays on host until that overlap
        # has a dedicated multi-buffer contract.
        if has_structured_outputs and (
            not is_decode
            or not self.supports_device_grammar
            or self.scheduler_config.async_scheduling
        ):
            return False

        # Logprobs on device require multi-device setups (num_devices in {8,32}).
        # On single device, all logprobs require host sampling.
        # https://github.com/tenstorrent/tt-metal/issues/34077
        #
        # Top-K logprobs (max_lp > 0) are only supported on device by models
        # that set use_topk_logprobs=True (e.g. gpt-oss-120b), which return
        # top-32 logprobs as a (logprobs, indices) tuple. Other models only
        # return the sampled token's logprob, so max_lp > 0 falls back to
        # host sampling to compute full top-N from logits.
        max_lp = input_batch.max_num_logprobs
        if has_structured_outputs and max_lp is not None:
            # Grammar-aware device logprob parity is not part of the initial
            # contract. Keep vLLM's host path rather than publish incomplete
            # or force-argmax placeholder metadata.
            return False
        if max_lp is not None:
            if num_devices not in (8, 32):
                return False
            if max_lp > 0 and not self.supports_topk_logprobs:
                return False

        return True

    def submit_prefill(
        self,
        model_input: TTModelInput,
        batch_size_per_dp: list[int],
    ) -> Any:
        """Submit a prefill step and return the raw TT output.

        Launches TT prefill and returns the raw TT output used by the
        synchronous extraction path.
        """
        kwargs = {
            "tokens": model_input.input_tokens,
            "page_table": model_input.block_tables,
            "kv_cache": self.kv_caches,
            "enable_trace": self.trace_mode in ["all"],
            "prompt_lens": model_input.prompt_lens,
            "start_pos": model_input.input_positions,
        }
        # Hybrid attention models route per-layer block tables; the
        # runner already expanded ``block_tables_per_group`` into a
        # per-layer list at submission time when the kv_cache_config has
        # multiple groups. Legacy/uniform models leave it as ``None``
        # and never see the kwarg.
        if model_input.block_tables_per_layer is not None:
            kwargs["page_tables_per_layer"] = model_input.block_tables_per_layer
        kwargs.update(model_input.multi_modal_kwargs)
        if getattr(self.model, "supports_prefill_history_finality", False):
            mask = model_input.intermediate_prefill_mask
            if mask is None or mask.numel() != len(model_input.prompt_lens):
                raise ValueError("Final-prefill history release requires exact row finality")
            kwargs["retain_prefill_history"] = mask.tolist()
        if model_input.perform_device_sampling:
            sampling_params = model_input.tt_sampling_params
            sampling_param_dict = {
                field.name: (
                    getattr(sampling_params, field.name).tolist()
                    if getattr(sampling_params, field.name) is not None
                    else None
                )
                for field in fields(sampling_params)
            }
            sampling_param_dict["seed"] = [
                None if s == SEED_NONE_SENTINEL else s
                for s in sampling_param_dict["seed"]
            ]
            kwargs["sampling_params"] = TTSamplingParams(**sampling_param_dict)
        # The slots go to the model whenever the build supplied them: a stateful
        # model's ``range(N)`` default clobbers live state, DP=1 included.
        empty_slots = model_input.prefill_empty_slots
        if empty_slots is None and len(batch_size_per_dp) > 1:
            # TODO: the model should only require DP ranks, but passing
            # "global" user ids instead for backwards compatibility.
            stride = self.tt_per_lane_max_num_seqs
            empty_slots = []
            for dp_rank, sz in enumerate(batch_size_per_dp):
                for i in range(int(sz)):
                    empty_slots.append(dp_rank * stride + i)
        if empty_slots is not None:
            kwargs["empty_slots"] = list(empty_slots)

        if self.request_specific_rope:
            tt_out, rope_deltas = self.model.prefill_forward(**kwargs)
            # Store rope_deltas for each prefilled request
            for i, req_id in enumerate(self.input_batch.req_ids):
                self.requests[req_id].mrope_position_delta = rope_deltas[i].item()
            self.async_decode.note_prefill_submitted()
            return tt_out
        self._gemma_batch_observation_index = getattr(self, "_gemma_batch_observation_index", 0) + 1
        _gemma_observation_index = self._gemma_batch_observation_index
        _gemma_observation_start = time.perf_counter()
        logger.info("GEMMA_PREFILL_ENTRY " + json.dumps({
            "index": _gemma_observation_index,
            "wall_s": time.time(),
            "request_ids": list(model_input.row_req_ids),
            "prompt_lens": list(map(int, model_input.prompt_lens)),
            "starts": model_input.input_positions.tolist(),
            "slots": kwargs.get("empty_slots"),
            "retain_history": kwargs.get("retain_prefill_history"),
            "device_sampling": bool(model_input.perform_device_sampling),
        }))
        tt_out = self.model.prefill_forward(**kwargs)
        logger.info("GEMMA_PREFILL_ENTRY_END " + json.dumps({
            "index": _gemma_observation_index,
            "wall_s": time.time(),
            "submit_s": time.perf_counter() - _gemma_observation_start,
        }))
        self.async_decode.note_prefill_submitted()
        return tt_out

    def _forward_with_model_input(
        self,
        model_input: TTModelInput,
    ) -> _SyncForward | None:
        """Run the synchronous device forward for a prebuilt model input.

        Submits prefill or decode and finalizes the device read, but does not
        sample. Returns the materialized forward state, or ``None`` when the
        batch is empty (no rows to sample).
        """
        is_decode = model_input.prompt_lens is None

        batch_size_per_dp = model_input.unpadded_batch_size
        if not isinstance(batch_size_per_dp, list):
            batch_size_per_dp = [batch_size_per_dp]
        if not any(bs > 0 for bs in batch_size_per_dp):
            return None

        sampling_params = model_input.tt_sampling_params
        perform_device_sampling = model_input.perform_device_sampling
        tt_log_probs = None
        decode_submission = None

        # Execute model
        spec_hidden = None
        if not is_decode:
            tt_out = self.submit_prefill(model_input, batch_size_per_dp)
            # Prefill returns the raw model output: an optional
            # ``(tokens/logits, logprobs)`` tuple when device sampling ran.
            # Unpack it here. The decode branch below is already unpacked by
            # ``finalize_decode``, so re-running this would double-unpack and
            # assert on an already-extracted tensor.
            assert isinstance(sampling_params.enable_log_probs, torch.Tensor)
            if perform_device_sampling and sampling_params.enable_log_probs.any():
                assert isinstance(tt_out, tuple) and len(tt_out) == 2
                tt_out, tt_log_probs = tt_out
            elif isinstance(tt_out, tuple):
                tt_out, _ = tt_out
        else:
            decode_submission = self.async_decode.submit_decode(
                model_input, read_from_device=False, async_read=False
            )
            if model_input.defer_device_sampling:
                tt_out = decode_submission.tt_out
            else:
                finalized = self.async_decode.finalize_decode(decode_submission)
                assert finalized is not None
                # ``finalize_decode`` already unpacked ``(tt_out, tt_log_probs)``.
                tt_out = finalized.tt_out
                tt_log_probs = finalized.tt_log_probs
            batch_size_per_dp = decode_submission.batch_size_per_dp
            sampling_params = decode_submission.sampling_params
            perform_device_sampling = decode_submission.perform_device_sampling
            spec_hidden = decode_submission.spec_hidden

        return _SyncForward(
            tt_out=tt_out,
            tt_log_probs=tt_log_probs,
            sampling_params=sampling_params,
            model_input=model_input,
            batch_size_per_dp=batch_size_per_dp,
            perform_device_sampling=perform_device_sampling,
            is_decode=is_decode,
            decode_submission=decode_submission,
            spec_hidden=spec_hidden,
        )

    def _sample_sync_forward(
        self, fwd: _SyncForward
    ) -> tuple[list[torch.Tensor], list[LogprobsTensors | None]]:
        """Sample a forward produced by ``_forward_with_model_input``."""
        if getattr(fwd.model_input, "defer_device_sampling", False):
            if fwd.decode_submission is None:
                raise RuntimeError("deferred decode is missing its device submission")
            finalized = self.async_decode.complete_deferred_device_sampling(
                fwd.decode_submission,
                fwd.model_input,
            )
            fwd = replace(
                fwd,
                tt_out=finalized.tt_out,
                tt_log_probs=finalized.tt_log_probs,
                model_input=replace(
                    fwd.model_input,
                    grammar_bitmask=[None],
                    defer_device_sampling=False,
                ),
            )
        return self._get_output_tokens(
            tt_out=fwd.tt_out,
            tt_log_probs=fwd.tt_log_probs,
            sampling_params=fwd.sampling_params,
            model_input=fwd.model_input,
            batch_size_per_dp=fwd.batch_size_per_dp,
            perform_device_sampling=fwd.perform_device_sampling,
            is_decode=fwd.is_decode,
        )

    def _get_output_tokens(
        self,
        tt_out: torch.Tensor,
        tt_log_probs: torch.Tensor | None,
        sampling_params: TTSamplingParams,
        model_input: TTModelInput,
        batch_size_per_dp: list[int],
        perform_device_sampling: bool,
        is_decode: bool,
    ) -> tuple[list[torch.Tensor], list[LogprobsTensors | None]]:
        """Return sampled tokens per DP rank using concatenated model
        outputs, plus optional logprobs per DP rank.

        If perform_device_sampling is True, tokens are already sampled on
        device. Otherwise, sample on host using host_sampler.

        Args:
            tt_out: Model output (logits or tokens depending on sampling mode)
            tt_log_probs: Optional logprobs from device sampling

        Returns:
            Tuple of (sampled_token_ids_per_dp, logprobs_per_dp).
            Each element in logprobs_per_dp is None if logprobs were not
            requested for that DP rank.
        """
        sampled_token_ids_per_dp: list[torch.Tensor] = []
        logprobs_per_dp: list[LogprobsTensors | None] = []

        start = 0
        for dp_rank, sz in enumerate(batch_size_per_dp):
            if sz <= 0:
                sampled_token_ids_per_dp.append(
                    torch.empty((0, self._output_tokens_per_step), dtype=torch.int32)
                )
                logprobs_per_dp.append(None)
                if is_decode:
                    # Fixed stride segments per DP rank for decode
                    start += self.tt_per_lane_max_num_seqs
                continue

            # Active requests are packed at the front of each rank's segment,
            # so this rank's rows are the contiguous ``[start, start + sz)``
            # range.
            rows = torch.arange(start, start + sz, dtype=torch.long)

            def _take(tensor: torch.Tensor, _rows: torch.Tensor = rows) -> torch.Tensor:
                return tensor[_rows]

            # An adaptive block model's PREFILL emits one plain host-sampled
            # anchor token (its blocks come only from solo decode steps); any
            # other host-sampled block step cannot construct the canvas.
            if (
                not perform_device_sampling
                and self._is_block_output_model
                and not (self._is_adaptive_block_output and not is_decode)
            ):
                raise ValueError(
                    "Block-output step fell back to host sampling; "
                    "host sampling cannot construct a multi-token canvas"
                )

            if (
                not is_decode
                and model_input.intermediate_prefill_mask is not None
                and bool(model_input.intermediate_prefill_mask[rows].all())
            ):
                # Every row is mid-prompt; there is nothing to sample and the
                # placeholders are dropped when the output is built.
                assert self._output_tokens_per_step == 1, (
                    "Intermediate-prefill suppression assumes one sampled token "
                    "per request"
                )
                next_token_ids = torch.zeros(sz, dtype=torch.int32)
                logprobs_per_dp.append(None)
            elif not perform_device_sampling:
                logits = tt_out[rows, -1, :]

                grammar_bitmask = model_input.grammar_bitmask[dp_rank]

                if grammar_bitmask is not None:
                    # match shape of logits, which are now unpadded on batch dim
                    grammar_bitmask = grammar_bitmask[:sz, :]
                    self.apply_grammar_bitmask(logits, grammar_bitmask)

                # Extract sampling params for this DP rank from concatenated
                # tensors.
                assert isinstance(sampling_params.temperature, torch.Tensor)
                assert isinstance(sampling_params.top_k, torch.Tensor)
                assert isinstance(sampling_params.top_p, torch.Tensor)
                assert isinstance(sampling_params.presence_penalty, torch.Tensor)
                assert isinstance(sampling_params.frequency_penalty, torch.Tensor)
                assert isinstance(sampling_params.repetition_penalty, torch.Tensor)
                assert isinstance(sampling_params.seed, torch.Tensor)
                temperature = _take(sampling_params.temperature)
                top_k = _take(sampling_params.top_k)
                top_p = _take(sampling_params.top_p)
                presence_penalty = _take(sampling_params.presence_penalty)
                frequency_penalty = _take(sampling_params.frequency_penalty)
                repetition_penalty = _take(sampling_params.repetition_penalty)

                # Determine if all greedy (temperature == 0.0) or all random
                all_greedy = (temperature == 0.0).all().item()
                all_random = (temperature != 0.0).all().item()

                generators = model_input.generators_list[dp_rank]

                # Determine if penalties are needed
                no_penalties = (
                    (presence_penalty == 0.0).all().item()
                    and (frequency_penalty == 0.0).all().item()
                    and (repetition_penalty == 1.0).all().item()
                )

                # Output history as list[list[int]] (filter TT -1 padding).
                output_token_ids: list[list[int]] = []
                if is_decode and model_input.output_tokens is not None:
                    output_tokens = _take(model_input.output_tokens)
                    for i in range(sz):
                        output_tokens_i = output_tokens[i].tolist()
                        output_token_ids.append(
                            [tok for tok in output_tokens_i if tok != -1]
                        )
                else:
                    output_token_ids = [[] for _ in range(sz)]

                # Prompt tokens for penalties: must be int64 and padded with a
                # valid index (vocab_size), not TT's -1 sentinel.
                prompt_token_ids: torch.Tensor | None = None
                if not no_penalties:
                    if is_decode and model_input.prompt_tokens is not None:
                        prompt_token_ids = _take(model_input.prompt_tokens).to(
                            torch.int64
                        )
                        prompt_token_ids = prompt_token_ids.masked_fill(
                            prompt_token_ids == -1, self.vocab_size
                        )
                    elif not is_decode:
                        prompt_token_ids = model_input.input_tokens[
                            start : start + sz
                        ].to(torch.int64)
                        assert model_input.prompt_lens is not None
                        prompt_lens_t = torch.as_tensor(
                            model_input.prompt_lens[start : start + sz],
                            dtype=torch.int64,
                        )
                        positions = torch.arange(
                            prompt_token_ids.shape[1],
                        ).unsqueeze(0)
                        pad_mask = positions >= prompt_lens_t.unsqueeze(1)
                        prompt_token_ids = prompt_token_ids.masked_fill(
                            pad_mask, self.vocab_size
                        )

                # Get host-only sampling params from model_input
                # (per-rank lists).
                # These are populated for both DP and non-DP cases.
                rank_max_num_logprobs = model_input.max_num_logprobs[dp_rank]
                allowed_token_ids_mask = model_input.allowed_token_ids_mask_list[  # noqa: E501
                    dp_rank
                ]
                if allowed_token_ids_mask is not None:
                    # Slice to actual batch size for this rank
                    allowed_token_ids_mask = allowed_token_ids_mask[:sz]

                bad_words_token_ids = model_input.bad_words_token_ids_list[dp_rank]

                logitsprocs = model_input.logitsprocs_list[dp_rank]
                if logitsprocs is None:
                    logitsprocs = LogitsProcessors()

                # Create SamplingMetadata for this DP rank
                sampling_metadata = SamplingMetadata(
                    temperature=temperature if not all_greedy else None,
                    all_greedy=all_greedy,
                    all_random=all_random,
                    top_p=top_p,
                    top_k=top_k,
                    generators=generators,
                    max_num_logprobs=rank_max_num_logprobs,
                    no_penalties=no_penalties,
                    prompt_token_ids=prompt_token_ids,
                    frequency_penalties=frequency_penalty,
                    presence_penalties=presence_penalty,
                    repetition_penalties=repetition_penalty,
                    output_token_ids=output_token_ids,
                    allowed_token_ids_mask=allowed_token_ids_mask,
                    bad_words_token_ids=bad_words_token_ids,
                    logitsprocs=logitsprocs,
                )

                sampler_output = self.host_sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
                next_token_ids = sampler_output.sampled_token_ids
                # Capture logprobs for this DP rank
                logprobs_per_dp.append(sampler_output.logprobs_tensors)
            else:  # sample on device
                assert model_input.grammar_bitmask[dp_rank] is None, (
                    "grammar bitmask is set but device sampling can't apply it"
                )

                next_token_ids = _take(tt_out).reshape(sz, -1)
                allowed_widths = (
                    (1, self._output_tokens_per_step)
                    if self._is_adaptive_block_output
                    else (self._output_tokens_per_step,)
                )
                if next_token_ids.shape[1] not in allowed_widths:
                    raise ValueError(
                        "Model output width violates output_tokens_per_step: "
                        f"{next_token_ids.shape[1]} not in {allowed_widths}"
                    )
                rank_max_num_logprobs = model_input.max_num_logprobs[dp_rank]
                # Extract logprobs if available from device sampling
                # Always tensors - turned into lists only when passing to model
                assert isinstance(sampling_params.enable_log_probs, torch.Tensor)
                rank_enable_lp = _take(sampling_params.enable_log_probs)
                if rank_enable_lp.any():
                    # Sanity check for if we correctly detect
                    # when logprobs are supported.
                    if next_token_ids.shape[1] != 1:
                        raise ValueError(
                            "Device logprobs support one output token per step; "
                            "block-output requests must reject logprobs"
                        )
                    if tt_log_probs is None:
                        raise ValueError(
                            "Model did not return device logprobs for a request "
                            "that enabled them"
                        )
                    logprobs_per_dp.append(
                        build_device_logprobs(
                            tt_log_probs=tt_log_probs,
                            sampled_token_ids=next_token_ids.reshape(sz),
                            rows=rows,
                            max_num_logprobs=rank_max_num_logprobs or 0,
                        )
                    )
                else:
                    logprobs_per_dp.append(None)

            sampled_token_ids_per_dp.append(next_token_ids.reshape(sz, -1))

            if is_decode:
                # Fixed stride segments per DP rank for decode
                start += self.tt_per_lane_max_num_seqs
            else:
                # Prefill packed contiguously
                start += sz

        return sampled_token_ids_per_dp, logprobs_per_dp

    def apply_grammar_bitmask(
        self, logits: torch.Tensor, grammar_bitmask: torch.Tensor
    ) -> None:
        """Apply structured output grammar constraints to logits in-place"""
        # The grammar bitmask is compressed as packed int32 values
        # where each bit represents one token. We need to unpack it
        # like the TPU model runner does.
        # Ones in the compressed bitmask represent tokens that are allowed.

        # TODO this is likely a quite inefficient way of doing it on host.

        # grammar_bitmask: (batch_size, bitmask_size)
        # logits: (batch_size, vocab_size)
        unpacked_bitmask = (
            torch.bitwise_right_shift(
                grammar_bitmask[:, :, None],
                self.structured_output_arange[None, None, :],
            )
            & 1
        ) == 0
        unpacked_bitmask = unpacked_bitmask.reshape(grammar_bitmask.shape[0], -1)[
            :, : logits.shape[-1]
        ]
        logits.masked_fill_(unpacked_bitmask, -float("inf"))

    def _build_runner_output(
        self,
        sampled_token_ids: torch.Tensor,
        logprobs: LogprobsLists | None = None,
        req_ids: list[str] | None = None,
        req_id_to_index: dict[str, int] | None = None,
    ) -> ModelRunnerOutput:
        num_reqs = len(req_ids) if req_ids is not None else self.input_batch.num_reqs
        output_req_ids = (
            list(req_ids)
            if req_ids is not None
            else list(self.input_batch.req_ids[:num_reqs])
        )
        output_req_id_to_index = (
            dict(req_id_to_index)
            if req_id_to_index is not None
            else {req_id: idx for idx, req_id in enumerate(output_req_ids)}
        )
        sampled_token_ids = _coerce_output_block(
            sampled_token_ids, num_reqs, self._tt_committed_width(sampled_token_ids)
        )

        sampled_token_ids_np = sampled_token_ids.numpy()
        if sampled_token_ids_np.dtype != np.int32:
            sampled_token_ids_np = sampled_token_ids_np.astype(np.int32, copy=False)

        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = dict.fromkeys(
            (output_req_ids[i] for i in range(num_reqs)), None
        )
        sampled_token_id_lists = [
            [int(token_id) for token_id in row] for row in sampled_token_ids_np.tolist()
        ]

        return ModelRunnerOutput(
            req_ids=output_req_ids,
            req_id_to_index=output_req_id_to_index,
            sampled_token_ids=sampled_token_id_lists,
            logprobs=logprobs,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
        )

    def _apply_sampled_tokens_to_state(
        self,
        sampled_token_ids: torch.Tensor,
        req_ids: list[str] | None = None,
        skip_req_ids: set[str] | None = None,
    ) -> None:
        # When applying a deferred async step, the write row is resolved live
        # from ``req_id_to_index`` (below), not from the row captured at submit
        # time: lane mode pins each request to a stable slot for its lifetime,
        # while the scheduler lifecycle skip set rejects stale results before a
        # live row is resolved. ``req_id_to_index`` is therefore the source of
        # truth for the target row.
        use_captured_req_ids = req_ids is not None
        num_reqs = len(req_ids) if req_ids is not None else self.input_batch.num_reqs
        num_out_tokens = self._tt_committed_width(sampled_token_ids)
        sampled_token_ids = _coerce_output_block(
            sampled_token_ids, num_reqs, num_out_tokens
        )

        sampled_token_ids_np = sampled_token_ids.numpy()
        if sampled_token_ids_np.dtype != np.int32:
            sampled_token_ids_np = sampled_token_ids_np.astype(np.int32, copy=False)

        max_model_len = self.model_config.max_model_len

        if not use_captured_req_ids:
            rows = np.arange(num_reqs)
            start_idxs = self.input_batch.num_tokens[rows]
            end_idxs = start_idxs + num_out_tokens
            max_end = int(end_idxs.max()) if num_reqs > 0 else 0
            if max_end > max_model_len:
                if num_out_tokens == 1:
                    raise ValueError(
                        "Sampled token IDs exceed the max model length. "
                        f"Total number of tokens: {max_end} > max_model_len: "
                        f"{max_model_len}"
                    )
                # A block canvas is physical: a bypassed request admitted with
                # a zero-canvas budget still gets one full canvas back. Keep
                # only the slice that fits; the scheduler's stop check then
                # finishes the request length-capped through the normal output
                # path instead of this raise killing the engine.
                logger.warning(
                    "Block canvas exceeds max_model_len=%d for request(s) %s; "
                    "keeping only the slice that fits",
                    max_model_len,
                    ", ".join(
                        self.input_batch.req_ids[i]
                        for i in range(num_reqs)
                        if int(end_idxs[i]) > max_model_len
                    ),
                )
                end_idxs = np.maximum(np.minimum(end_idxs, max_model_len), start_idxs)

            for req_idx in range(num_reqs):
                start_idx = int(start_idxs[req_idx])
                end_idx = int(end_idxs[req_idx])
                block = sampled_token_ids_np[req_idx][: end_idx - start_idx]
                self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = block
                output_token_ids = self.input_batch.req_output_token_ids[req_idx]
                assert output_token_ids is not None
                output_token_ids.extend(int(token_id) for token_id in block)
            self.input_batch.num_tokens[rows] = end_idxs
            return

        assert req_ids is not None
        captured_req_ids = req_ids
        if num_out_tokens == 1:
            # Autoregressive overflow means scheduler accounting corrupted;
            # validate every live row before mutating any of them, so the
            # raise never leaves a partially applied batch. Block canvases
            # are clipped below instead of raising.
            for req_idx, req_id in enumerate(captured_req_ids):
                if skip_req_ids is not None and req_id in skip_req_ids:
                    continue
                req_state = self.requests.get(req_id)
                assert req_state is not None, (
                    "captured request missing from runner state while applying "
                    f"sampled tokens: req_id={req_id!r}"
                )
                current_row = self.input_batch.req_id_to_index.get(req_id)
                if current_row is not None:
                    end_idx = (
                        int(self.input_batch.num_tokens[current_row]) + num_out_tokens
                    )
                    if end_idx > max_model_len:
                        raise ValueError(
                            "Sampled token IDs exceed the max model length. "
                            f"Total number of tokens: {end_idx} > max_model_len: "
                            f"{max_model_len}"
                        )

        for req_idx, req_id in enumerate(captured_req_ids):
            if skip_req_ids is not None and req_id in skip_req_ids:
                continue
            req_state = self.requests.get(req_id)
            assert req_state is not None, (
                "captured request missing from runner state while applying sampled "
                f"tokens: req_id={req_id!r}"
            )
            current_row = self.input_batch.req_id_to_index.get(req_id)
            if current_row is not None:
                start_idx = int(self.input_batch.num_tokens[current_row])
                end_idx = start_idx + num_out_tokens
                if end_idx > max_model_len:
                    logger.warning(
                        "Block canvas exceeds max_model_len=%d for request %s; "
                        "keeping only the slice that fits",
                        max_model_len,
                        req_id,
                    )
                    end_idx = max(start_idx, max_model_len)
                block = sampled_token_ids_np[req_idx][: end_idx - start_idx]
                self.input_batch.token_ids_cpu[current_row, start_idx:end_idx] = block
                self.input_batch.num_tokens[current_row] = end_idx
            else:
                block = sampled_token_ids_np[req_idx]

            req_state.output_token_ids.extend(int(token_id) for token_id in block)

    def _tt_committed_width(self, sampled_token_ids: torch.Tensor) -> int:
        """Resolve one step's committed output width.

        Fixed at ``output_tokens_per_step``, except an ADAPTIVE block model's
        non-block steps (batched decodes and prefill anchors) emit exactly one
        valid token per request -- the scheduler reserved exactly one
        placeholder for those steps, so the emitted row length IS the step's
        contract. Any other width still fails _coerce_output_block.
        """
        if (
            self._is_adaptive_block_output
            and sampled_token_ids.dim() == 2
            and sampled_token_ids.shape[1] == 1
        ):
            return 1
        return self._output_tokens_per_step

    def apply_and_build_runner_output(
        self,
        sampled_token_ids: torch.Tensor,
        logprobs: LogprobsLists | None = None,
        req_ids: list[str] | None = None,
        req_id_to_index: dict[str, int] | None = None,
    ):
        """Apply sampled tokens to runner state and build `ModelRunnerOutput`.

        Updates persistent runner state from sampled tokens and returns the
        `ModelRunnerOutput` consumed by the rest of vLLM.
        """
        self._apply_sampled_tokens_to_state(
            sampled_token_ids=sampled_token_ids,
            req_ids=req_ids,
        )
        return self._build_runner_output(
            sampled_token_ids=sampled_token_ids,
            logprobs=logprobs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )

    def _enqueue_deferred_state_apply(
        self,
        sampled_token_ids: torch.Tensor,
        req_ids: list[str],
        runner_output: ModelRunnerOutput,
    ) -> None:
        """Queue sampled host state until scheduler lifecycle is known.

        Final prefills are sampled synchronously, but under AsyncScheduler their
        published output is still protected by a placeholder. Queue them beside
        completed async decodes so a forced prefix reset can leave the published
        row intact for discard accounting without first appending its stale
        token to runner state.
        """
        now = time.perf_counter_ns()
        self.async_decode.enqueue_completed_decode_step(
            CompletedDecodeStep(
                sampled_token_ids=sampled_token_ids,
                logprobs=None,
                context=SubmittedStepContext(
                    req_ids=list(req_ids),
                    req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
                    submit_time_ns=now,
                ),
                completion_time_ns=now,
                runner_output=runner_output,
            )
        )

    def defer_state_apply_and_build_runner_output(
        self,
        sampled_token_ids: torch.Tensor,
        logprobs: LogprobsLists | None = None,
        req_ids: list[str] | None = None,
        req_id_to_index: dict[str, int] | None = None,
    ) -> ModelRunnerOutput:
        assert req_ids is not None, "deferred synchronous output needs request IDs"
        runner_output = self._build_runner_output(
            sampled_token_ids=sampled_token_ids,
            logprobs=logprobs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )
        self._enqueue_deferred_state_apply(sampled_token_ids, req_ids, runner_output)
        return runner_output

    def warmup_model(self) -> None:
        # Two-phase warmup: eager preparation, then trace-enabled warmup.
        #
        # Phase 1 warms the eager prefill/decode paths without capturing
        # traces. Generators own preparation of the programs and persistent
        # buffers their traces need. Keep prefill first in Phase 2 because
        # some custom adapters defer its preparation to that call.
        #
        # Assumptions / limitations:
        #   1. Traced and non-traced code paths must use the same ops.
        #      If a model uses different operators when enable_trace=False
        #      vs True, Phase 1 will not compile the ops that Phase 2
        #      traces, and new compilations during trace capture will
        #      allocate corruptible buffers.
        #   2. Prefill warmup must cover all supported sequence lengths.
        #      If a new sequence length appears during inference, its
        #      first compilation will allocate new kernel cache entries
        #      (including reshape caches) that can corrupt active traces.
        #
        # See: https://github.com/tenstorrent/tt-metal/commit/5043de3df5
        trace_prefill_mode = self.trace_mode in ["all"]
        trace_decode_mode = self.trace_mode in ["all", "decode_only"]
        sample_on_device_mode = getattr(TTPlatform, "sample_on_device_mode", None)
        assert sample_on_device_mode in (None, "all", "decode_only")
        prefill_kwargs = dict(
            kv_cache=self.kv_caches,
            can_sample_on_device=sample_on_device_mode == "all",
        )
        decode_kwargs = dict(
            kv_cache=self.kv_caches,
            max_batch_size=self.tt_max_batch_size,
            num_blocks=self.max_num_blocks_per_req,
            can_sample_on_device=sample_on_device_mode in ("all", "decode_only"),
        )
        if self.supports_device_grammar:
            decode_kwargs["can_sample_device_grammar"] = True

        # Phase 1: eager warmup (no trace capture). tt-metal #55343 also
        # stages generic decode trace inputs here, using the profiled
        # num_blocks above, so Phase 2 can reuse their device allocations.
        self.model.warmup_model_prefill(enable_trace=False, **prefill_kwargs)
        self.model.warmup_model_decode(enable_trace=False, **decode_kwargs)

        # Reset prefill warmup flag so Phase 2 re-runs with tracing
        if hasattr(self.model, "already_warmed_up_prefill"):
            self.model.already_warmed_up_prefill = False

        # Phase 2: capture prefill before decode. Custom adapters such as
        # Qwen can defer persistent prefill allocations / compilation until
        # this trace-enabled warmup; they must not run behind decode traces.
        # Generic decode inputs are already staged by tt-metal #55343, so
        # restoring this order does not require allocating them after prefill
        # capture. Older tt-metal revisions lack that protection (see #122).
        if trace_prefill_mode:
            self.model.warmup_model_prefill(enable_trace=True, **prefill_kwargs)
        if trace_decode_mode:
            self.model.warmup_model_decode(enable_trace=True, **decode_kwargs)
