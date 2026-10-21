# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

import gemma_history_reservation as _history_reservation
import sys
from enum import Enum
from typing import TYPE_CHECKING, cast

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_tt_plugin.config import (
    get_tt_admitted_request_limit,
    get_tt_adaptive_block_max_prompt_tokens,
    get_tt_block_kv_extent_tokens,
    get_tt_decode_interleave_config,
    get_tt_output_tokens_per_step,
    get_tt_prefill_chunk_alignment,
    get_tt_spec_plan,
    is_tt_adaptive_block_output_model,
    is_tt_block_output_model,
)
from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.spec_admission import MODEL_OWNED_DRAFT_METHOD

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_tt_logger(__name__)


# ``SchedulerOutput`` has no backend metadata field. Like lane step state, this
# TT-only signal is attached to the mutable output object and therefore follows
# it through the executor serialization boundary. Counts, rather than a set of
# request IDs, identify exactly how many newest in-flight frames a wholesale
# prefix-cache reset made stale.
_TT_FORCED_RESET_DISCARD_COUNTS_ATTR = "_tt_forced_reset_discard_counts"
_TT_OUTPUT_FRAME_REQ_IDS_ATTR = "_tt_output_frame_req_ids"


def set_tt_forced_reset_discard_counts(
    scheduler_output: SchedulerOutput, counts: dict[str, int]
) -> None:
    if counts:
        setattr(scheduler_output, _TT_FORCED_RESET_DISCARD_COUNTS_ATTR, dict(counts))


def get_tt_forced_reset_discard_counts(
    scheduler_output: SchedulerOutput,
) -> dict[str, int]:
    return dict(getattr(scheduler_output, _TT_FORCED_RESET_DISCARD_COUNTS_ATTR, {}))


# Per-step block-output reconciliation decisions, keyed by request id. Attached
# to the ``SchedulerOutput`` -- the ONLY object that re-associates a step's
# output with the scheduling decision that produced it: under async scheduling
# ``schedule()`` for step K+1 (which mutates per-``Request`` state) runs BEFORE
# ``update_from_output`` commits step K, so a single overwritten ``Request``
# slot would hand step K's commit the K+1 decision. The engine core holds each
# in-flight step's ``SchedulerOutput`` in its batch queue and passes it back to
# ``update_from_output``, so a map carried here always matches the output being
# committed (in sync mode the pairing is trivially the same step). Mirrors the
# forced-reset-count pattern above.
_TT_BLOCK_STEP_DECISIONS_ATTR = "_tt_block_step_decisions"


def set_tt_block_step_decisions(
    scheduler_output: SchedulerOutput, decisions: dict[str, bool]
) -> None:
    setattr(scheduler_output, _TT_BLOCK_STEP_DECISIONS_ATTR, dict(decisions))


def get_tt_block_step_decisions(
    scheduler_output: SchedulerOutput,
) -> dict[str, bool]:
    return dict(getattr(scheduler_output, _TT_BLOCK_STEP_DECISIONS_ATTR, {}))


class TTSchedulingMode(Enum):
    DEFAULT = "default"
    DECODE_ONLY = "decode_only"
    PREFILL_ONLY = "prefill_only"

    @classmethod
    def from_prefill_intent(cls, prefill_intent: int) -> "TTSchedulingMode":
        if prefill_intent == 0:
            return cls.DECODE_ONLY
        if prefill_intent == 1:
            return cls.PREFILL_ONLY
        raise ValueError(f"Invalid TT scheduling intent: {prefill_intent}")


def spec_lookahead_tokens(plan, num_spec_tokens: int, method: str | None) -> int:
    """KV slots to reserve past a step's own tokens for a model-owned drafter.

    Such a drafter proposes for the next step inside the current one: after
    the accept walk commits, its fused body runs over the anchor plus every
    draft and writes K/V for those K+1 positions, none of which the current
    step's allocation covers. Upstream reserves lookahead only for the drafter
    methods it knows, and ``custom_class`` is not one of them, so without this
    reservation the rows that cross into the next block land in the null block
    and the first token that reads them diverges from plain decode.

    An ngram launch also carries an admitted plan, but its drafts come from the
    plugin and its target verifies them inside the step's own allocation, so it
    reserves nothing here.
    """
    if plan is None or num_spec_tokens <= 0 or method != MODEL_OWNED_DRAFT_METHOD:
        return 0
    return num_spec_tokens + 1


class TTDecodeInterleavePolicy:
    """Bounds how many consecutive prefill steps may stall running decodes.

    A TT step carries either prefill rows or decode rows, never both, so a
    prompt split into N chunks occupies N consecutive prefill steps and every
    running decode waits for all of them. This policy inserts decode-only steps
    into such a run. ``docs/SCHEDULING.md`` covers why, how the two counts
    interact, and what the defaults rest on.

    Three invariants the code depends on:

    - Reaching the decode allowance never forces prefill. It only stops the
      policy choosing decode; whether prefill then runs is up to the scheduler
      and to whether prefill work is pending.
    - ``has_running_decode`` must exclude partial-prefill continuations. A
      decode step cannot advance one, so interleaving on its account trades a
      productive prefill step for an empty one.
    - A decode step taken with no prefill pending is an ordinary decode, not
      part of an insertion, and clears both counters.
    """

    def __init__(self, vllm_config: "VllmConfig") -> None:
        (
            self._enabled,
            self._prefill_steps,
            self._decode_steps,
        ) = get_tt_decode_interleave_config(vllm_config)
        self._prefill_run = 0
        self._decode_run = 0

    def wants_decode_step(
        self, *, has_pending_prefill: bool, has_running_decode: bool
    ) -> bool:
        """Whether to spend this step on decode although prefill work is pending.

        ``has_running_decode`` must exclude partial-prefill continuations: a
        decode step cannot advance one, so interleaving on its account would
        trade a productive prefill step for an empty one.
        """
        if not self._enabled or not has_pending_prefill or not has_running_decode:
            return False
        return (
            self._prefill_run >= self._prefill_steps
            and self._decode_run < self._decode_steps
        )

    def record_step(self, *, is_decode: bool, prefill_pending: bool) -> None:
        """Advance the counters with the phase the step actually ran.

        Called for every step, including one that scheduled no tokens. A
        decode-only step that schedules nothing (upstream's running loop skips a
        request whose async placeholders have already reached ``max_tokens``)
        still consumes its decode allowance; leaving the counters untouched
        there would re-pick decode on every following step and livelock the
        engine on empty steps.

        ``prefill_pending`` says whether any prefill work existed when the step
        was chosen. A decode step taken with nothing pending is an ordinary
        decode, not part of an insertion, so it clears both counters: counting
        it would leave ``_prefill_run`` at its bound with part of the decode
        allowance already spent, and the next arriving prompt would then lose
        its first prefill step to an insertion it was never part of.
        """
        if not prefill_pending:
            self._prefill_run = 0
            self._decode_run = 0
            return
        if is_decode:
            self._decode_run += 1
            if self._decode_run >= self._decode_steps:
                # Allowance spent: require a fresh run of prefill steps before
                # the next insertion, which is what stops the policy from
                # holding the device in decode while a prompt waits.
                self._decode_run = 0
                self._prefill_run = 0
        else:
            self._prefill_run += 1
            self._decode_run = 0


def _effective_prefill_step_limit(scheduler_config) -> tuple[int, str]:
    """The most tokens one scheduling step can hand a single new request, named.

    Resolved exactly as vLLM's ``Scheduler.__init__`` resolves its per-step
    token budget: ``max_num_scheduled_tokens`` when it is set, otherwise
    ``max_num_batched_tokens``. A positive ``long_prefill_token_threshold``
    then caps that limit. Returns ``(limit, name_of_the_binding_cap)``; the
    limit is ``sys.maxsize`` when nothing is configured.
    """
    scheduled = getattr(scheduler_config, "max_num_scheduled_tokens", None)
    if scheduled is not None:
        limit, name = int(scheduled), "max_num_scheduled_tokens"
    else:
        batched = int(getattr(scheduler_config, "max_num_batched_tokens", 0) or 0)
        limit, name = (
            (batched, "max_num_batched_tokens")
            if batched > 0
            else (
                sys.maxsize,
                "max_num_batched_tokens",
            )
        )
    threshold = int(getattr(scheduler_config, "long_prefill_token_threshold", 0) or 0)
    if 0 < threshold < limit:
        limit, name = threshold, "long_prefill_token_threshold"
    return limit, name


def _validate_prefill_chunk_alignment(align: int, scheduler_config) -> None:
    """Reject caps that can never let a new request reach the alignment grid.

    ``_align_prefill_chunk_end`` defers a new request whose chunk falls short
    of the grid; that only converges if a later step can offer a larger chunk.
    The binding cap is the one vLLM schedules with (``max_num_scheduled_tokens``
    when set, else ``max_num_batched_tokens``), further capped by a positive
    ``long_prefill_token_threshold``.
    """
    limit, name = _effective_prefill_step_limit(scheduler_config)
    if limit < align:
        raise ValueError(
            f"additional_config['tt']['prefill_chunk_alignment']={align} cannot be "
            f"reached: scheduler {name}={limit} is a permanent per-step cap "
            "below the "
            "alignment. Raise the cap to at least the alignment or set "
            "prefill_chunk_alignment to 0."
        )


class TTScheduler(AsyncScheduler):
    """Scheduler for the TT (Tenstorrent) platform.

    TT constraints:
    - No mixed prefill+decode batches: each batch is either all-prefill
      or all-decode.
    - Token-chunked prefill is allowed: a long prefill may be split across
      steps. After a partial chunk the base scheduler keeps the request in
      ``running`` with ``is_prefill_chunk=True``; later prefill steps schedule
      the next chunk until the prompt is fully computed. Those continuations
      count as prefill work here, never as decodes.

    The base scheduler holds temporarily blocked prefill requests (e.g. while
    a structured-output grammar compiles) in ``skipped_waiting`` rather than
    ``waiting``. Its prefill loop revisits that queue and promotes requests
    whose dependency is ready. Decode-only scheduling must hide both queues so
    that the base scheduler cannot admit a prefill into a decode step.

    Inherits from AsyncScheduler to get num_output_placeholders support.
    TT uses this scheduler in both sync and async execution modes:
    - with async_scheduling=False, it behaves as the single TT scheduler
      without execution overlap
    - with async_scheduling=True, placeholders allow decode requests to be
      re-scheduled before update_from_output processes the previous step's
      results, enabling host/device overlap
    - under async scheduling a preempted request keeps the tokens it had
      already scheduled but not yet received, and needs no TT-side handling
      for them. Those tokens are valid: the forward that produced them ran to
      completion before the preempt freed any block, device submits form a
      strict queue, and every async op is forced to complete before the next
      prefill, so no later write can reach the KV they were computed against.
      The base class marks them stale, appends them on arrival without
      touching the placeholder count, and keeps the request waiting until
      they drain, so the resumed prefill replays them instead of sampling
      against the old token history. Only the wholesale
      ``reset_prefix_cache`` teardown drops stale output
      (``Request.drop_stale_output``); wiring ordinary preemption into it
      drops valid tokens and silently truncates the response.

    Supports ``set_forced_mode`` for lane coordination:
    - ``TTSchedulingMode.DECODE_ONLY`` forces decode-only (even if waiting
      queue is non-empty).
    - ``TTSchedulingMode.PREFILL_ONLY`` forces prefill-only (and may return an
      empty batch when there is no pending prefill work).
    - ``TTSchedulingMode.DEFAULT`` uses the default policy: prefer prefill
      when pending prefill work exists, but fall back to decode-only if
      prefill cannot make progress and running decode requests exist.
    """

    waiting: RequestQueue
    running: list[Request]
    max_num_running_reqs: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_running_reqs = min(self.max_num_running_reqs, get_tt_admitted_request_limit(self.vllm_config))
        logger.info("GEMMA_SCHEDULER_ADMISSION max_running=%d physical=%d", self.max_num_running_reqs, self.scheduler_config.max_num_seqs)
        self._forced_mode = TTSchedulingMode.DEFAULT
        # Read only in DEFAULT mode. A lane coordinator sets a forced mode
        # before every one of its lanes' schedule() calls and owns the one
        # policy instance for the whole step, so this one stays dormant there.
        self._decode_interleave = TTDecodeInterleavePolicy(self.vllm_config)
        self._pending_forced_reset_discard_counts: dict[str, int] = {}
        # Block-step decisions for the step currently being committed, read off
        # its SchedulerOutput in update_from_output so _update_request_with_output
        # sees the decision that produced THIS output (not a later schedule).
        self._committing_block_step_decisions: dict[str, bool] = {}
        # Request id that owns the adaptive model's SINGLE speculative
        # session, mirrored from scheduling-side facts (see
        # _mirror_spec_session). Only its owner can emit a block.
        self._spec_session_owner: str | None = None
        self._pending_async_output_frames: dict[str, int] = {}
        self._widest_decode_batch_size = 0
        self._prefill_chunk_alignment = (
            get_tt_prefill_chunk_alignment(self.vllm_config)
            if self.scheduler_config.enable_chunked_prefill
            else 0
        )
        if self._prefill_chunk_alignment > 1:
            _validate_prefill_chunk_alignment(
                self._prefill_chunk_alignment, self.scheduler_config
            )
            logger.info(
                "TT scheduler: partial prefill chunks end on %d-token boundaries",
                self._prefill_chunk_alignment,
            )
        self._output_tokens_per_step = get_tt_output_tokens_per_step(self.vllm_config)
        self._is_block_output_model = is_tt_block_output_model(self.vllm_config)
        # Adaptive: emit the block only on a solo decode step; batch >1 decodes
        # as plain baseline. Lets max_num_seqs>1 coexist with block-output.
        self._is_adaptive_block = is_tt_adaptive_block_output_model(self.vllm_config)
        # Prompt-length frontier for the block path (0 = none): a longer prompt
        # is served as plain baseline by the model for its whole lifetime, so
        # its steps reserve width-1 even when solo.
        self._adaptive_block_max_prompt = get_tt_adaptive_block_max_prompt_tokens(
            self.vllm_config
        )
        speculative_config = self.vllm_config.speculative_config
        self.num_lookahead_tokens = max(
            self.num_lookahead_tokens,
            spec_lookahead_tokens(
                get_tt_spec_plan(self.vllm_config),
                self.num_spec_tokens,
                speculative_config.method if speculative_config is not None else None,
            ),
        )
        if self._is_block_output_model:
            # KV pages for the WHOLE step, not just the one token upstream
            # counts. A block-output decode runs several internal verify
            # iterations against vLLM-owned KV before it returns, and those
            # iterations write past the position upstream allocated for:
            # schedule() calls allocate_slots with num_lookahead_tokens, which
            # is 0 here because there is no vLLM speculative_config, and
            # raising Request.num_output_placeholders afterwards accounts for
            # pending OUTPUT tokens without allocating anything. The model's
            # refresh_page_tables then pads the missing columns with zero and
            # the verify reads and writes the null block.
            #
            # Reserved as twice the emitted width: the committed block, plus
            # headroom for the final iteration's verify positions and for
            # accepted tokens carried past the emitted width. Both are bounded
            # by the packed-verify width, which is far below the block width
            # (6 against 64 at the shipped default), so this is generous and
            # costs a page or two per request.
            #
            # Twice the width is NOT a bound in general, though: it only covers
            # the physical extent while the block is at least as wide as the
            # verification. A model configured the other way round -- dFlash at
            # GEMMA4_DFLASH_SERVE_BLOCK=2 with GEMMA4_DFLASH_VERIFY=7 emits 2
            # tokens and writes 8 verification rows -- needs 2 + 8 while this
            # reserves 4, and the step then writes positions that have no
            # request block (vllm-tt-plugin#118 review, finding 2). So a model
            # may declare the extent it actually touches and we honour whichever
            # is larger.
            declared_extent = get_tt_block_kv_extent_tokens(self.vllm_config)
            self.num_lookahead_tokens = max(
                int(getattr(self, "num_lookahead_tokens", 0) or 0),
                2 * int(self._output_tokens_per_step),
                int(declared_extent),
            )
            assert self.num_sampled_tokens_per_step == 1, (
                "Block-output accounting requires upstream to reserve exactly "
                "one sampled-token placeholder"
            )
            assert not self.kv_cache_manager.enable_caching, (
                "Block-output models bypass the kv_cache_manager.cache_blocks "
                "call made by AsyncScheduler._update_request_with_output and "
                "must disable prefix caching"
            )

    def set_forced_mode(self, mode: TTSchedulingMode) -> None:
        self._forced_mode = mode

    # Host-only sampling controls and their neutral values: any of these
    # forces the step onto host sampling (check_perform_device_sampling),
    # which cannot construct a multi-token canvas and kills the engine.
    # Penalties do not force host sampling, but a non-neutral value flips
    # InputBatch.no_penalties and makes every block decode step build (and
    # discard) penalty token tensors that grow with the committed session
    # length; the frontend neutralizes them, so mirror that here for
    # prebuilt requests that bypassed it.
    _BLOCK_HOST_ONLY_SAMPLING_NEUTRAL = (
        ("min_p", 0.0),
        ("min_tokens", 0),
        ("logit_bias", None),
        ("allowed_token_ids", None),
        ("bad_words", None),
        ("_bad_words_token_ids", None),
        ("presence_penalty", 0.0),
        ("frequency_penalty", 0.0),
        ("repetition_penalty", 1.0),
    )

    def add_request(self, request: Request) -> None:
        if self._is_block_output_model:
            existing = self.requests.get(request.request_id)
            if existing is not None and existing.streaming_queue is None:
                # A continuation (next input chunk or the closing sentinel) of
                # a session whose resumable flag the neutralization below
                # scrubbed. The base scheduler asserts on the missing
                # streaming_queue, which would tear down EngineCore. The
                # session cannot accept more input, so drop the message; the
                # live request finishes and notifies its client on its own.
                logger.warning(
                    "Dropping streaming-input continuation for request %s: "
                    "block-output models do not support resumable sessions",
                    request.request_id,
                )
                return
            self._truncate_unservable_block_prompt(request)
            self._align_block_output_max_tokens(request)
            self._neutralize_block_output_host_sampling(request)
        super().add_request(request)

    def _truncate_unservable_block_prompt(self, request: Request) -> None:
        """Contain a bypassed prompt that leaves no room for a whole canvas.

        Frontend validation rejects such prompts; a prebuilt EngineCoreRequest
        skips it, and neither upstream EngineCore.add_request nor the base
        scheduler re-checks prompt length. Admitted untouched, the prompt
        either can never be scheduled (chunked prefill is disabled and the
        prompt exceeds the token budget: parked in WAITING forever,
        head-of-line blocking every later request), overflows the worker's
        max_model_len-wide token buffer, or — even when it fits
        max_model_len — trips the adapter's own canvas-capacity validation,
        which raises out of execute_model in eager (no-upfront-capture) mode
        where there is no graceful stop-canvas rejection. Truncate to the
        largest tile-aligned prompt that still fits one whole canvas below
        the tile-floored max_model_len, so every admitted request is
        genuinely servable and finishes through the normal notifying path.
        Prefix caching is disabled for block models (__init__ asserts it), so
        the request's stale block hashes stay inert.
        """
        from vllm_tt_plugin.platform import _TT_TOKEN_TILE_SIZE

        if request.mm_features:
            # A text-only block model has a zero encoder budget: a feature at
            # offset 0 forces zero-token schedules forever (head-of-line
            # stall), and an interior offset carves a partial prefill chunk
            # that flips the step onto host sampling and kills the engine.
            # Dropped, the placeholder positions decode as ordinary tokens.
            logger.warning(
                "Request %s bypassed frontend validation with multimodal "
                "features a block-output model cannot encode; dropping them",
                request.request_id,
            )
            request.mm_features = []
        if request.prompt_token_ids is None and request.num_prompt_tokens > 0:
            # The frontend rejects prompt_embeds for every TT model; admitted
            # bare, the worker's request-state builder raises
            # NotImplementedError out of execute_model. Replace with
            # placeholder tokens; an embeds-only Request already carries
            # [0] * num_prompt_tokens in _all_token_ids, so the replacement
            # keeps every derived view consistent.
            logger.warning(
                "Request %s bypassed frontend validation with a "
                "prompt_embeds-only prompt the TT backend does not support; "
                "replacing with %d placeholder tokens",
                request.request_id,
                request.num_prompt_tokens,
            )
            request.prompt_token_ids = [0] * request.num_prompt_tokens
            request.prompt_embeds = None
        if request.prompt_embeds is not None:
            # Mixed-mode prompts (token ids + embeds + prompt_is_token_ids
            # mask) carry placeholder ids at the embed positions, so the TT
            # model decodes them as ordinary tokens either way; scrub the
            # embeds so the prompt_len x hidden_size tensor is not pinned for
            # the request lifetime, and warn like the embeds-only branch.
            logger.warning(
                "Request %s bypassed frontend validation with mixed "
                "token/embeds prompt content the TT backend does not "
                "support; dropping the embeds (placeholder ids decode as "
                "ordinary tokens)",
                request.request_id,
            )
            request.prompt_embeds = None
            request.prompt_is_token_ids = None
        if request.num_prompt_tokens == 0:
            # The frontend rejects empty prompts; admitted bare, the waiting
            # loop schedules zero new tokens and upstream's num_new_tokens
            # assert tears down the engine. Pad to one placeholder token so
            # the request schedules and finishes through the normal path.
            logger.warning(
                "Request %s bypassed frontend validation with an empty "
                "prompt; padding to one placeholder token",
                request.request_id,
            )
            request.prompt_token_ids = [0]
            request.prompt_embeds = None
            request._all_token_ids.append(0)
            request.num_prompt_tokens = 1
            return
        tile = _TT_TOKEN_TILE_SIZE
        max_model_len = int(self.vllm_config.model_config.max_model_len)
        aligned_max_model_len = max_model_len // tile * tile
        keep = (aligned_max_model_len - self._output_tokens_per_step) // tile * tile
        if request.num_prompt_tokens <= keep:
            return
        logger.warning(
            "Request %s bypassed frontend validation with a %d-token prompt "
            "that leaves no room for a whole %d-token canvas within "
            "max_model_len; truncating to %d tokens so the request can "
            "finish length-capped",
            request.request_id,
            request.num_prompt_tokens,
            self._output_tokens_per_step,
            keep,
        )
        if request.prompt_token_ids is not None:
            request.prompt_token_ids = request.prompt_token_ids[:keep]
        if getattr(request, "prompt_embeds", None) is not None:
            request.prompt_embeds = request.prompt_embeds[:keep]
        del request._all_token_ids[keep:]
        request.num_prompt_tokens = keep

    def _align_block_output_max_tokens(self, request: Request) -> None:
        """Clamp max_tokens so a bypassed EngineCoreRequest cannot overshoot.

        Frontend validation rejects an oversized limit. Prebuilt requests skip
        that path; the last canvas would then be applied past max_model_len and
        kill the engine. Shrink the logical cap to the largest whole-canvas
        budget that still fits. Lane mode reaches this through each lane.
        """
        from vllm_tt_plugin.platform import _fit_block_output_max_tokens

        prompt_ids = request.prompt_token_ids
        if prompt_ids is None or request.sampling_params is None:
            return
        fitted = _fit_block_output_max_tokens(
            len(prompt_ids),
            request.max_tokens,
            self._output_tokens_per_step,
            int(self.vllm_config.model_config.max_model_len),
        )
        if fitted == request.max_tokens:
            return
        # The prompt truncation above guarantees at least one whole canvas
        # fits, so a clamp always leaves a positive, servable budget.
        logger.debug(
            "Clamping block-output max_tokens from %s to %s for request %s "
            "so physical canvases fit max_model_len",
            request.max_tokens,
            fitted,
            request.request_id,
        )
        request.max_tokens = fitted
        request.sampling_params.max_tokens = fitted

    def _neutralize_block_output_host_sampling(self, request: Request) -> None:
        """Strip controls a block-output model cannot honor from a bypassed
        request: host-sampling forcers, structured outputs, and resumable
        streaming-input sessions.

        Frontend validation rejects these; a prebuilt EngineCoreRequest skips
        it, and any of them flips the step to host sampling mid-flight (which
        cannot construct a multi-token canvas) or parks the request forever.
        Raising here is no safer: an add_request exception also tears down
        EngineCore, so neutralize instead, the way the sampling controls are
        neutralized at the frontend.
        """
        params = request.sampling_params
        if params is None:
            return
        stripped = []
        if request.structured_output_request is not None:
            request.structured_output_request = None
            stripped.append("structured_outputs")
            # Request.__init__ parked the request on grammar compilation; with
            # the structured-output request gone nothing would ever promote it
            # out of skipped_waiting, so restore schedulability.
            if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
                request.status = RequestStatus.WAITING
        if params.structured_outputs is not None:
            params.structured_outputs = None
        if request.resumable:
            # A resumable session parks the stopped request to wait for more
            # input instead of finishing it, permanently leaking the
            # model-owned state slot. The frontend rejects it for block models.
            request.resumable = False
            stripped.append("resumable")
        for field, neutral in self._BLOCK_HOST_ONLY_SAMPLING_NEUTRAL:
            if getattr(params, field, neutral) not in (neutral, [], {}):
                setattr(params, field, neutral)
                stripped.append(field.lstrip("_"))
        if stripped:
            logger.warning(
                "Request %s bypassed frontend validation; stripped "
                "controls unsupported by block-output models: %s",
                request.request_id,
                ", ".join(stripped),
            )

    def _has_pending_prefill(self) -> bool:
        """Whether any request still needs prefill work.

        A request in ``skipped_waiting`` still needs a future prefill pass:
        that is where the base scheduler retries promotion after its dependency
        becomes ready. In decode-only mode this check also ensures both waiting
        queues are hidden from the base scheduler.

        A running ``is_prefill_chunk`` request is a partial prefill whose next
        chunk can only be scheduled by a prefill step.
        """
        return (
            bool(self.waiting)
            or bool(getattr(self, "skipped_waiting", False))
            or any(request.is_prefill_chunk for request in self.running)
        )

    # vLLM runs ``_mamba_block_aligned_split`` on every prefill chunk when
    # ``need_mamba_block_aligned_split`` is set; the TT scheduler reuses that
    # hook so chunk ends can be aligned before any block is allocated.
    @property
    def need_mamba_block_aligned_split(self) -> bool:
        return _history_reservation.enabled() or bool(getattr(self, "_base_need_mamba_split", False)) or (
            getattr(self, "_prefill_chunk_alignment", 0) > 1
        )

    @need_mamba_block_aligned_split.setter
    def need_mamba_block_aligned_split(self, value: bool) -> None:
        self._base_need_mamba_split = bool(value)

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        if getattr(self, "_base_need_mamba_split", False):
            num_new_tokens = super()._mamba_block_aligned_split(
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        count = self._align_prefill_chunk_end(request, start, num_new_tokens)
        reservation = _history_reservation.context()
        if reservation is not None:
            if num_external_computed_tokens:
                raise RuntimeError("External KV unsupported by reservation")
            return reservation.consider(request, start, count, self)
        return count

    def _align_prefill_chunk_end(
        self, request: Request, start: int, num_new_tokens: int
    ) -> int:
        """Round a *partial* prefill chunk down so it ends on the alignment grid.

        Chunks that finish the prompt, decode steps and multimodal prompts are
        left alone. A new request whose aligned chunk would be empty waits for
        the next step; a running request never gets an empty chunk.
        """
        align = getattr(self, "_prefill_chunk_alignment", 0)
        if align <= 1 or num_new_tokens <= 0:
            return num_new_tokens
        end = start + num_new_tokens
        if end >= request.num_prompt_tokens:
            return num_new_tokens
        if getattr(request, "mm_features", None) or getattr(
            request, "has_encoder_inputs", False
        ):
            return num_new_tokens
        aligned_end = end // align * align
        if aligned_end <= start:
            if start != 0:
                return num_new_tokens
            # A new request waits only when this step's leftover budget is what
            # stopped it. If a full step could not reach the grid either (a
            # permanent cap below the alignment), it must still make progress.
            return (
                0 if num_new_tokens < self._full_step_prefill_cap() else num_new_tokens
            )
        return aligned_end - start

    def _full_step_prefill_cap(self) -> int:
        """Most prompt tokens one scheduling step can give a single new request."""
        cfg = getattr(self, "scheduler_config", None)
        if cfg is None:
            return sys.maxsize
        limit, _ = _effective_prefill_step_limit(cfg)
        return limit

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        # NOTE: `throttle_prefills` accepted for interface compatibility with the base
        #        scheduler but unused - TT separates prefill/decode explicitly.
        reservation = _history_reservation.context()
        if reservation is None:
            return self._history_original_schedule(throttle_prefills)
        try:
            reservation.begin(self)
            output = self._history_original_schedule(throttle_prefills)
            reservation.selected_output(self, output)
            return output
        except BaseException:
            reservation.fail()
            raise

    def _history_original_schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        # NOTE: `throttle_prefills` accepted for interface compatibility with the base
        #        scheduler but unused - TT separates prefill/decode explicitly.
        has_pending_prefill = self._has_pending_prefill()
        # A partial prefill occupies ``running`` but cannot produce a decode
        # token, so it must not make the decode fallback below look viable.
        has_running_decode = any(
            not request.is_prefill_chunk for request in self.running
        )
        mode = self._forced_mode

        if mode == TTSchedulingMode.PREFILL_ONLY:
            # Forced mode is shared by every lane. Return an empty prefill
            # result unchanged so the coordinator can decide whether all lanes
            # should fall back to decode together.
            result = self._schedule_prefill_only()
            return self._finalize_scheduler_output(result, is_decode=False)
        if mode == TTSchedulingMode.DECODE_ONLY:
            if has_pending_prefill:
                # Hide the waiting queues and partial prefills so the base
                # scheduler cannot admit prefill work.
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result, is_decode=True)
            # No pending prefill: base scheduler naturally runs decode-only.
            result = super().schedule()
            return self._finalize_scheduler_output(result, is_decode=True)

        # Default mode:
        # Prefer prefill whenever prefill work is pending, so new requests are
        # admitted and partial prefills advance.
        if has_pending_prefill:
            if self._decode_interleave.wants_decode_step(
                has_pending_prefill=True, has_running_decode=has_running_decode
            ):
                # A run of prefill steps has reached its bound. Spend this step
                # on the running decodes so their inter-token latency does not
                # scale with the pending prompt's length; the pending prefill
                # resumes on the next step.
                self._decode_interleave.record_step(
                    is_decode=True, prefill_pending=True
                )
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result, is_decode=True)
            prefill_result = self._schedule_prefill_only()
            # If prefill cannot make progress (e.g. KV pressure), do not stall
            # decode. Fall back to decode-only so running requests can advance
            # and free capacity for a later prefill admission.
            if prefill_result.total_num_scheduled_tokens == 0 and has_running_decode:
                self._decode_interleave.record_step(
                    is_decode=True, prefill_pending=True
                )
                result = self._schedule_decode_only()
                # Even an empty prefill pass drains upstream cleanup events.
                # The runner must receive them with the replacement decode.
                result.finished_req_ids |= prefill_result.finished_req_ids
                result.free_encoder_mm_hashes = (
                    prefill_result.free_encoder_mm_hashes
                    + result.free_encoder_mm_hashes
                )
                if prefill_result.preempted_req_ids:
                    result.preempted_req_ids = (
                        result.preempted_req_ids or set()
                    ) | prefill_result.preempted_req_ids
                return self._finalize_scheduler_output(result, is_decode=True)
            self._decode_interleave.record_step(is_decode=False, prefill_pending=True)
            return self._finalize_scheduler_output(prefill_result, is_decode=False)

        # No pending prefill work in default mode: run decode-only naturally.
        self._decode_interleave.record_step(is_decode=True, prefill_pending=False)
        result = super().schedule()
        return self._finalize_scheduler_output(result, is_decode=True)

    def _finalize_scheduler_output(
        self, scheduler_output: SchedulerOutput, *, is_decode: bool
    ) -> SchedulerOutput:
        if is_decode:
            rows = len(scheduler_output.num_scheduled_tokens)
            if rows > getattr(self, "_widest_decode_batch_size", 0):
                self._widest_decode_batch_size = rows
                logger.info(
                    "TT scheduler: widest decode batch reached %d request row(s)",
                    rows,
                )
        pending_reset_discards = getattr(
            self, "_pending_forced_reset_discard_counts", {}
        )
        if pending_reset_discards:
            set_tt_forced_reset_discard_counts(scheduler_output, pending_reset_discards)
            self._pending_forced_reset_discard_counts = {}
        return scheduler_output

    def _schedule_prefill_only(self) -> SchedulerOutput:
        """Schedule prefill work: waiting requests and partial continuations.

        Temporarily hides the running *decode* requests so the base scheduler's
        running loop only advances partial prefills, and its waiting loop
        admits new ones.  Adjusts max_num_running_reqs so the waiting loop
        respects the true capacity with the decodes hidden.
        """
        pure_decodes = [r for r in self.running if not r.is_prefill_chunk]
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]

        saved_max = self.max_num_running_reqs
        self.running = cast(list[Request], partial_prefills)
        self.max_num_running_reqs = max(0, saved_max - len(pure_decodes))
        try:
            result = super().schedule()
        finally:
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max
        return result

    def _schedule_decode_only(self) -> SchedulerOutput:
        """Schedule only running decode requests.

        Temporarily hides both the ``waiting`` and ``skipped_waiting`` queues
        so the base scheduler's waiting loop is a no-op and cannot promote a
        grammar-ready structured-output request into this decode step, and
        hides partial prefills so their next chunk is not scheduled into it
        either.  Any requests that get preempted during decode scheduling are
        merged back into the original queues afterwards.
        """
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]

        saved_waiting = self.waiting
        saved_skipped = getattr(self, "skipped_waiting", None)
        self.waiting = create_request_queue(self.policy)
        if saved_skipped is not None:
            self.skipped_waiting = create_request_queue(self.policy)
        if partial_prefills:
            self.running = [r for r in self.running if not r.is_prefill_chunk]
        try:
            result = super().schedule()
        finally:
            if self.waiting:
                saved_waiting.prepend_requests(self.waiting)
            if saved_skipped is not None:
                if self.skipped_waiting:
                    saved_skipped.prepend_requests(self.skipped_waiting)
                self.skipped_waiting = saved_skipped
            self.waiting = saved_waiting
            if partial_prefills:
                self.running.extend(partial_prefills)
        return result

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Backstop against a stale-canvas resume vLLM's AR reset cannot represent.

        The engine layer (``_install_block_output_reset_abort_patch``) aborts
        running block requests and notifies their clients before delegating,
        so a reset requested through ``EngineCoreProc`` succeeds. Live block
        requests reach this guard only from engines that cannot notify the
        owning clients (a bare in-process ``EngineCore``) or callers that
        bypassed the engine layer; refusing here beats silently removing a
        request someone is still waiting on.
        """
        if self._is_block_output_model and reset_running_requests and self.running:
            message = (
                "Cannot reset prefix cache while a block-output request is "
                "running; finish or abort the request first."
            )
            # pause_generation(mode="keep") reaches this through an unguarded
            # idle-state callback while intentionally retaining live requests.
            # Returning False preserves them without letting an exception
            # escape EngineCore and strand the callback's Future.
            if self.pause_state == PauseState.PAUSED_ALL:
                logger.error("%s", message)
                return False
            raise RuntimeError(message)
        # Upstream preempts these requests in drop mode and discards their stale
        # output itself. TT receives one output frame for a speculative
        # reservation of 1+K tokens, so publish frame counts to the runner,
        # which must keep those frames out of request state before
        # resumed-prefill inputs are built.
        reset_candidates = (
            [
                (
                    request,
                    self._pending_async_output_frames.get(request.request_id, 0),
                )
                for request in self.running
                if request.num_output_placeholders > 0
            ]
            if reset_running_requests
            else []
        )
        try:
            return super().reset_prefix_cache(reset_running_requests, reset_connector)
        finally:
            for request, frame_count in reset_candidates:
                if request.status == RequestStatus.PREEMPTED and frame_count > 0:
                    pending = self._pending_forced_reset_discard_counts
                    pending[request.request_id] = (
                        pending.get(request.request_id, 0) + frame_count
                    )

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Reserve the physical output each scheduled step will commit.

        The platform removes the upstream diffusion marker, so vLLM reserves its
        normal one sampled-token placeholder. A plain block-output model commits
        a K-token canvas on every step, so reserve the remaining K-1 positions.
        An adaptive block-output model commits K only on a step that satisfies
        the ``block_step`` predicate below, and one token on every other step,
        so the extra reservation is per step. Each step's decision is recorded
        on its own SchedulerOutput because update_from_output must reconcile a
        commit against the decision that produced it.
        """
        super()._update_after_schedule(scheduler_output)
        output_frame_req_ids = tuple(
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if not self.requests[req_id].is_prefill_chunk
        )
        # The parent computes whether this chunk produces output. Retain that
        # decision on this step because a later schedule changes the request.
        setattr(scheduler_output, _TT_OUTPUT_FRAME_REQ_IDS_ATTR, output_frame_req_ids)
        pending = getattr(self, "_pending_async_output_frames", None)
        if pending is None:
            pending = self._pending_async_output_frames = {}
        for req_id in output_frame_req_ids:
            pending[req_id] = pending.get(req_id, 0) + 1
        if self.num_spec_tokens and not self.scheduler_config.async_scheduling:
            # ``AsyncScheduler`` leaves every scheduled request holding
            # ``[-1] * num_spec_tokens``, which upstream's GPU runner
            # overwrites from its own state in ``_prepare_input_ids``. On a
            # synchronous launch the TT runner has no such step: it verifies
            # whatever the scheduler delivers, so a placeholder surviving here
            # becomes a draft the accept walk compares against, matches (the
            # model is handed the same placeholder), and commits as an output
            # token.
            #
            # Cleared rather than restored to the ids just scheduled: a
            # proposal is handed over once, so a request whose row proposed
            # nothing this step must speculate on nothing next step rather
            # than replay a spent proposal. Drafts reach a request only
            # through ``update_draft_token_ids``.
            #
            # Left standing on an asynchronous launch, because there they are
            # the only lookahead reservation the request gets: upstream stops
            # routing drafts through the scheduler (``EngineCore.post_step``
            # skips ``take_draft_token_ids``), the next schedule budgets
            # ``1 + len(spec_token_ids)`` positions for the request, and
            # ``TTModelRunner._drafts_to_verify`` reads the placeholders as
            # that reservation and verifies the proposal the runner holds.
            for req_id in scheduler_output.num_scheduled_tokens:
                self.requests[req_id].spec_token_ids = []
        if not self._is_block_output_model:
            return
        extra_placeholders = (
            self._output_tokens_per_step - self.num_sampled_tokens_per_step
        )
        # Adaptive: the model emits its block ONLY on a SOLO DECODE step; every
        # other step (batched decodes, and the prefill that host-samples the
        # anchor) commits exactly one plain token and reserves one placeholder.
        # This MUST match the model's own gate, which runs the spec block in
        # decode_forward when batch == 1 and a plain token otherwise:
        #   - solo == batch 1 (exactly one request scheduled this step);
        #   - decode == the prompt was fully computed BEFORE this step (see the
        #     decode test below, which subtracts this step's scheduled tokens).
        # The decision is attached to THIS step's SchedulerOutput (not a Request
        # slot): under async the next step's schedule() overwrites Request state
        # before this step's output commits, but the engine core hands the
        # matching SchedulerOutput back to update_from_output, so the map always
        # pairs with the output being committed.
        solo = len(scheduler_output.num_scheduled_tokens) == 1
        if self._is_adaptive_block:
            self._mirror_spec_session(scheduler_output, solo)
        decisions: dict[str, bool] = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue
            if self._is_adaptive_block:
                # num_computed_tokens is already advanced by THIS step's
                # scheduled tokens here, so subtract them back out: a step is a
                # decode iff the prompt was fully computed BEFORE it. This is a
                # pure scheduling-side quantity -- output-commit timing (which
                # differs between sync tests and the pipelined engine loop)
                # cannot skew it. A resumed replay scheduling prompt+output
                # tokens lands back below the prompt boundary and correctly
                # stays a non-block step.
                scheduled = scheduler_output.num_scheduled_tokens[req_id]
                is_decode = (
                    request.num_computed_tokens - scheduled >= request.num_prompt_tokens
                )
                # Owning the model's single spec session is the whole gate:
                # the model serves a solo decode as plain baseline whenever it
                # has no session for THAT request, so reserving a block for a
                # non-owner would reserve a width the model cannot emit (see
                # _mirror_spec_session). The capture frontier is deliberately
                # NOT re-derived here -- it is a property of the PREFILL that
                # armed the session, already recorded in the ownership mirror,
                # and re-deriving it from the request's CURRENT length would
                # flip a live session's reservation to width 1 mid-generation
                # while the model keeps emitting blocks.
                block_step = solo and is_decode and self._spec_session_owner == req_id
            else:
                block_step = True
            if block_step:
                request.num_output_placeholders += extra_placeholders
            decisions[req_id] = block_step
        set_tt_block_step_decisions(scheduler_output, decisions)

    def _spec_frontier_ok(self, measured_len: int) -> bool:
        """Whether the model arms a session for a prefill of ``measured_len``.

        The model measures its capture frontier against the ``prompt_lens`` the
        runner hands it, which is ``input_positions + chunk_lens`` -- the tokens
        computed INCLUDING this step. On a replay resumed from preemption that
        spans the generated output too, so it is NOT ``num_prompt_tokens``:
        measuring the prompt alone let a resumed request cross the frontier on
        the model side only, which drops the session while this scheduler still
        reserved a block, and the width check then kills the engine core
        (vllm-tt-plugin#118.2). The caller passes ``num_computed_tokens``, which
        ``_update_after_schedule`` has already advanced by this step.
        """
        return (
            self._adaptive_block_max_prompt == 0
            or measured_len <= self._adaptive_block_max_prompt
        )

    def _mirror_spec_session(self, scheduler_output, solo: bool) -> None:
        """Track which request owns the adaptive model's SINGLE spec session.

        The model keeps one global session: drafter taps are captured during a
        request's own prefill, and only the request that owns them can emit a
        multi-token block. Every transition of that session is driven by the
        shape of a step, so the scheduler can mirror it exactly without asking
        the model -- and that is what keeps the width this scheduler RESERVES
        equal to the width the model EMITS:

        * a PREFILL step re-seats the session. The model captures taps only for
          a solo, spec-eligible prefill and drops the session on any other
          prefill (batched, or a prompt over the capture frontier).
        * a BATCHED decode step destroys it. The model releases the decoder and
          serves plain baseline from then on, and it never re-arms, because
          taps only ever come from a prefill. This is the case that made a
          benchmark sweep fail: as concurrency drains back to one request, that
          request is solo and eligible but no longer owns a session.
        * a SOLO decode by the OWNER leaves ownership alone -- it bootstraps
          its pending taps and keeps the session for the rest of its life.
        * a SOLO decode by ANY OTHER request destroys it. Async scheduling can
          skip the owner once it has reached max_tokens (upstream guards that
          skip on num_output_placeholders), leaving a different request alone
          on the next step while the session is still armed. The model releases
          the session on that step for the same reason -- it would otherwise
          speculate from the owner's residual taps -- and this mirrors that
          release, so the reserved width stays 1 for the non-owner
          (vllm-tt-plugin#118.1).

        A TT step is never mixed prefill+decode (docs/SCHEDULING.md), so the
        first scheduled request classifies the whole step.
        """
        owner = self._spec_session_owner
        # A finished or aborted owner no longer holds the session: the model
        # clears it in release_request, and the id is gone from self.requests.
        if owner is not None and owner not in self.requests:
            owner = None
        scheduled_tokens = scheduler_output.num_scheduled_tokens
        first_id = next(iter(scheduled_tokens), None)
        if first_id is not None:
            request = self.requests[first_id]
            step_is_decode = (
                request.num_computed_tokens - scheduled_tokens[first_id]
                >= request.num_prompt_tokens
            )
            if not step_is_decode:
                # num_computed_tokens is advanced by this step already, so it is
                # exactly the prompt_lens the model measures its frontier on.
                owner = (
                    first_id
                    if (solo and self._spec_frontier_ok(request.num_computed_tokens))
                    else None
                )
            elif not solo or first_id != owner:
                owner = None
        self._spec_session_owner = owner

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Bind this step's block-step decisions before the base loop commits
        its outputs, so ``_update_request_with_output`` reads the decision that
        produced THIS output rather than a later schedule's overwrite, and
        consume only the output frames this scheduled step reserved."""
        if self._is_block_output_model:
            self._committing_block_step_decisions = get_tt_block_step_decisions(
                scheduler_output
            )
        try:
            return super().update_from_output(scheduler_output, model_runner_output)
        finally:
            self._committing_block_step_decisions = {}
            pending = self._pending_async_output_frames
            for req_id in getattr(scheduler_output, _TT_OUTPUT_FRAME_REQ_IDS_ATTR, ()):
                count = pending.get(req_id, 0)
                if count <= 1:
                    pending.pop(req_id, None)
                else:
                    pending[req_id] = count - 1

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        """Commit one block and reconcile its full physical reservation."""
        if not self._is_block_output_model:
            return super()._update_request_with_output(
                request, new_token_ids, is_stale=is_stale
            )
        # Adaptive: a request that decoded batched (or over-frontier) this step
        # committed a single baseline token (no block was reserved) -> plain
        # reconciliation. The decision is read from THIS step's SchedulerOutput
        # (set in _update_after_schedule), which the engine core pairs with this
        # output even when async scheduling has already run a later schedule().
        if self._is_adaptive_block:
            block_step = self._committing_block_step_decisions.get(request.request_id)
            if block_step is None:
                raise RuntimeError(
                    "adaptive block-output request committed output without a "
                    f"scheduling decision: req_id={request.request_id!r}"
                )
            if not block_step:
                # The scheduler stamped this step width 1, so the model owes
                # exactly one token. Check before delegating: super() appends
                # whatever it is handed, so a model that returned a BLOCK here
                # would commit extra tokens against a single reserved
                # placeholder and the mismatch would surface later as a
                # placeholder leak or a corrupted continuation, far from its
                # cause. This is the mirror of the block-width check below.
                if len(new_token_ids) != self.num_sampled_tokens_per_step:
                    raise ValueError(
                        "Model output width violates the scheduled baseline "
                        f"width: got {len(new_token_ids)}, expected "
                        f"{self.num_sampled_tokens_per_step} "
                        f"(req_id={request.request_id!r}); the scheduler "
                        "stamped this step width 1 (batched, or a prompt over "
                        "the spec frontier), so the model must return a single "
                        "baseline token -- the scheduler and model block gates "
                        "disagree"
                    )
                return super()._update_request_with_output(
                    request, new_token_ids, is_stale=is_stale
                )
        if is_stale:
            # The request was preempted after this block was submitted, and
            # preemption zeroed the K placeholders the block would reconcile.
            # Block requests are never reset-preempted (reset_prefix_cache
            # raises while one runs) and solo spec steps are not KV-preempted,
            # so this must not happen -- fail loudly rather than commit a block
            # against a reservation that no longer exists.
            raise RuntimeError(
                "A stale async output reached block serving for a block step; "
                "block-output frames cannot be discarded (reset/preempt of a "
                "running block request is unsupported)"
            )
        if len(new_token_ids) != self._output_tokens_per_step:
            raise ValueError(
                "Model output width violates output_tokens_per_step: "
                f"{len(new_token_ids)} != {self._output_tokens_per_step} "
                f"(req_id={request.request_id!r}); the scheduler reserved a "
                "block but the model returned a different width -- the "
                "scheduler and model block gates disagree"
            )

        # Scheduler appends token-by-token and trims at EOS, stop tokens,
        # max_tokens, or max_model_len. The reservation is physical, so consume
        # all K placeholders even when the client-visible block is trimmed.
        # Calling Scheduler directly intentionally skips AsyncScheduler's
        # cache_blocks hook. Platform validation disables prefix caching for
        # block-output models, and __init__ asserts that invariant.
        new_token_ids, stopped = Scheduler._update_request_with_output(
            self, request, new_token_ids
        )
        request.num_output_placeholders -= self._output_tokens_per_step
        if request.num_output_placeholders < 0:
            raise RuntimeError(
                "Output placeholders underflowed after block-output reconciliation"
            )
        return new_token_ids, stopped
