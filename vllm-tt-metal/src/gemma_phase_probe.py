"""Proof-only vLLM 0.29 request phase records for Gemma agentic timing.

One JSON record is queued per completed request. A daemon thread writes those
records to server stdout, which the existing CI docker log already captures.
There is no per-token logging and no disk I/O in the model scheduling path.
"""

import json
import queue
import sys
import threading
import time

from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats


_original_update = IterationStats.update_from_finished_request


def _update_with_event_times(self, *args, **kwargs):
    """Carry existing vLLM event times into its completed-request record."""
    previous_count = len(self.finished_requests)
    result = _original_update(self, *args, **kwargs)
    if len(self.finished_requests) > previous_count:
        req_stats = kwargs.get("req_stats")
        if req_stats is None and len(args) > 4:
            req_stats = args[4]
        if req_stats is not None:
            finished = self.finished_requests[-1]
            for name in (
                "arrival_time", "queued_ts", "scheduled_ts",
                "first_token_ts", "last_token_ts",
            ):
                setattr(finished, name, getattr(req_stats, name, None))
    return result


if not getattr(IterationStats.update_from_finished_request, "_gemma_phase_wrapped", False):
    _update_with_event_times._gemma_phase_wrapped = True
    IterationStats.update_from_finished_request = _update_with_event_times


class GemmaPhaseLogger(StatLoggerBase):
    """vLLM stat logger plugin: one buffered line per finished request."""

    def __init__(self, vllm_config, engine_index=0):
        self.engine_index = engine_index
        self.records = queue.SimpleQueue()
        self.step_bucket = None
        self.step_counts = None
        self.writer = threading.Thread(
            target=self._write_records, name="gemma-phase-record-writer", daemon=True
        )
        self.writer.start()

    def _write_records(self):
        while True:
            prefix, record = self.records.get()
            try:
                sys.stdout.write(prefix + " " + json.dumps(record, separators=(",", ":")) + "\n")
                sys.stdout.flush()
            except Exception:
                # A broken log destination must never change model responses.
                pass

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        self._record_step(scheduler_stats)
        if iteration_stats is None or not iteration_stats.finished_requests:
            return
        callback_wall = time.time()
        callback_mono = time.monotonic()
        scheduler = scheduler_stats
        for req in iteration_stats.finished_requests:
            finish_reason = getattr(req.finish_reason, "name", str(req.finish_reason)).lower()
            self.records.put(("GEMMA_PHASE_RECORD", {
                "schema": 1,
                "request_id": req.request_id,
                "finish_reason": finish_reason,
                "prompt_tokens": req.num_prompt_tokens,
                "generated_tokens": req.num_generation_tokens,
                "cached_tokens": req.num_cached_tokens,
                "max_tokens_param": req.max_tokens_param,
                "e2e_s": req.e2e_latency,
                "queue_s": req.queued_time,
                "prefill_s": req.prefill_time,
                "decode_s": req.decode_time,
                "inference_s": req.inference_time,
                "frontend_arrival_wall_s": getattr(req, "arrival_time", None),
                "queued_mono_s": getattr(req, "queued_ts", None),
                "scheduled_mono_s": getattr(req, "scheduled_ts", None),
                "first_token_mono_s": getattr(req, "first_token_ts", None),
                "last_token_mono_s": getattr(req, "last_token_ts", None),
                "callback_wall_s": callback_wall,
                "callback_mono_s": callback_mono,
                "running_at_completion_sample": (
                    getattr(scheduler, "num_running_reqs", None) if scheduler else None
                ),
                "waiting_at_completion_sample": (
                    getattr(scheduler, "num_waiting_reqs", None) if scheduler else None
                ),
                "skipped_waiting_at_completion_sample": (
                    getattr(scheduler, "num_skipped_waiting_reqs", None) if scheduler else None
                ),
                "kv_cache_usage_at_completion_sample": (
                    getattr(scheduler, "kv_cache_usage", None) if scheduler else None
                ),
                "engine_index": self.engine_index,
            }))

    def _record_step(self, scheduler):
        """One low-rate summary of existing iteration details per wall second.

        vLLM's elapsed_ms times the wait on the model future. It is not a
        device kernel timer; no per-step device synchronization is added.
        """
        detail = getattr(scheduler, "iteration_details", None) if scheduler else None
        if detail is None:
            return
        now = time.time()
        bucket = int(now)
        if self.step_bucket != bucket:
            if self.step_counts is not None:
                self.records.put(("GEMMA_STEP_SUMMARY", self.step_counts))
            self.step_bucket = bucket
            self.step_counts = {
                "schema": 1, "wall_second": bucket, "first_callback_wall_s": now,
                "last_callback_wall_s": now, "engine_index": self.engine_index,
                "steps": 0, "prefill_steps": 0, "decode_steps": 0,
                "dummy_steps": 0, "prefill_tokens": 0, "decode_tokens": 0,
                "future_wait_ms_prefill": 0.0, "future_wait_ms_decode": 0.0,
                "future_wait_ms_dummy": 0.0,
            }
        item = self.step_counts
        item["last_callback_wall_s"] = now
        item["steps"] += 1
        ctx_tokens = int(getattr(detail, "num_ctx_tokens", 0) or 0)
        gen_tokens = int(getattr(detail, "num_generation_tokens", 0) or 0)
        elapsed = float(getattr(detail, "elapsed_ms", 0.0) or 0.0)
        if ctx_tokens and gen_tokens:
            raise ValueError("mixed prefill/decode step in TT scheduler")
        if ctx_tokens:
            item["prefill_steps"] += 1
            item["prefill_tokens"] += ctx_tokens
            item["future_wait_ms_prefill"] += elapsed
        elif gen_tokens:
            item["decode_steps"] += 1
            item["decode_tokens"] += gen_tokens
            item["future_wait_ms_decode"] += elapsed
        else:
            item["dummy_steps"] += 1
            item["future_wait_ms_dummy"] += elapsed

    def log_engine_initialized(self):
        pass
