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
        self.writer = threading.Thread(
            target=self._write_records, name="gemma-phase-record-writer", daemon=True
        )
        self.writer.start()

    def _write_records(self):
        while True:
            record = self.records.get()
            try:
                sys.stdout.write("GEMMA_PHASE_RECORD " + json.dumps(record, separators=(",", ":")) + "\n")
                sys.stdout.flush()
            except Exception:
                # A broken log destination must never change model responses.
                pass

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        if iteration_stats is None or not iteration_stats.finished_requests:
            return
        callback_wall = time.time()
        callback_mono = time.monotonic()
        scheduler = scheduler_stats
        for req in iteration_stats.finished_requests:
            finish_reason = getattr(req.finish_reason, "name", str(req.finish_reason)).lower()
            self.records.put({
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
            })

    def log_engine_initialized(self):
        pass
