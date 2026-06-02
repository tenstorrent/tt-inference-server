# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import re
from typing import Dict

# Raw data key → display header. Canonical name per key (no per-task
# variations — pick one, the renderer applies it everywhere). Names
# without a unit suffix appear as-is; metrics with units include them in
# parens so the column header is self-describing.
DISPLAY_NAMES: Dict[str, str] = {
    # Sequence / concurrency
    "input_sequence_length": "ISL",
    "isl": "ISL",
    "output_sequence_length": "OSL",
    "osl": "OSL",
    "max_con": "Concurrency",
    "max_concurrency": "Concurrency",
    "concurrency": "Concurrency",
    "num_requests": "Num Requests",
    "num_prompts": "Num Prompts",
    "num_responses": "Num Responses",
    "num_concurrent_requests": "Num Concurrent Requests",
    # Latency
    "mean_ttft_ms": "TTFT (ms)",
    "std_ttft_ms": "TTFT Std (ms)",
    "ttft": "TTFT",
    "ttft_ms": "TTFT (ms)",
    "p5_ttft": "P5 TTFT (ms)",
    "p25_ttft": "P25 TTFT (ms)",
    "p50_ttft": "P50 TTFT (ms)",
    "p90_ttft": "P90 TTFT (ms)",
    "p95_ttft": "P95 TTFT (ms)",
    "p99_ttft": "P99 TTFT (ms)",
    "mean_tpot_ms": "TPOT (ms)",
    "std_tpot_ms": "TPOT Std (ms)",
    "tpot": "TPOT",
    "mean_itl_ms": "ITL (ms)",
    "itl": "ITL",
    "mean_e2el_ms": "E2EL (ms)",
    "e2el": "E2EL",
    # Throughput
    "mean_tps": "Tput User (TPS)",
    "std_tps": "Tput User Std (TPS)",
    "tput_user": "Tput User (TPS)",
    "tps_decode_throughput": "Tput Decode (TPS)",
    "tput_decode": "Tput Decode (TPS)",
    "tput": "Tput Decode (TPS)",
    "tps_prefill_throughput": "Tput Prefill (TPS)",
    "tput_prefill": "Tput Prefill (TPS)",
    "request_throughput": "Req Tput (RPS)",
    "req_tput": "Req Tput (RPS)",
    "total_token_throughput": "Total Token Throughput (tokens/s)",
    "total_input_tokens": "Total Input Tokens",
    "total_output_tokens": "Total Output Tokens",
    # Image-generation / CNN / video
    "num_inference_steps": "Inference Steps",
    "inference_steps_per_second": "Steps/Sec",
    "image_height": "Image Height",
    "image_width": "Image Width",
    "image_resolution": "Image Resolution",
    "images_per_prompt": "Images per Prompt",
    "images": "Images",
    "requests_duration": "Request Duration (s)",
    "average_duration": "Avg Duration (s)",
    "max_duration": "Max Duration (s)",
    "avg_duration": "Avg Duration (s)",
    "target_time": "Target (s)",
    "within_target": "Within Target",
    "total_requests": "Total Requests",
    "batch_size": "Batch Size",
    # Image-eval scores (image_generation_evals + image_eval)
    "fid_score": "FID",
    "average_clip": "CLIP Score",
    "deviation_clip_score": "CLIP Std",
    "same_seed_ssim": "Same-Seed SSIM",
    "diff_params_ssim": "Diff-Params SSIM",
    "same_requests_match": "Same-Request Match",
    "diff_params_differs": "Diff-Params Differs",
    "lora_differentiation": "LoRA Differentiation",
    "lora_configs": "LoRA Configs",
    # Audio / TTS
    "wer": "WER",
    "rtr": "RTR",
    "t/s/u": "T/S/U",
    "audio_duration": "Audio Duration (s)",
    "streaming_enabled": "Streaming Enabled",
    "preprocessing_enabled": "Preprocessing Enabled",
    "reference_text": "Reference Text",
    # Embedding
    "embedding_dimension": "Embedding Dimension",
    # Eval (orchestrator)
    "task_name": "Task",
    "tolerance": "Tolerance",
    "published_score": "Published Score",
    "published_score_ref": "Published Score Ref",
    "score": "Score",
    "accuracy_check": "Accuracy Check",
    # Liveness / infra
    "status": "Status",
    "expected_devices": "Expected Devices",
    "ready_workers": "Ready Workers",
    "alive_workers": "Alive Workers",
    "ready_count": "Ready Count",
    "alive_count": "Alive Count",
    "model_ready": "Model Ready",
    "runner_in_use": "Runner",
    "child_result": "Child Result",
    # Common envelope
    "name": "Name",
    "success": "Success",
    "attempts": "Attempts",
    "backend": "Source",
    "task_type": "Task Type",
    "dataset": "Dataset",
    "structured_output_ratio": "SO Ratio",
    "correct_rate_pct": "Correct Rate (%)",
}

# Raw key → digits after the decimal point. Missing keys use
# :data:`report_module.formatting._format_float`'s significant-digits
# default. Pull these straight from v1's sig_digits_map and
# decimal_places_map;
DECIMAL_PLACES: Dict[str, int] = {
    "mean_ttft_ms": 1,
    "ttft_ms": 1,
    "std_ttft_ms": 1,
    "p5_ttft": 1,
    "p25_ttft": 1,
    "p50_ttft": 1,
    "p90_ttft": 1,
    "p95_ttft": 1,
    "p99_ttft": 1,
    "mean_tpot_ms": 1,
    "std_tpot_ms": 1,
    "mean_itl_ms": 1,
    "mean_e2el_ms": 1,
    "mean_tps": 2,
    "std_tps": 2,
    "tput_user": 2,
    "tps_decode_throughput": 1,
    "tps_prefill_throughput": 1,
    "request_throughput": 3,
    "total_token_throughput": 2,
    "requests_duration": 2,
    "average_duration": 2,
    "max_duration": 2,
    "avg_duration": 2,
    "target_time": 2,
    "inference_steps_per_second": 1,
    "elapsed_seconds": 2,
    "fid_score": 2,
    "average_clip": 4,
    "deviation_clip_score": 4,
    "same_seed_ssim": 4,
    "diff_params_ssim": 4,
    "tput_decode": 1,
    "tput_prefill": 1,
    "rtr": 3,
    "wer": 4,
    "audio_duration": 2,
}

# Footnote glossary. Looked up by raw key. Only add entries for terms
# that aren't already obvious from the display name itself.
EXPLANATIONS: Dict[str, str] = {
    "input_sequence_length": "Input Sequence Length (tokens)",
    "output_sequence_length": "Output Sequence Length (tokens)",
    "max_con": "number of concurrent requests (batch size)",
    "num_requests": "total number of requests (sample size, N)",
    "mean_ttft_ms": "Time To First Token (ms)",
    "mean_tpot_ms": "Time Per Output Token (ms)",
    "mean_tps": "Throughput per user (TPS)",
    "tps_decode_throughput": "Throughput for decode tokens, across all users (TPS)",
    "tps_prefill_throughput": "Throughput for prefill tokens (TPS)",
    "mean_e2el_ms": "End-to-End Latency (ms)",
    "request_throughput": "Request Throughput (RPS)",
}


def display_name(raw_key: str) -> str:
    """Translate a raw data key to its display header, or fall back."""
    return DISPLAY_NAMES.get(raw_key, raw_key)


def decimal_places(raw_key: str) -> int | None:
    """Per-key float precision, or None when the default formatter applies."""
    return DECIMAL_PLACES.get(raw_key)


_UNIT_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")
_CHECK_SUFFIX = "_check"
_RATIO_SUFFIX = "_ratio"


def _base_display(raw_key: str) -> str:
    return _UNIT_SUFFIX.sub("", display_name(raw_key))


def target_checks_header(col: str) -> str:
    if col == "name":
        return "Tier"
    if col.endswith(_CHECK_SUFFIX):
        return f"{_base_display(col[: -len(_CHECK_SUFFIX)])} Check"
    if col.endswith(_RATIO_SUFFIX):
        return f"{_base_display(col[: -len(_RATIO_SUFFIX)])} Ratio"
    return f"{_base_display(col)} Target"


__all__ = [
    "DISPLAY_NAMES",
    "DECIMAL_PLACES",
    "EXPLANATIONS",
    "display_name",
    "decimal_places",
    "target_checks_header",
]
