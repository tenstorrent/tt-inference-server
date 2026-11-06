# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Parser for vLLM's structured-output benchmark (``benchmark_serving_structured_output.py``).

Consumes the result JSON that
:class:`llm_module.drivers.vllm_structured_output.VLLMStructuredOutputDriver`
produces: the script's own result plus ``tt_structured_output`` (the output
checks of ``llm_module/structured_output_client.py``) and the driver's
``tt_*`` run fields. Each run becomes one ``structured_output`` Block; the
kind is its own because the runs have no ISL and no perf targets.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from report_module.schema import Block

from .base import LLMResultParser, round_metric

STRUCTURED_OUTPUT_KIND = "structured_output"


def _pct(rate: Any) -> Optional[float]:
    return round(100 * rate, 2) if isinstance(rate, (int, float)) else None


class VLLMStructuredOutputParser(LLMResultParser):
    kind = STRUCTURED_OUTPUT_KIND
    tool = "vllm_structured_output"
    tool_label = "vLLM Structured Output"

    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        checks = raw.get("tt_structured_output") or {}
        dataset = raw.get("tt_dataset")
        ratio = raw.get("tt_structured_output_ratio")
        record: Dict[str, Any] = {
            "tool": self.tool,
            "model": str(raw.get("model_id", "") or ""),
            "device": device,
            "dataset": dataset,
            "structured_output_ratio": ratio,
            "structured_requests": checks.get("structured_requests"),
            "chat_template_kwargs": checks.get("chat_template_kwargs"),
            "concurrency": raw.get("max_concurrency"),
            "num_requests": raw.get("completed"),
            "output_sequence_length": raw.get("tt_osl"),
            "mean_ttft_ms": round_metric(raw.get("mean_ttft_ms"), 4),
            "p99_ttft": round_metric(raw.get("p99_ttft_ms"), 4),
            "mean_tpot_ms": round_metric(raw.get("mean_tpot_ms"), 4),
            "p99_tpot": round_metric(raw.get("p99_tpot_ms"), 4),
            "mean_itl_ms": round_metric(raw.get("mean_itl_ms"), 4),
            "mean_e2el_ms": round_metric(raw.get("mean_e2el_ms"), 4),
            "tps_output_throughput": round_metric(raw.get("output_throughput"), 4),
            "request_throughput": round_metric(raw.get("request_throughput"), 4),
            "correct_rate_pct": round_metric(raw.get("correct_rate(%)"), 2),
            "grammar_active_pct": _pct(checks.get("grammar_active_rate")),
            "complete_json_pct": _pct(checks.get("complete_json_rate")),
            "schema_valid_pct": _pct(checks.get("schema_valid_rate")),
        }
        title = f"{self.tool_label}: {dataset}, ratio {ratio}"
        return self._wrap_record(record, title=title)


__all__ = ["STRUCTURED_OUTPUT_KIND", "VLLMStructuredOutputParser"]
