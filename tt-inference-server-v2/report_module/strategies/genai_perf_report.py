# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

import logging
from glob import glob
from pathlib import Path
from typing import Any, Dict, List


from report_module.base_strategy import ReportStrategy
from report_module.parsing.common import deduplicate_by_config, extract_percentile_result
from report_module.strategies.aiperf_report import METRIC_DEFINITIONS, _percentile_table_markdown
from report_module.types import ReportContext, ReportResult

logger = logging.getLogger(__name__)


class GenAiPerfStrategy(ReportStrategy):
    """GenAI-Perf detailed benchmark report — fully self-contained."""

    name = "benchmarks_genai_perf"

    def is_applicable(self, context: ReportContext) -> bool:
        benchmarks_dir = f"{context.workflow_log_dir}/benchmarks_output"
        model_id = context.model_spec.model_id
        return len(glob(f"{benchmarks_dir}/genai_benchmark_{model_id}_*.json")) > 0

    def generate(self, context: ReportContext) -> Dict[str, ReportResult]:
        benchmarks_dir = f"{context.workflow_log_dir}/benchmarks_output"
        model_id = context.model_spec.model_id

        genai_files = deduplicate_by_config(glob(f"{benchmarks_dir}/genai_benchmark_{model_id}_*.json"))
        logger.info(f"GenAI-Perf: {len(genai_files)} files after dedup")

        if not genai_files:
            return {self.name: ReportResult.empty(self.name)}

        genai_text_files = [f for f in genai_files if "images" not in Path(f).name]
        genai_vlm_files = [f for f in genai_files if "images" in Path(f).name]

        text_results = self._parse_files(genai_text_files, "genai-perf")
        vlm_results = self._parse_files(genai_vlm_files, "genai-perf", is_vlm=True)

        if not text_results and not vlm_results:
            return {self.name: ReportResult.empty(self.name)}

        text_results.sort(key=lambda x: (x["isl"], x["osl"], x["concurrency"]))
        vlm_results.sort(key=lambda x: (x["isl"], x["osl"], x["concurrency"], x.get("images_per_prompt", 0), x.get("image_height", 0), x.get("image_width", 0)))

        release_str = f"### GenAI-Perf Benchmark Performance Results for {context.model_name} on {context.device_str}\n\n"

        if text_results:
            release_str += "#### GenAI-Perf Text Benchmarks - Detailed Percentiles\n\n"
            release_str += "**Benchmarking Tool:** [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer)\n\n"
            release_str += _percentile_table_markdown(text_results)
            release_str += "\n*Note: GenAI-Perf does not natively support total token throughput metrics.*\n\n"

        if vlm_results:
            release_str += "#### GenAI-Perf VLM Benchmarks - Detailed Percentiles\n\n"
            release_str += "**Benchmarking Tool:** [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer)\n\n"
            release_str += _percentile_table_markdown(vlm_results, is_vlm=True)
            release_str += "\n*Note: GenAI-Perf does not natively support total token throughput metrics.*\n\n"

        release_str += METRIC_DEFINITIONS

        all_results = text_results + vlm_results
        result = ReportResult(
            name=self.name,
            markdown=release_str,
            data=all_results,
            md_filename=f"genai_perf_benchmark_display_{context.report_id}.md",
        )
        return {self.name: result}

    @staticmethod
    def _parse_files(files: List[str], source: str, is_vlm: bool = False) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for f in sorted(files):
            result = extract_percentile_result(f, source, is_vlm=is_vlm)
            if result:
                results.append(result)
        return results
