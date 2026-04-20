# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

import json
import logging
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmarking.benchmark_config import cap_benchmark_params
from evals.eval_config import EVAL_CONFIGS
from report_module.base_strategy import ReportStrategy
from report_module.markdown.report_renderers import tiered_targets_markdown
from report_module.markdown.visualizer import MarkdownVisualizer
from report_module.parsing.benchmark_parser import process_benchmark_files
from report_module.parsing.benchmark_summary import build_summary_row
from report_module.parsing.common import deduplicate_by_config
from report_module.parsing.target_checks import (
    compute_text_target_checks,
    compute_vlm_target_checks,
)
from report_module.types import ReportContext, ReportResult
from workflows.model_spec import ModelSpec
from workflows.utils import (
    is_preprocessing_enabled_for_whisper,
    is_streaming_enabled_for_whisper,
)
from workflows.workflow_types import ModelType, ReportCheckTypes

logger = logging.getLogger(__name__)

_EVAL_GLOB_PATTERN = {
    ModelType.AUDIO: "*_results.json",
}
_DEFAULT_EVAL_GLOB = "results_*.json"

SUB_SECTION_BENCHMARKS = "benchmarks"
SUB_SECTION_BENCHMARK_SUMMARY = "benchmarks_summary"
SUB_SECTION_EVALS = "evals"

_TEXT_VLM_MODEL_TYPES = {ModelType.LLM.name, ModelType.VLM.name}

_MODEL_TYPE_TO_ROW_TASK_TYPE = {
    ModelType.IMAGE.name: "image",
    ModelType.CNN.name: "cnn",
    ModelType.VIDEO.name: "video",
    ModelType.AUDIO.name: "audio",
    ModelType.TEXT_TO_SPEECH.name: "tts",
    ModelType.EMBEDDING.name: "embedding",
}

_MODEL_TYPE_TO_TASK_LABEL = {
    ModelType.IMAGE.name: "Image",
    ModelType.CNN.name: "CNN",
    ModelType.VIDEO.name: "Video",
    ModelType.AUDIO.name: "Audio",
    ModelType.TEXT_TO_SPEECH.name: "Text-to-Speech",
    ModelType.EMBEDDING.name: "Embedding",
}


class _WhisperSpecWrapper:

    __slots__ = ("model_spec",)

    def __init__(self, model_spec: ModelSpec) -> None:
        self.model_spec = model_spec


class StandardReportStrategy(ReportStrategy):

    SUB_SECTIONS = (
        SUB_SECTION_BENCHMARKS,
        SUB_SECTION_BENCHMARK_SUMMARY,
        SUB_SECTION_EVALS,
    )

    name = "standard"

    def is_applicable(self, context: ReportContext) -> bool:
        return True

    def generate(self, context: ReportContext) -> Dict[str, ReportResult]:
        # Evals run first so benchmarks_summary (for CNN/IMAGE/VIDEO) can
        # read ``tput_user`` out of the eval results.  Results are still
        # returned in release-section order: benchmarks, summary, evals.
        evals_results: Dict[str, ReportResult] = {}
        if self._section_requested(context, SUB_SECTION_EVALS):
            evals_results = self._generate_evals(context)

        evals_tput_user = self._extract_evals_tput_user(evals_results)

        results: Dict[str, ReportResult] = {}
        if self._section_requested(context, SUB_SECTION_BENCHMARKS):
            results.update(self._generate_benchmarks(context, evals_tput_user))

        results.update(evals_results)
        return results

    @staticmethod
    def _section_requested(context: ReportContext, section: str) -> bool:
        if not context.selected_sections:
            return True
        return section in context.selected_sections or "standard" in context.selected_sections

    @staticmethod
    def _extract_evals_tput_user(
        evals_results: Dict[str, ReportResult],
    ) -> Optional[float]:
        evals_result = evals_results.get(SUB_SECTION_EVALS)
        if not evals_result or not evals_result.data:
            return None
        first = evals_result.data[0]
        if not isinstance(first, dict):
            return None
        value = first.get("tput_user")
        return float(value) if isinstance(value, (int, float)) else None

    # ── Benchmark sweeps + targets ───────────────────────────────────────

    _TOOL_LABELS = ("vLLM", "AIPerf", "GenAI-Perf")

    def _generate_benchmarks(
        self,
        context: ReportContext,
        evals_tput_user: Optional[float],
    ) -> Dict[str, ReportResult]:
        rows_by_tool = self._collect_rows_by_tool(context)
        if not rows_by_tool:
            return {
                SUB_SECTION_BENCHMARKS: ReportResult.empty(SUB_SECTION_BENCHMARKS),
                SUB_SECTION_BENCHMARK_SUMMARY: ReportResult.empty(
                    SUB_SECTION_BENCHMARK_SUMMARY
                ),
            }

        self._inject_whisper_flags(rows_by_tool, context.model_spec)

        all_results: List[Dict[str, Any]] = []
        for rows in rows_by_tool.values():
            all_results.extend(rows)

        sweep_md = MarkdownVisualizer.build_benchmark_sweeps_markdown(
            rows_by_tool=rows_by_tool,
            model_name=context.model_name,
            device_str=context.device_str,
        )

        if self._is_text_vlm_model(context.model_spec):
            targets_md, summary_data = self._build_target_markdown(
                context, all_results
            )
        else:
            targets_md, summary_data = self._build_tiered_summary_markdown(
                context, rows_by_tool, evals_tput_user
            )

        return {
            SUB_SECTION_BENCHMARKS: ReportResult(
                name=SUB_SECTION_BENCHMARKS,
                markdown=sweep_md,
                data=all_results,
                display_markdown=sweep_md,
                md_filename=f"benchmark_display_{context.report_id}.md",
            ),
            SUB_SECTION_BENCHMARK_SUMMARY: ReportResult(
                name=SUB_SECTION_BENCHMARK_SUMMARY,
                markdown=targets_md,
                data=summary_data,
                md_filename=f"benchmark_summary_{context.report_id}.md",
            ),
        }

    @staticmethod
    def _is_text_vlm_model(model_spec: ModelSpec) -> bool:
        return model_spec.model_type.name in _TEXT_VLM_MODEL_TYPES

    def _collect_rows_by_tool(
        self, context: ReportContext
    ) -> Dict[str, List[Dict[str, Any]]]:
        benchmarks_dir = f"{context.workflow_log_dir}/benchmarks_output"
        model_id = context.model_spec.model_id

        file_patterns = {
            "vLLM": f"{benchmarks_dir}/benchmark_{model_id}_*.json",
            "AIPerf": f"{benchmarks_dir}/aiperf_benchmark_{model_id}_*.json",
            "GenAI-Perf": f"{benchmarks_dir}/genai_benchmark_{model_id}_*.json",
        }
        tool_files = {
            tool: deduplicate_by_config(glob(pattern))
            for tool, pattern in file_patterns.items()
        }

        logger.info(
            "Standard benchmarks: "
            + ", ".join(f"{len(files)} {tool}" for tool, files in tool_files.items())
        )

        rows_by_tool: Dict[str, List[Dict[str, Any]]] = {}
        for tool_label in self._TOOL_LABELS:
            files = tool_files[tool_label]
            if not files:
                continue
            rows_by_tool[tool_label] = process_benchmark_files(files)
        return rows_by_tool

    @staticmethod
    def _inject_whisper_flags(
        rows_by_tool: Dict[str, List[Dict[str, Any]]],
        model_spec: ModelSpec,
    ) -> None:
       
        if "whisper" not in model_spec.hf_model_repo.lower():
            return
        wrapper = _WhisperSpecWrapper(model_spec)
        streaming = str(is_streaming_enabled_for_whisper(wrapper))
        preprocessing = str(is_preprocessing_enabled_for_whisper(wrapper))
        for rows in rows_by_tool.values():
            for row in rows:
                if row.get("task_type") != "audio":
                    continue
                row["streaming_enabled"] = streaming
                row["preprocessing_enabled"] = preprocessing

    # ── Benchmark target checks ──────────────────────────────────────────

    def _build_target_markdown(
        self,
        context: ReportContext,
        all_results: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        model_spec = context.model_spec
        perf_refs = self._get_capped_perf_refs(model_spec)
        if not perf_refs:
            return "", []

        vllm_rows = [r for r in all_results if r.get("backend") in ("vllm", "openai-chat")]
        text_rows = [r for r in vllm_rows if r.get("task_type", "text") == "text"]
        vlm_rows = [r for r in vllm_rows if r.get("task_type", "text") == "vlm"]

        text_refs = [p for p in perf_refs if getattr(p, "task_type", "text") == "text"]
        vlm_refs = [p for p in perf_refs if getattr(p, "task_type", "text") == "vlm"]

        text_targets = (
            compute_text_target_checks(text_rows, text_refs, model_spec, context.device_str)
            if text_refs and text_rows
            else []
        )
        vlm_targets = (
            compute_vlm_target_checks(vlm_rows, vlm_refs, model_spec, context.device_str)
            if vlm_refs and vlm_rows
            else []
        )

        md = MarkdownVisualizer.build_benchmark_targets_markdown(
            text_target_rows=text_targets,
            vlm_target_rows=vlm_targets,
            text_rows_without_refs=bool(text_rows and not text_refs),
            vlm_rows_without_refs=bool(vlm_rows and not vlm_refs),
            model_name=context.model_name,
            device_str=context.device_str,
        )
        return md, text_targets + vlm_targets

    @staticmethod
    def _get_capped_perf_refs(model_spec: ModelSpec) -> List:
        dms = model_spec.device_model_spec
        raw_refs = dms.perf_reference if dms.perf_reference else []
        return [
            cap_benchmark_params(
                p, dms.max_context, dms.max_tokens_all_users, dms.max_concurrency, model_spec.model_name
            )
            for p in raw_refs
        ]

    # ── Tiered benchmark summary (image / cnn / video / audio / tts / embedding)

    def _build_tiered_summary_markdown(
        self,
        context: ReportContext,
        rows_by_tool: Dict[str, List[Dict[str, Any]]],
        evals_tput_user: Optional[float],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        model_spec = context.model_spec
        model_type_name = model_spec.model_type.name
        task_type_key = _MODEL_TYPE_TO_ROW_TASK_TYPE.get(model_type_name)
        if task_type_key is None:
            return "", []

        task_label = _MODEL_TYPE_TO_TASK_LABEL.get(model_type_name, model_type_name)

        tool_tables: List[Tuple[str, str]] = []
        rendered_summaries: List[Tuple[str, Dict[str, Any]]] = []
        for tool_label, rows in rows_by_tool.items():
            typed_rows = [r for r in rows if r.get("task_type") == task_type_key]
            if not typed_rows:
                continue
            summary = build_summary_row(
                model_spec, context.device_str, typed_rows, evals_tput_user
            )
            if summary is None:
                continue
            table = tiered_targets_markdown(summary, model_type_name)
            if not table:
                continue
            tool_tables.append((tool_label, table))
            rendered_summaries.append((tool_label, summary))

        md = MarkdownVisualizer.build_tiered_summary_markdown(
            tool_tables=tool_tables,
            task_label=task_label,
            model_name=context.model_name,
            device_str=context.device_str,
        )
        summary_data = [{"tool": tool, **summary} for tool, summary in rendered_summaries]
        return md, summary_data

    # ── Accuracy evaluations ─────────────────────────────────────────────

    def _generate_evals(self, context: ReportContext) -> Dict[str, ReportResult]:
        model_spec = context.model_spec
        eval_run_id = model_spec.model_id

        file_pattern = _eval_file_pattern(model_spec, eval_run_id)
        file_path_pattern = f"{context.workflow_log_dir}/evals_output/{file_pattern}"
        files = glob(file_path_pattern)

        if "image" in model_spec.supported_modalities:
            files = _extend_with_image_eval_files(files, context.workflow_log_dir, eval_run_id, model_spec)

        files = list(dict.fromkeys(files))
        logger.info(f"Evals: processing {len(files)} files")

        raw_evals_data = _load_raw_eval_json(files)

        if _is_simple_eval_model(model_spec):
            return self._generate_simple_eval(context, raw_evals_data)

        dict_files, list_files = _separate_files_by_format(files)

        results: Dict[str, Any] = {}
        meta_data: Dict[str, Any] = {}

        if dict_files:
            dr, dm = _extract_eval_results(dict_files)
            results.update(dr)
            meta_data.update(dm)
        if list_files:
            lr, lm = _process_list_format_eval_files(list_files)
            results.update(lr)
            meta_data.update(lm)

        if not results:
            logger.warning("No evaluation files found. Skipping evals.")
            return {SUB_SECTION_EVALS: ReportResult.empty(SUB_SECTION_EVALS)}

        report_rows = _evals_release_report_data(
            results, meta_data, model_spec, context.device_str
        )
        release_md, summary_md = MarkdownVisualizer.build_evals_markdown(
            report_rows,
            context.model_name,
            context.device_str,
            results=results,
            meta_data=meta_data,
        )

        return {
            SUB_SECTION_EVALS: ReportResult(
                name=SUB_SECTION_EVALS,
                markdown=release_md,
                data=report_rows,
                display_markdown=summary_md,
                md_filename=f"summary_{context.report_id}.md",
            )
        }

    def _generate_simple_eval(
        self, context: ReportContext, raw_evals_data: List[Dict[str, Any]]
    ) -> Dict[str, ReportResult]:
        release_md, summary_md = MarkdownVisualizer.build_evals_markdown(
            [], context.model_name, context.device_str
        )

        return {
            SUB_SECTION_EVALS: ReportResult(
                name=SUB_SECTION_EVALS,
                markdown=release_md,
                data=raw_evals_data,
                display_markdown=summary_md,
                md_filename=f"summary_{context.report_id}.md",
            )
        }


# ══════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════════


def _load_raw_eval_json(files: List[str]) -> List[Dict[str, Any]]:
    raw: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                raw.extend(data)
            else:
                raw.append(data)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load raw eval file {fp}: {e}")
    return raw


def _is_simple_eval_model(model_spec: ModelSpec) -> bool:
    return model_spec.model_type.name in (
        ModelType.CNN.name,
        ModelType.IMAGE.name,
        ModelType.EMBEDDING.name,
        ModelType.VIDEO.name,
        ModelType.TEXT_TO_SPEECH.name,
    )


def _eval_file_pattern(model_spec: ModelSpec, eval_run_id: str) -> str:
    glob_name = _EVAL_GLOB_PATTERN.get(model_spec.model_type, _DEFAULT_EVAL_GLOB)
    return f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/{glob_name}"


def _extend_with_image_eval_files(
    files: List[str], workflow_log_dir: Path, eval_run_id: str, model_spec: ModelSpec
) -> List[str]:
    base = f"{workflow_log_dir}/evals_output/eval_{eval_run_id}"
    for pattern in (
        "*_results.json",
        f"{model_spec.hf_model_repo.replace('/', '__')}/*results.json",
    ):
        files.extend(glob(f"{base}/{pattern}"))
    return files


# ── Eval JSON parsing ────────────────────────────────────────────────────


def _extract_eval_json_data(json_path: Path) -> Tuple[List[Dict], Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})
    configs = data.get("configs", {})

    first_key = list(results.keys())[0]
    first_results = results[first_key]
    extracted_metrics = {
        k: v for k, v in first_results.items()
        if "alias" not in k and "_stderr" not in k
    }
    extracted = [{first_key: extracted_metrics}]

    config = configs.get(first_key, {})
    task_name = config.get("task", first_key)

    dataset_path = list(configs.values())[0]["dataset_path"]
    for cfg in configs.values():
        assert dataset_path == cfg.get("dataset_path")

    assert task_name == first_key, f"Task name mismatch: {task_name} != {first_key}"
    meta_data = {"task_name": task_name, "dataset_path": dataset_path}
    return extracted, meta_data


def _extract_eval_results(files: List[str]) -> Tuple[Dict, Dict]:
    files = sorted(files, key=lambda f: Path(f).stat().st_mtime, reverse=True)
    results: Dict[str, Any] = {}
    meta_data: Dict[str, Any] = {}
    for json_file in files:
        res, meta = _extract_eval_json_data(Path(json_file))
        meta.pop("task_name", None)
        for task_dict in res:
            for task_name, metrics in task_dict.items():
                if task_name not in results:
                    results[task_name] = metrics
                    meta_data[task_name] = meta
    return results, meta_data


def _separate_files_by_format(files: List[str]) -> Tuple[List[str], List[str]]:
    dict_files: List[str] = []
    list_files: List[str] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                list_files.append(fp)
            elif isinstance(data, dict):
                dict_files.append(fp)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not read or parse file {fp}: {e}")
    return dict_files, list_files


def _process_list_format_eval_files(list_files: List[str]) -> Tuple[Dict, Dict]:
    list_files = sorted(list_files, key=lambda f: Path(f).stat().st_mtime, reverse=True)
    results: Dict[str, Any] = {}
    meta_data: Dict[str, Any] = {}
    for fp in list_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list) or len(data) == 0:
                logger.warning(f"List format file {fp} is empty or invalid")
                continue
            eval_data = data[0]
            task_name = eval_data.get("task_name", "image_generation")
            if task_name in results:
                continue
            results[task_name] = eval_data
            meta_data[task_name] = {
                "task_name": task_name,
                "dataset_path": eval_data.get("dataset_path", "N/A"),
            }
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not process list format file {fp}: {e}")
    return results, meta_data


# ── Eval scoring + markdown ──────────────────────────────────────────────


def _evals_release_report_data(
    results: Dict[str, Any],
    meta_data: Dict[str, Any],
    model_spec: ModelSpec,
    device_str: str,
) -> List[Dict[str, Any]]:
    eval_config = EVAL_CONFIGS[model_spec.model_name]
    report_rows: List[Dict[str, Any]] = []

    for task in eval_config.tasks:
        if not task.score:
            logger.info(f"Skipping report for task:= {task.task_name}, no eval score defined.")
            continue

        target_keys = _resolve_eval_target_keys(task.task_name, results)
        if not target_keys:
            report_rows.append(
                _build_na_eval_row(model_spec.model_name, device_str, task, meta_data)
            )
            continue

        for t_key in target_keys:
            logger.info(f"eval processing task_name: {t_key}")
            score = _compute_eval_score(task, t_key, results)
            report_rows.append(
                _build_scored_eval_row(model_spec.model_name, device_str, task, t_key, score, meta_data)
            )

    return report_rows


def _resolve_eval_target_keys(task_name: str, results: Dict) -> List[str]:
    if task_name in results:
        return [task_name]
    prefix = f"{task_name}_"
    subtasks = sorted(k for k in results if k.startswith(prefix))
    return subtasks


def _compute_eval_score(task, t_key: str, results: Dict) -> float:
    kwargs = dict(task.score.score_func_kwargs)
    kwargs["task_name"] = t_key
    configured_keys = kwargs.get("result_keys", [])
    actual_data = results.get(t_key, {})

    if not any(k in actual_data for k in configured_keys):
        valid = [
            k for k, v in actual_data.items()
            if isinstance(v, (int, float)) and "stderr" not in k and "alias" not in k
        ]
        if valid:
            logger.info(f"  Metric mismatch for {t_key}. Auto-detected replacement: {valid[0]}")
            kwargs["result_keys"] = [valid[0]]

    try:
        score = task.score.score_func(results, task_name=t_key, kwargs=kwargs)
    except Exception as e:
        logger.warning(f"  Could not calculate score for {t_key}: {e}")
        score = 100.0 if kwargs.get("unit") == "WER" else 0.0

    if kwargs.get("unit") == "WER":
        score = 100 - score

    return score


def _build_scored_eval_row(
    model_name: str, device_str: str, task, t_key: str, score: float, meta_data: Dict
) -> Dict[str, Any]:
    ts = task.score

    ratio_to_published = score / ts.published_score if ts.published_score else "N/A"
    if ts.gpu_reference_score:
        ratio_to_reference = score / ts.gpu_reference_score
        accuracy_check = ReportCheckTypes.from_result(
            ratio_to_reference >= (1.0 - ts.tolerance)
        )
    else:
        ratio_to_reference = "N/A"
        accuracy_check = (
            ReportCheckTypes.from_result(ratio_to_published >= (1.0 - ts.tolerance))
            if ts.published_score
            else ReportCheckTypes.NA
        )

    return {
        "model": model_name,
        "device": device_str,
        "task_name": t_key,
        "accuracy_check": accuracy_check,
        "score": score,
        "ratio_to_reference": ratio_to_reference,
        "gpu_reference_score": ts.gpu_reference_score,
        "gpu_reference_score_ref": ts.gpu_reference_score_ref,
        "ratio_to_published": ratio_to_published,
        "published_score": ts.published_score,
        "published_score_ref": ts.published_score_ref,
        "metadata": meta_data.get(t_key),
    }


def _build_na_eval_row(
    model_name: str, device_str: str, task, meta_data: Dict
) -> Dict[str, Any]:
    ts = task.score
    return {
        "model": model_name,
        "device": device_str,
        "task_name": task.task_name,
        "accuracy_check": ReportCheckTypes.NA,
        "score": "N/A",
        "ratio_to_reference": "N/A",
        "gpu_reference_score": ts.gpu_reference_score,
        "gpu_reference_score_ref": ts.gpu_reference_score_ref,
        "ratio_to_published": "N/A",
        "published_score": ts.published_score,
        "published_score_ref": ts.published_score_ref,
        "metadata": meta_data.get(task.task_name),
    }

