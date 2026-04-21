# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Per-reference performance target-check computation for text/VLM benchmarks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from workflows.model_spec import ModelSpec
from workflows.workflow_types import ReportCheckTypes

_NA_TARGET_CHECKS: Dict[str, ReportCheckTypes] = {
    "ttft_check": ReportCheckTypes.NA,
    "tput_user_check": ReportCheckTypes.NA,
    "tput_check": ReportCheckTypes.NA,
}


def compute_text_target_checks(
    text_rows: List[Dict[str, Any]],
    text_refs: List,
    model_spec: ModelSpec,
    device_str: str,
) -> List[Dict[str, Any]]:
    res_dict = {
        (r["input_sequence_length"], r["output_sequence_length"], r["max_con"]): r
        for r in text_rows
    }

    perf_results: Dict[tuple, Dict[str, Any]] = {}
    for p_ref in text_refs:
        key = (p_ref.isl, p_ref.osl, p_ref.max_concurrency)
        res = res_dict.get(key)
        entry: Dict[str, Any] = {
            "isl": p_ref.isl,
            "osl": p_ref.osl,
            "max_concurrency": res["max_con"] if res else p_ref.max_concurrency,
            "model": model_spec.model_name,
            "device": device_str,
        }
        if res:
            entry.update(
                ttft=res["mean_ttft_ms"],
                tput_user=res["mean_tps"],
                tput=res["tps_decode_throughput"],
            )
            entry["target_checks"] = {
                tgt_name: _build_target_checks_for_ref(res, perf_target)
                for tgt_name, perf_target in p_ref.targets.items()
            }
        else:
            entry.update(ttft="N/A", tput_user="N/A", tput="N/A")
            entry["target_checks"] = {
                tgt_name: dict(_NA_TARGET_CHECKS) for tgt_name in p_ref.targets
            }
        perf_results[key] = entry

    return [perf_results[k] for k in sorted(perf_results)]


def compute_vlm_target_checks(
    vlm_rows: List[Dict[str, Any]],
    vlm_refs: List,
    model_spec: ModelSpec,
    device_str: str,
) -> List[Dict[str, Any]]:

    res_dict = {
        (
            r.get("isl", r.get("input_sequence_length", 0)),
            r.get("osl", r.get("output_sequence_length", 0)),
            r["image_height"],
            r["image_width"],
            r["images_per_prompt"],
            r["max_con"],
        ): r
        for r in vlm_rows
    }

    perf_results: Dict[tuple, Dict[str, Any]] = {}
    for p_ref in vlm_refs:
        key = (
            p_ref.isl,
            p_ref.osl,
            p_ref.image_height,
            p_ref.image_width,
            p_ref.images_per_prompt,
            p_ref.max_concurrency,
        )
        res = res_dict.get(key)
        entry: Dict[str, Any] = {
            "isl": p_ref.isl,
            "osl": p_ref.osl,
            "max_concurrency": res["max_con"] if res else p_ref.max_concurrency,
            "image_height": p_ref.image_height,
            "image_width": p_ref.image_width,
            "images_per_prompt": p_ref.images_per_prompt,
            "num_requests": res["num_requests"] if res else "N/A",
            "model": model_spec.model_name,
            "device": device_str,
        }
        if res:
            entry.update(
                ttft=res["mean_ttft_ms"],
                tput_user=res["mean_tps"],
                tput=res["tps_decode_throughput"],
            )
            entry["target_checks"] = {
                tgt_name: _build_target_checks_for_ref(res, perf_target)
                for tgt_name, perf_target in p_ref.targets.items()
            }
        else:
            entry.update(ttft="N/A", tput_user="N/A", tput="N/A")
            entry["target_checks"] = {
                tgt_name: dict(_NA_TARGET_CHECKS) for tgt_name in p_ref.targets
            }
        perf_results[key] = entry

    return [perf_results[k] for k in sorted(perf_results)]


def flatten_target_checks(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat_rows: List[Dict[str, Any]] = []
    for row in rows:
        flat = {k: v for k, v in row.items() if k != "target_checks"}
        for target_name, checks in row.get("target_checks", {}).items():
            for metric, value in checks.items():
                flat[f"{target_name}_{metric}"] = value
        flat_rows.append(flat)
    return flat_rows


def _compute_perf_target_check(
    measured: float,
    reference: Optional[float],
    tolerance: float,
    higher_is_better: bool,
) -> Tuple[Optional[float], ReportCheckTypes]:
    """Compute ratio and pass/fail check for a single metric against a reference."""
    if reference is None or reference <= 0:
        return None, ReportCheckTypes.NA
    ratio = measured / reference
    if higher_is_better:
        passed = ratio > (1 - tolerance)
    else:
        passed = ratio < (1 + tolerance)
    return ratio, ReportCheckTypes.from_result(passed)


def _build_target_checks_for_ref(res: Dict[str, Any], perf_target) -> Dict[str, Any]:
    target_check: Dict[str, Any] = {}

    if perf_target.ttft_ms is not None:
        assert perf_target.ttft_ms > 0, f"ttft_ms is not > 0: {perf_target.ttft_ms}"
        ratio, check = _compute_perf_target_check(
            res["mean_ttft_ms"],
            perf_target.ttft_ms,
            perf_target.tolerance,
            higher_is_better=False,
        )
        target_check.update(
            ttft=perf_target.ttft_ms, ttft_ratio=ratio, ttft_check=check
        )
    else:
        target_check["ttft_check"] = ReportCheckTypes.NA

    if perf_target.tput_user is not None:
        assert perf_target.tput_user > 0, (
            f"tput_user is not > 0: {perf_target.tput_user}"
        )
        ratio, check = _compute_perf_target_check(
            res["mean_tps"],
            perf_target.tput_user,
            perf_target.tolerance,
            higher_is_better=True,
        )
        target_check.update(
            tput_user=perf_target.tput_user,
            tput_user_ratio=ratio,
            tput_user_check=check,
        )
    else:
        target_check["tput_user_check"] = ReportCheckTypes.NA

    if perf_target.tput is not None:
        assert perf_target.tput > 0, f"tput is not > 0: {perf_target.tput}"
        ratio, check = _compute_perf_target_check(
            res["tps_decode_throughput"],
            perf_target.tput,
            perf_target.tolerance,
            higher_is_better=True,
        )
        target_check.update(tput=perf_target.tput, tput_ratio=ratio, tput_check=check)
    else:
        target_check["tput_check"] = ReportCheckTypes.NA

    return target_check
