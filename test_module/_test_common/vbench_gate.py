# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Wan2.2 I2V VBench gate: the eval's clips against the GPU reference runs.

Follows the validation plan's release CI. Each run is compared with the GPU
run made the same way (``drift``: did our implementation drift?), and a cached
run also with the uncached GPU run (``budget``: are we inside the expected cost
of caching?). Two kinds of check, with thresholds in GATE_CONFIG:

- VBench: each gated dimension's mean over the same clips may be at most
  ``tolerance_pct`` below the reference's. Scoring higher never fails.
- Output, drift comparison only: each clip's mean LPIPS under a cap, and the
  run's mean PSNR / SSIM / LPIPS within a margin of the calibration run's over
  the same clips. Set per resolution; a resolution without calibration is not
  gated.

A comparison that could not be made is NA, not a pass. The VBench report
section shows the checks (report_module.renderers.render_vbench).
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from .report_types import ReportCheckTypes
from .vbench_i2v import GPU_REFERENCE_RUN, GPU_REFERENCE_RUN_CACHED, resolution

GATE_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "reference_config"
    / "evals"
    / "eval_targets"
    / "wan22_i2v_vbench_gate.json"
)
PASS, FAIL, NA = "PASS", "FAIL", "NA"
BOOTSTRAP_SAMPLES = 2000


def load_config(path: Path = GATE_CONFIG) -> dict:
    return json.loads(path.read_text())


def reference_runs(dbcache: bool | None) -> list[tuple[str, str]]:
    """``(run, role)`` to compare with: the GPU run made the same way first."""
    if dbcache:
        return [(GPU_REFERENCE_RUN_CACHED, "drift"), (GPU_REFERENCE_RUN, "budget")]
    return [(GPU_REFERENCE_RUN, "drift")]


def _run_of(reference: dict, runs: list[tuple[str, str]]) -> tuple[str, str] | None:
    name = reference.get("reference", "")
    return next(((r, role) for r, role in runs if name.endswith(f"/{r}")), None)


def _ci_pct(per_clip: dict, base: float) -> list[float] | None:
    """95% bootstrap interval of the mean per-clip difference, as % of ``base``."""
    deltas = list(per_clip.values())
    if len(deltas) < 2 or not base:
        return None
    rng = random.Random(0)
    means = sorted(
        statistics.fmean(rng.choices(deltas, k=len(deltas)))
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    lo, hi = means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]
    return [round(100 * lo / base, 2), round(100 * hi / base, 2)]


def _vbench_checks(reference: dict, run: str, role: str, config: dict) -> list[dict]:
    tolerance = config["tolerance_pct"][role]
    checks = []
    dimensions = reference.get("dimensions") or {}
    for name in config["gated_dimensions"] + config["reported_dimensions"]:
        record = dimensions.get(name)
        gated = name in config["gated_dimensions"]
        check = {"reference": run, "role": role, "check": name}
        if not record or not record.get("gpu"):
            checks.append({**check, "status": NA, "detail": "not scored"})
            continue
        change = 100 * (record["tt"] - record["gpu"]) / record["gpu"]
        check.update(
            value=round(change, 2),
            threshold=-tolerance if gated else None,
            ci_pct=_ci_pct(record.get("per_clip") or {}, record["gpu"]),
            clips=record.get("clips"),
            detail=f"TT {record['tt']:.4f} vs GPU {record['gpu']:.4f}",
        )
        if gated:
            check["status"] = PASS if change >= -tolerance else FAIL
        else:
            check["status"] = "REPORTED"
        checks.append(check)
    return checks


def _output_checks(
    reference: dict, run: str, clips: list[dict], config: dict
) -> list[dict]:
    """Per-clip LPIPS cap and run-mean shifts, per resolution, against the calibration."""
    by_resolution: dict[str, list[dict]] = {}
    for clip in clips:
        if clip.get("width") and clip.get("height"):
            res = resolution(clip["width"], clip["height"])
            by_resolution.setdefault(res, []).append(clip)
    checks = []
    for res, res_clips in by_resolution.items():
        limits = (config.get("output") or {}).get(res)
        base = {"reference": run, "role": "drift", "resolution": res}
        if not limits:
            checks.append(
                {**base, "check": "output", "status": NA, "detail": "no calibration"}
            )
            continue
        measured = {}
        for clip in res_clips:
            metrics = {
                key: (reference.get(key) or {}).get(clip["file"], {}).get("mean")
                for key in ("psnr", "ssim", "lpips")
            }
            if None not in metrics.values():
                measured[clip.get("video_id") or clip["file"]] = metrics
        if not measured:
            checks.append(
                {**base, "check": "output", "status": NA, "detail": "no clip compared"}
            )
            continue
        cap = limits["max_clip_lpips"]
        over = sorted(v for v, m in measured.items() if m["lpips"] > cap)
        checks.append(
            {
                **base,
                "check": "clip_lpips",
                "value": max(m["lpips"] for m in measured.values()),
                "threshold": cap,
                "status": FAIL if over else PASS,
                "detail": f"over the cap: {', '.join(over)}" if over else "",
            }
        )
        calibration = (limits.get("calibration") or {}).get(run) or {}
        shared = [v for v in measured if v in calibration]
        if not shared:
            checks.append(
                {
                    **base,
                    "check": "mean_shift",
                    "status": NA,
                    "detail": "no calibrated clip",
                }
            )
            continue
        for key, margin in limits["mean_margin"].items():
            now = statistics.fmean(measured[v][key] for v in shared)
            then = statistics.fmean(calibration[v][key] for v in shared)
            # LPIPS: lower is closer; PSNR and SSIM: higher is closer.
            shift = (then - now) if key != "lpips" else (now - then)
            checks.append(
                {
                    **base,
                    "check": f"mean_{key}",
                    "value": round(now, 4),
                    "threshold": round(
                        then - margin if key != "lpips" else then + margin, 4
                    ),
                    "clips": len(shared),
                    "status": FAIL if shift > margin else PASS,
                    "detail": f"calibration {then:.4f} over the same clips",
                }
            )
    return checks


def grade(result: dict, runs: list[tuple[str, str]], config: dict) -> dict:
    """The gate over a scorer result: every check, and the verdict as an accuracy_check."""
    checks = []
    compared = {}
    for reference in result.get("gpu_references") or []:
        match = _run_of(reference, runs)
        if match:
            compared[match[0]] = reference
    for run, role in runs:
        reference = compared.get(run)
        if reference is None or "error" in reference:
            detail = (reference or {}).get("error") or "no comparison"
            checks.append(
                {
                    "reference": run,
                    "role": role,
                    "check": "comparison",
                    "status": NA,
                    "detail": detail,
                }
            )
            continue
        checks += _vbench_checks(reference, run, role, config)
        if role == "drift":
            checks += _output_checks(reference, run, result.get("clips") or [], config)
    statuses = {c["status"] for c in checks}
    if FAIL in statuses:
        verdict = ReportCheckTypes.FAIL
    elif PASS in statuses:
        verdict = ReportCheckTypes.PASS
    else:
        verdict = ReportCheckTypes.NA
    return {"accuracy_check": verdict, "checks": checks}
