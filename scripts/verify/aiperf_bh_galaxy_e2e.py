#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""End-to-end verification of the AIPerf benchmark path for Blackhole Galaxy.

Readiness item 7.1 (llm-gauntlet#72): confirm the ``--tools aiperf`` benchmark
path works end to end against a model "served" on a 32-chip Blackhole Galaxy.
Real Galaxy hardware is not required here: a mock OpenAI-compatible server
(``llm-d-inference-sim``) stands in for the served model, seeded with
Blackhole-Galaxy-representative latency. See
``scripts/verify/run_aiperf_bh_galaxy_e2e.sh`` for the wrapper that starts the
mock and this harness.

What it exercises — the *real* repo code path, not a reimplementation:

    AIPerfDriver (`python -m aiperf profile`)  →  raw profile_export_aiperf.json
      →  AIPerfParser.parse  →  Block
      →  apply_target_checks (tiered functional/complete/target grading)
      →  report_module.generate_report  →  markdown + JSON report

Only the inference server and the tokenizer are stand-ins; everything from the
driver invocation through parsing, grading and report rendering is the code that
runs on real hardware.

The performance targets below are illustrative placeholders. Replace them with
the AIPerf-measured Blackhole Galaxy numbers (Appendix B target sheet) via
``--targets-json`` once they exist; the mock latency (set on the sim, not here)
is what makes a tier pass or fail.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_module.config import DriverContext, LLMRunConfig, ServerConnection  # noqa: E402
from llm_module.drivers.aiperf import AIPerfDriver  # noqa: E402
from llm_module.runner import LLMPerformanceRunner  # noqa: E402
from llm_module.server_control import HttpServerController  # noqa: E402
from report_module.generator import generate_report  # noqa: E402
from report_module.schema import ReportSchema  # noqa: E402
from workflows.utils_report import PerformanceTarget  # noqa: E402
from workflows.workflow_types import ReportCheckTypes  # noqa: E402

logger = logging.getLogger("aiperf_bh_galaxy_e2e")


# Illustrative per-system Blackhole Galaxy targets (NOT measured). Loose enough
# that the seeded mock latency passes, so the E2E demonstrates real grading.
# Replace with the Appendix B AIPerf sheet values via --targets-json.
_DEFAULT_TARGETS = {
    "functional": {"ttft_ms": 2000.0, "tput_user": 10.0, "tput": 8.0},
    "complete": {"ttft_ms": 1000.0, "tput_user": 20.0, "tput": 15.0},
    "target": {"ttft_ms": 500.0, "tput_user": 30.0, "tput": 20.0},
}

# Small Blackhole-Galaxy-shaped sweep. Honors readiness §5.7 (#64): every graded
# concurrency level carries >= 3 distinct input lengths so a scaling-quality fit
# is possible. Kept tiny (short ISL/OSL, few prompts) so the mock run is quick.
_DEFAULT_ISLS = [128, 512, 1024]
_DEFAULT_CONCURRENCIES = [1, 8]
_DEFAULT_OSL = 128


_TIERS = ("functional", "complete", "target")


def _build_targets(spec: Dict[str, Dict[str, float]]) -> Dict[str, PerformanceTarget]:
    tolerance = 0.05
    # Only the known grading tiers are turned into targets; other keys (e.g. a
    # "_derivation" note documenting where the numbers came from) are ignored.
    return {
        tier: PerformanceTarget(
            ttft_ms=vals.get("ttft_ms"),
            tput_user=vals.get("tput_user"),
            tput=vals.get("tput"),
            tolerance=vals.get("tolerance", tolerance),
        )
        for tier, vals in spec.items()
        if tier in _TIERS and isinstance(vals, dict)
    }


def _build_sweep(
    isls: List[int],
    concurrencies: List[int],
    osl: int,
    num_prompts: int,
    targets: Dict[str, PerformanceTarget],
) -> List[LLMRunConfig]:
    configs: List[LLMRunConfig] = []
    for concurrency in concurrencies:
        for isl in isls:
            configs.append(
                LLMRunConfig(
                    isl=isl,
                    osl=osl,
                    max_concurrency=concurrency,
                    num_prompts=max(num_prompts, concurrency),
                    targets=dict(targets),
                )
            )
    return configs


def _summarize_block(block) -> Dict[str, object]:
    data = block.data if isinstance(block.data, dict) else {}
    verdict = data.get("target_check")
    verdict_str = (
        ReportCheckTypes.to_display_string(verdict)
        if isinstance(verdict, ReportCheckTypes)
        else str(verdict)
    )
    return {
        "title": block.title,
        "isl": data.get("input_sequence_length"),
        "osl": data.get("output_sequence_length"),
        "concurrency": data.get("concurrency"),
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "mean_tpot_ms": data.get("mean_tpot_ms"),
        "tput_user": data.get("tput_user"),
        "tps_decode_throughput": data.get("tps_decode_throughput"),
        "request_throughput": data.get("request_throughput"),
        "target_check": verdict_str,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost")
    parser.add_argument("--service-port", type=int, default=8000)
    parser.add_argument(
        "--served-model",
        default="tenstorrent/blackhole-galaxy-mock",
        help="Model name the mock serves and AIPerf requests (must match the sim).",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Real HF tokenizer AIPerf uses for synthetic prompts / token counts. "
        "The served model is a mock, so a small real tokenizer stands in.",
    )
    parser.add_argument("--device", default="BLACKHOLE_GALAXY")
    parser.add_argument(
        "--venv-python",
        default=None,
        help="Python interpreter of the aiperf venv (defaults to this interpreter).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "workflow_logs" / "aiperf_bh_galaxy_e2e"),
    )
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument(
        "--isls",
        default=None,
        help="Comma-separated input-sequence-lengths to sweep "
        f"(default {_DEFAULT_ISLS}). Keep >= 3 distinct values per graded "
        "concurrency level for the scaling-quality fit (readiness §5.7).",
    )
    parser.add_argument(
        "--concurrencies",
        default=None,
        help=f"Comma-separated concurrency levels to sweep (default {_DEFAULT_CONCURRENCIES}).",
    )
    parser.add_argument("--osl", type=int, default=_DEFAULT_OSL)
    parser.add_argument(
        "--targets-json",
        default=None,
        help="Path to a JSON file overriding the tiered targets "
        "({tier: {ttft_ms, tput_user, tput, tolerance}}).",
    )
    parser.add_argument("--per-run-timeout-s", type=float, default=600.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    target_spec = _DEFAULT_TARGETS
    if args.targets_json:
        target_spec = json.loads(Path(args.targets_json).read_text())
    targets = _build_targets(target_spec)

    isls = (
        [int(x) for x in args.isls.split(",") if x.strip()]
        if args.isls
        else _DEFAULT_ISLS
    )
    concurrencies = (
        [int(x) for x in args.concurrencies.split(",") if x.strip()]
        if args.concurrencies
        else _DEFAULT_CONCURRENCIES
    )

    configs = _build_sweep(isls, concurrencies, args.osl, args.num_prompts, targets)
    logger.info(
        "Blackhole Galaxy AIPerf E2E: %d sweep points (ISLs=%s, concurrencies=%s)",
        len(configs),
        isls,
        concurrencies,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    server = ServerConnection(
        base_url=args.base_url,
        service_port=args.service_port,
        model=args.served_model,
        tokenizer=args.tokenizer,
    )
    context = DriverContext(
        output_dir=output_dir / "llm",
        device=args.device,
        per_run_timeout_s=args.per_run_timeout_s,
    )
    venv_python = Path(args.venv_python) if args.venv_python else None
    driver = AIPerfDriver(venv_python=venv_python)
    controller = HttpServerController(
        base_url=args.base_url, service_port=args.service_port
    )

    runner = LLMPerformanceRunner(driver=driver, server_controller=controller)
    result = runner.run(configs, server, context)

    summaries = [_summarize_block(b) for b in result.blocks]
    logger.info("Per sweep-point results:")
    for row in summaries:
        logger.info(
            "  ISL=%s OSL=%s conc=%s | ttft=%.1fms tpot=%.1fms tput_user=%.1f "
            "tput=%.1f | %s",
            row["isl"],
            row["osl"],
            row["concurrency"],
            row["mean_ttft_ms"] or 0.0,
            row["mean_tpot_ms"] or 0.0,
            row["tput_user"] or 0.0,
            row["tps_decode_throughput"] or 0.0,
            row["target_check"],
        )

    report_dir = output_dir / "reports"
    report_result = None
    if result.blocks:
        schema = ReportSchema(
            metadata={
                "report_id": f"aiperf_bh_galaxy_e2e_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                "model_name": args.served_model,
                "device": args.device,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            sections=list(result.blocks),
        )
        report_result = generate_report(schema, report_dir)

    verdict_path = output_dir / "e2e_summary.json"
    verdict_path.write_text(
        json.dumps(
            {
                "served_model": args.served_model,
                "tokenizer": args.tokenizer,
                "device": args.device,
                "num_sweep_points": len(configs),
                "num_blocks": len(result.blocks),
                "return_codes": result.return_codes,
                "parse_failures": result.parse_failures,
                "ok": result.ok,
                "results": summaries,
                "report_markdown": str(report_result.markdown_path)
                if report_result
                else None,
            },
            indent=2,
        )
    )

    logger.info("Wrote E2E summary: %s", verdict_path)
    if report_result:
        logger.info("Wrote report: %s", report_result.markdown_path)

    graded = [r for r in summaries if r["target_check"] not in ("N/A", "None", None)]
    if not result.ok:
        logger.error(
            "AIPerf E2E FAILED: return_codes=%s parse_failures=%s",
            result.return_codes,
            result.parse_failures,
        )
        return 1
    if not result.blocks:
        logger.error("AIPerf E2E FAILED: no benchmark blocks produced.")
        return 1
    logger.info(
        "AIPerf E2E PASSED: %d/%d sweep points graded, %d block(s) produced.",
        len(graded),
        len(configs),
        len(result.blocks),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
