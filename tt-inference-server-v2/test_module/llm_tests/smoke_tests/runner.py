# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Drive the prefill/decode smoke-test suite, stack bring-up included."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from report_module.schema import Block

logger = logging.getLogger(__name__)

_SMOKE_DIR = Path(__file__).resolve().parent
_RUN_TESTS = _SMOKE_DIR / "run_tests.sh"
# run_stack.sh serves tokenizers/$MODEL from the cpp_server tree (repo root is
# four levels up from this file).
_TOKENIZERS_DIR = (
    _SMOKE_DIR.parents[3] / "tt-media-server" / "cpp_server" / "tokenizers"
)


def _available_served_models() -> List[str]:
    """org/model dirs present under the cpp_server tokenizers tree."""
    if not _TOKENIZERS_DIR.is_dir():
        return []
    return sorted(
        f"{org.name}/{model.name}"
        for org in _TOKENIZERS_DIR.iterdir()
        if org.is_dir()
        for model in org.iterdir()
        if model.is_dir()
    )


# RESULT_LOG lines are timestamp-prefixed ("[HH:MM:SS] msg"); per-test
# sections open with a "--------- TestNN ..." banner. Within a section each
# request emits one usage/stream line we parse into a structured row.
_TS_PREFIX = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]\s*")
_BANNER = re.compile(r"-{3,}\s*Test0*(\d+)\b")
# Multi-turn tests log a "--------- Conversation X/Y ---------" banner per
# conversation; each turn within it is a separate request line.
_CONV = re.compile(r"-{3,}\s*Conversation\s+(\d+)\s*/")
_TEST_NUM = re.compile(r"test_0*(\d+)")
# _chat_messages: "usage prompt=8 cached=0 completion=16 total=24 finish=stop ..."
_USAGE = re.compile(r"usage prompt=(\d+) cached=(\d+) completion=(\d+)")
# _chat_stream_messages: "stream ttft=82.0ms total=.. tps=.. chunks=16 finish=..
#   prompt=404 cached=0 completion=16" (ttft/prompt/completion may be "None")
_STREAM = re.compile(
    r"stream ttft=(\S+) total=\S+ tps=(\S+) chunks=\d+ finish=\S+ "
    r"prompt=(\S+) cached=(\d+) completion=(\S+)"
)


def _to_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_THRESHOLD = int(os.environ.get("THRESHOLD", "1000"))


def _prefill_on(prompt_tokens: Optional[int]) -> Optional[str]:
    if prompt_tokens is None:
        return None
    return "prefill" if prompt_tokens >= _THRESHOLD else "decode"


def _to_ttft_ms(value: str) -> Optional[float]:
    if not value or value == "None":
        return None
    try:
        if value.endswith("ms"):
            return round(float(value[:-2]), 1)
        return round(float(value.rstrip("s")) * 1000, 1)
    except ValueError:
        return None


def _to_tps(value: str) -> Optional[float]:
    if not value or value == "None":
        return None
    try:
        return round(float(value), 1)
    except ValueError:
        return None


@dataclass(frozen=True)
class SmokeTestResult:
    suite: str
    return_code: int
    elapsed_seconds: float


def _parse_junit(path: Path) -> List[Dict]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return []
    records: List[Dict] = []
    for tc in root.iter("testcase"):
        failure = tc.find("failure")
        error = tc.find("error")
        skipped = tc.find("skipped")
        bad = failure if failure is not None else error
        if bad is not None:
            result, node = "FAIL", bad
        elif skipped is not None:
            result, node = "SKIP", skipped
        else:
            result, node = "PASS", None
        try:
            duration = round(float(tc.get("time", "0") or 0), 2)
        except ValueError:
            duration = None
        detail = ""
        if node is not None:
            detail = (node.get("message") or node.text or "").strip().replace("\n", " ")
        records.append(
            {
                "test": tc.get("name", ""),
                "result": result,
                "duration_s": duration,
                "detail": detail[:200],
            }
        )
    return records


def _parse_requests(path: Path) -> Dict[int, List[Dict]]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return {}
    out: Dict[int, List[Dict]] = {}
    cur: Optional[int] = None
    conv: Optional[int] = None
    turn = 0
    for raw in lines:
        line = _TS_PREFIX.sub("", raw).strip()
        banner = _BANNER.search(line)
        if banner:
            cur, conv, turn = int(banner.group(1)), None, 0
            continue
        conv_banner = _CONV.search(line)
        if conv_banner:
            conv, turn = int(conv_banner.group(1)), 0
            continue
        if cur is None:
            continue
        m = _STREAM.search(line)
        if m:
            stream, groups = True, (m.group(3), m.group(4), m.group(5))
            ttft, tps = _to_ttft_ms(m.group(1)), _to_tps(m.group(2))
        else:
            m = _USAGE.search(line)
            if not m:
                continue
            stream = False
            groups, ttft, tps = (m.group(1), m.group(2), m.group(3)), None, None
        turn += 1
        prompt = _to_int(groups[0])
        out.setdefault(cur, []).append(
            {
                "stream": stream,
                "conv": conv,
                "turn": turn,
                "ttft_ms": ttft,
                "tps": tps,
                "prompt_tokens": prompt,
                "cached_tokens": _to_int(groups[1]),
                "completion_tokens": _to_int(groups[2]),
                "prefill_on": _prefill_on(prompt),
            }
        )
    return out


def _build_records(junit: List[Dict], requests: Dict[int, List[Dict]]) -> List[Dict]:
    records: List[Dict] = []
    for t in junit:
        m = _TEST_NUM.search(t["test"])
        reqs = requests.get(int(m.group(1)), []) if m else []
        base = {"test": t["test"], "result": t["result"], "duration_s": t["duration_s"]}
        if not reqs:
            records.append({**base, "stream": None})
            continue
        for r in reqs:
            records.append({**base, **r})
    return records


def run_smoke_tests(ctx) -> List[SmokeTestResult]:
    from workflow_module import accept_blocks

    out_dir = Path(ctx.output_path) / "prefill_decode"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_log = out_dir / "tt_test_results.log"
    junit_xml = out_dir / "junit.xml"

    env = os.environ.copy()
    env.setdefault("RESULT_LOG", str(result_log))
    # Pick the model the stack + test use (run_stack.sh serves tokenizers/$MODEL;
    # test_prefill_decode.py sends {"model": $MODEL}). --served-model arrives as
    # $MODEL from run.py and wins; otherwise default to the --model's HF repo
    # (the tokenizer-dir path). Either way an explicit $MODEL is respected.
    served = getattr(ctx.model_spec, "hf_model_repo", None)
    if served:
        env.setdefault("MODEL", served)
    served_model = env.get("MODEL")  # what the stack will actually serve

    # Fail fast with a clear message: run_stack.sh would otherwise launch a
    # worker against a missing tokenizers/$MODEL dir and the cpp binary core-
    # dumps with no obvious cause. (None means run_stack.sh uses its own default.)
    if served_model and not (_TOKENIZERS_DIR / served_model).is_dir():
        avail = _available_served_models()
        raise RuntimeError(
            f"No tokenizer dir for served model {served_model!r} at "
            f"{_TOKENIZERS_DIR / served_model}. The prefill_decode mock stack "
            f"serves tokenizers/$MODEL; pass --served-model with one of: "
            f"{avail or '(none found — tokenizers tree missing)'}."
        )

    logger.info(
        "[prefill_decode] bring up stack + run %s (model=%s)",
        _RUN_TESTS.name,
        env.get("MODEL", "<run_stack default>"),
    )
    started = time.time()
    # Args after run_tests.sh are forwarded to pytest; --junitxml needs no plugin.
    proc = subprocess.run(["bash", str(_RUN_TESTS), f"--junitxml={junit_xml}"], env=env)
    elapsed = time.time() - started

    junit = _parse_junit(junit_xml)
    records = _build_records(junit, _parse_requests(result_log))
    failures = [
        {"test": t["test"], "message": t["detail"]}
        for t in junit
        if t["result"] == "FAIL"
    ]
    summary = {
        "total": len(junit),
        "passed": sum(1 for t in junit if t["result"] == "PASS"),
        "failed": sum(1 for t in junit if t["result"] == "FAIL"),
        "skipped": sum(1 for t in junit if t["result"] == "SKIP"),
    }

    mark = "✅" if proc.returncode == 0 else "❌"
    logger.info(
        "%s [prefill_decode] rc=%d %d/%d passed (%.1fs); log -> %s",
        mark,
        proc.returncode,
        summary["passed"],
        summary["total"],
        elapsed,
        result_log,
    )

    block = Block(
        kind="prefill_decode",
        id="prefill_decode",
        title="Prefill/decode smoke tests",
        task_type="prefill_decode",
        data={
            "suite": "prefill_decode",
            "return_code": proc.returncode,
            "elapsed_seconds": round(elapsed, 1),
            "summary": summary,
            "records": records,
            "failures": failures,
            "results_dir": str(out_dir),
            "result_files": sorted(f.name for f in out_dir.iterdir() if f.is_file()),
        },
    )
    accept_blocks(
        [block],
        envelope={
            "model_name": served_model or getattr(ctx.model_spec, "model_name", ""),
            "device": ctx.device.name
            if hasattr(ctx.device, "name")
            else str(ctx.device),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return [SmokeTestResult("prefill_decode", proc.returncode, elapsed)]


__all__ = ["SmokeTestResult", "run_smoke_tests"]
