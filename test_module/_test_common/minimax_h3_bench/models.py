# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Paths, tunables, the per-case timeout table, and failure classification.

No network calls. The only subprocess is ffprobe/ffmpeg inside ``has_audio``.
Output and asset locations are set with :func:`configure` (a test passes its own
directories) and fall back to ``H3_OUT`` / ``H3_ASSETS``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

BASE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(BASE, "cases.json")
# The pinned input pack. The repo carries the prompts and the manifests; media files
# are staged next to them (see ``host.resolve_assets_dir``).
REPO_ASSETS = os.path.normpath(
    os.path.join(BASE, "..", "..", "..", "test_fixtures", "datasets", "minimax_h3")
)
SHARED_ASSETS = "/mnt/MLPerf/tt-shield/persistent-volume/h3-assets"

_STATE = {
    "assets": os.environ.get("H3_ASSETS") or REPO_ASSETS,
    # Never inside the package: a test that logs before configure() must not litter the repo.
    "out": os.environ.get("H3_OUT")
    or os.path.join(tempfile.gettempdir(), "minimax_h3_bench"),
}


def configure(assets_dir: str | None = None, out_dir: str | None = None) -> None:
    """Point the engine at an asset pack and a results directory for this run."""
    if assets_dir:
        _STATE["assets"] = str(assets_dir)
    if out_dir:
        _STATE["out"] = str(out_dir)


def assets_dir() -> str:
    return _STATE["assets"]


def out_dir() -> str:
    return _STATE["out"]


def clips_dir() -> str:
    return os.path.join(out_dir(), "out")


def logs_dir() -> str:
    return os.path.join(out_dir(), "logs")


def results_path() -> str:
    return os.path.join(out_dir(), "results.jsonl")


def results_csv_path() -> str:
    return os.path.join(out_dir(), "results.csv")


CSV_COLUMNS = [
    "ts", "combo", "case", "tag", "measured", "outcome", "failure_class",
    "engine", "hw", "gpus", "node", "primary_metric_s", "engine_inference_s",
    "e2e_s", "queue_s", "gen_s", "mvhd_duration_s", "has_audio", "out_bytes",
    "out_sha256", "submit_http", "content_http", "job_id", "attempt",
    "timeout_s", "steps_requested", "steps_effective", "error_message", "out_file",
]  # fmt: skip
POLL_S = 2.0
# Re-submits of a TRANSIENT submit/transport failure (connection reset, 5xx, non-JSON
# body); is_transient() never retries a live job, a timeout, a refusal or a device fault.
MAX_RETRIES = int(os.environ.get("H3_MAX_RETRIES", "2"))
RETRY_BACKOFF_BASE_S = float(os.environ.get("H3_RETRY_BACKOFF_S", "5"))
TIMEOUT_FLOOR_S = int(os.environ.get("H3_TIMEOUT_FLOOR", "300"))
TIMEOUT_CEIL_S = int(os.environ.get("H3_TIMEOUT_CEIL", "14400"))
COST_SECONDS_PER_UNIT = float(os.environ.get("H3_COST_SECONDS_PER_UNIT", "0.28"))
# Zero (status, progress) change while still queued for this long -> the endpoint
# never started the job. Anything that reached in_progress is governed by the budget.
STALL_S = float(os.environ.get("H3_STALL_S", "900"))
POLL_HISTORY_SAMPLE_S = 30
POLL_HISTORY_MAX = 200
TERMINAL_STATUSES = {"completed", "succeeded", "failed", "cancelled"}
QUEUED_STATUSES = {"queued", "pending", "submitted", "accepted"}
LOST_POLLS = int(os.environ.get("H3_LOST_POLLS", "30"))
TRANSPORT_POLLS = 150  # ~5 min of continuous unreachability at POLL_S

# Per-(table, case) budgets in seconds: 3x the slowest observed run of that shape.
#   T1-T4    Tenstorrent hosted single Galaxy, T2VA only (measured 2026-08-19).
#   TT-SJC3  the hosted 4x Galaxy deployments, every task (the h3-benchmark table).
#   BH1X     one single-host Blackhole Galaxy, the CI target. PROVISIONAL: T2VA from
#            the single-host runbook (16:9 5/10/15 s = 69.5/174.7/325.4 s warm on the
#            2026-08-13 build, ~70/125/176 s on the 09-23 build), FL2VA/REF2VA scaled
#            from the quad ratios (1.1x / 4x / 5x), with headroom for one in-request
#            shape compile (4-16 min) when the shape was not warmed at startup.
#            Regenerate from results.jsonl after the first green weekly.
TIMEOUT_TABLE_S = {
    "T1": {"SMOKE": 600, "T2VA-L": 900, "T2VA-M": 1800, "T2VA-H": 3600},
    "T2": {"SMOKE": 600, "T2VA-L": 900, "T2VA-M": 1800, "T2VA-H": 3600},
    "T3": {"SMOKE": 600, "T2VA-L": 900, "T2VA-M": 1800, "T2VA-H": 3600},
    "T4": {"SMOKE": 600, "T2VA-L": 900, "T2VA-M": 1800, "T2VA-H": 3600},
    "TT-SJC3": {
        "SMOKE": 600,
        "T2VA-L": 900,
        "T2VA-M": 1800,
        "T2VA-H": 3600,
        "FL2VA-L": 900,
        "FL2VA-M": 1200,
        "FL2VA-H": 2400,
        "REF2VA-L": 900,
        "REF2VA-M": 3600,
        "REF2VA-H": 12000,
        "SIZE-V": 1800,
        "FL2VA-L2": 900,
        "FL2VA-M2": 1600,
        "FL2VA-H1": 1800,
        "REF2VA-L10": 1800,
        "REF2VA-L15": 2700,
        "REF2VA-M5": 1800,
        "REF2VA-M15": 5400,
        "REF2VA-H5": 4000,
        "REF2VA-H10": 8000,
        "SIZE-V10": 3600,
        "SIZE-V15": 5400,
    },  # fmt: skip
    "BH1X": {
        "SMOKE": 600,
        "SMOKE-FL2VA": 600,
        "SMOKE-REF2VA": 1200,
        "T2VA-L": 600,
        "T2VA-M": 900,
        "T2VA-H": 1200,
        "FL2VA-L": 600,
        "FL2VA-M": 900,
        "FL2VA-H": 1500,
        "FL2VA-L2": 700,
        "FL2VA-M2": 1000,
        "FL2VA-H1": 1300,
        "REF2VA-L": 1200,
        "REF2VA-L10": 2000,
        "REF2VA-L15": 2800,
        "REF2VA-M5": 1400,
        "REF2VA-M": 2400,
        "REF2VA-M15": 3400,
        "REF2VA-H5": 1800,
        "REF2VA-H10": 3000,
        "REF2VA-H": 4200,
        "SIZE-V": 1500,
        "SIZE-V10": 2500,
        "SIZE-V15": 3600,
    },  # fmt: skip
}
DEFAULT_TIMEOUT_TABLE = "BH1X"


# ---------------------------------------------------------------- basic helpers
def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(out_dir(), exist_ok=True)
        with open(os.path.join(out_dir(), "run.log"), "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: str):
    with open(path) as fh:
        return json.load(fh)


def load_cases() -> dict:
    return load(CASES)


def ffmpeg_binary() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # pyright: ignore[reportMissingImports]

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional dependency
        return None


def ffprobe_binary() -> str | None:
    return shutil.which("ffprobe")


def mvhd_duration(path: str) -> float | None:
    """Real duration from the MP4 mvhd atom; the API's own `seconds` field lies."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            data = fh.read(4 << 20)
        i = data.find(b"mvhd")
        if i < 0 and size > len(data):
            with open(path, "rb") as fh:  # moov at the end (no faststart)
                fh.seek(max(0, size - (4 << 20)))
                tail = fh.read()
            j = tail.find(b"mvhd")
            if j >= 0:
                data, i = tail, j
        if i < 0:
            return None
        ver = data[i + 4]
        if ver == 1:
            ts, dur = struct.unpack(">IQ", data[i + 24 : i + 36])
        else:
            ts, dur = struct.unpack(">II", data[i + 16 : i + 24])
        return round(dur / ts, 3) if ts else None
    except (OSError, struct.error, IndexError):
        return None


def has_audio(path: str) -> bool | None:
    """True/False when a probe ran, None when neither ffprobe nor ffmpeg is available."""
    probe = ffprobe_binary()
    try:
        if probe:
            out = subprocess.run(
                [probe, "-v", "error", "-select_streams", "a", "-show_entries",
                 "stream=codec_name", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=60,
            )  # fmt: skip
            return bool(out.stdout.strip())
        ffmpeg = ffmpeg_binary()
        if ffmpeg:
            out = subprocess.run(
                [ffmpeg, "-hide_banner", "-i", path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return bool(re.search(r"Stream #\d+:\d+.*: Audio:", out.stderr))
    except (OSError, subprocess.SubprocessError):
        return None
    return None


# ---------------------------------------------------------------- error extraction
def _elide_prompt(body):
    if not isinstance(body, dict):
        return body
    out = dict(body)
    if isinstance(out.get("prompt"), str):
        out["prompt"] = f"<elided {len(out['prompt'])} chars>"
    return out


def extract_error(body) -> tuple[str, str]:
    """(message, raw) from an engine response body; the engines echo the prompt back."""
    message = None
    if isinstance(body, dict):
        err = body.get("error")
        detail = body.get("detail")
        if err == "non-json" and isinstance(body.get("body"), str):
            message = f"non-JSON response body: {body['body']}"
        elif isinstance(err, dict):
            msg = err.get("message")
            message = str(msg) if msg not in (None, "") else json.dumps(err)[:500]
        elif err not in (None, ""):
            message = err if isinstance(err, str) else json.dumps(err)[:500]
        elif detail not in (None, ""):
            message = detail if isinstance(detail, str) else json.dumps(detail)[:500]
    if message is None:
        message = json.dumps(_elide_prompt(body))[:500]
    raw_body = _elide_prompt(body) if isinstance(body, dict) else body
    return message, json.dumps(raw_body, default=str)[:2000]


# ---------------------------------------------------------------- failure classification
TRANSIENT_PATTERNS = [
    r"internal server error", r"non-json response", r"timed?\s*out", r"connection reset",
    r"connection refused", r"temporarily unavailable", r"broken pipe", r"reset by peer",
    r"\bunavailable\b", r"bad gateway", r"gateway time-?out", r"service unavailable",
]  # fmt: skip
VALIDATION_PATTERNS = [
    r"must be (at (least|most)|exactly)", r"at (least|most)\s*\d", r"\bexceeds?\b",
    r"\btoo (long|short|large|many|few|big|small)\b", r"\binvalid\b", r"\bunsupported\b",
    r"\bnot supported\b", r"\brequired\b", r"\bnot allowed\b", r"\bmaximum\b",
    r"\bminimum\b", r"\bmust not\b", r"\bmust have\b", r"\bmust be\b", r"\bnot accepted\b", r"\bunknown field",
]  # fmt: skip
CLIENT_REJECT = "client-side capability limit: "
# Error text that means the mesh is in trouble rather than a request being refused.
DEVICE_TROUBLE_RE = re.compile(
    r"\btt_throw\b|\bdevice timeout\b|\bunrecoverable\b|\btimeout:|\bhang\b|\bhung\b|"
    r"\bout of memory\b|\btt_fatal\b",
    re.IGNORECASE,
)


def _match_any(text, patterns) -> bool:
    text = text or ""
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def is_device_trouble(rec: dict) -> bool:
    return bool(DEVICE_TROUBLE_RE.search(str(rec.get("error_message") or "")))


def is_transient(rec: dict) -> bool:
    """Worth re-submitting? Never a timeout, a validation refusal, a device fault, or a
    job that is still live on the server (a retry would stack a second one)."""
    if rec.get("outcome") == "timeout":
        return False
    if (
        rec.get("job_id")
        and rec.get("last_status") not in TERMINAL_STATUSES
        and rec.get("outcome")
        not in (
            "submit_failed",
            "content_failed",
        )
    ):
        return False
    if is_device_trouble(rec):
        return False
    if rec.get("submit_http") == 0:
        return "timed out" not in str(rec.get("error_message") or "").lower()
    if rec.get("outcome") == "unreachable":
        return True
    if rec.get("outcome") == "content_failed" and rec.get("content_http") == 0:
        return True
    if rec.get("outcome") in ("failed", "submit_failed"):
        msg = rec.get("error_message") or ""
        if _match_any(msg, VALIDATION_PATTERNS):
            return False
        return _match_any(msg, TRANSIENT_PATTERNS)
    return False


def classify_failure(rec: dict) -> str | None:
    """client_capability | transport | engine_validation | engine_internal | timeout |
    content_missing | unknown, from the real error message."""
    outcome = rec.get("outcome")
    if outcome == "ok":
        return None
    if outcome == "timeout":
        return "timeout"
    if rec.get("submit_http") == 0 or outcome == "unreachable":
        return "transport"
    if outcome == "content_failed":
        return "transport" if rec.get("content_http") == 0 else "content_missing"
    if outcome in ("failed", "submit_failed"):
        msg = rec.get("error_message") or rec.get("error") or ""
        if not msg:
            return "unknown"
        if msg.startswith(CLIENT_REJECT):
            return "client_capability"
        if _match_any(msg, VALIDATION_PATTERNS):
            return "engine_validation"
        return "engine_internal"
    return "unknown"


# ---------------------------------------------------------------- assets
def asset(name: str) -> str:
    path = os.path.join(assets_dir(), name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing asset: {path}")
    return path


_prompt_cache: dict = {}


def read_prompt(name: str) -> str:
    if name not in _prompt_cache:
        with open(asset(name)) as fh:
            _prompt_cache[name] = fh.read().strip()
    return _prompt_cache[name]


def case_ref_files(case: dict) -> list:
    return (
        list(case.get("images", []))
        + list(case.get("videos", []))
        + list(case.get("audios", []))
    )


def case_assets(case: dict) -> list:
    return [case["prompt"]] + case_ref_files(case)


def read_manifest(path: str) -> dict:
    """``sha256sum`` format -> {name: sha256}."""
    entries = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) == 64:
                entries[parts[-1].lstrip("*")] = parts[0]
    return entries


def verify_assets(names, manifest_name: str = "sha256s-bundle.txt") -> list:
    """Problems with the named assets against the pack's manifest: missing files, hash
    mismatches, and a missing manifest entry. [] means every named asset is pinned."""
    root = assets_dir()
    problems = []
    manifest = {}
    for candidate in (manifest_name, "sha256s.txt"):
        path = os.path.join(root, candidate)
        if os.path.exists(path):
            manifest = read_manifest(path)
            break
    for name in sorted(set(names)):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            problems.append(f"missing asset {path}")
            continue
        want = manifest.get(name)
        if want is None:
            problems.append(f"{name}: not in the asset manifest")
        elif sha256(path) != want:
            problems.append(
                f"{name}: sha256 differs from the manifest (not the pinned file)"
            )
    return problems


# ---------------------------------------------------------------- per-case timeout
def case_cost(case: dict) -> int:
    return case["duration_s"] * case["steps"] * (1 + len(case_ref_files(case)))


def case_timeout_s(case: dict, table: str = DEFAULT_TIMEOUT_TABLE) -> int:
    """Budget for one generation: ``H3_TIMEOUT`` overrides; else the table, the TT-SJC3
    row for a case the table lacks, else the cost formula clamped to the floor/ceiling."""
    cid = case["id"]
    override = os.environ.get("H3_TIMEOUT")
    if override:
        return int(override)
    base = TIMEOUT_TABLE_S.get(table, {}).get(cid)
    source = table
    if base is None:
        base = TIMEOUT_TABLE_S["TT-SJC3"].get(cid)
        source = "TT-SJC3"
    if base is None:
        est = case_cost(case) * COST_SECONDS_PER_UNIT * 3
        base = int(min(TIMEOUT_CEIL_S, max(TIMEOUT_FLOOR_S, est)))
        source = "cost formula"
    budget = int(min(TIMEOUT_CEIL_S, max(TIMEOUT_FLOOR_S, base)))
    log(f"[timeout] {cid}: {budget}s ({source})")
    return budget


def ensure_dirs() -> None:
    for d in (out_dir(), clips_dir(), logs_dir()):
        Path(d).mkdir(parents=True, exist_ok=True)
