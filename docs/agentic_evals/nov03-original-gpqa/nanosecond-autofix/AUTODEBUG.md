# CI41582292198 pre-request timestamp failure

Failing contract: parse the actual Docker timestamp/current EngineCore PID while retaining exact generator/full64/policy and engine-generation checks. Native requests must not begin before this check.

Evidence from independent collector: owned server reached health at05:22:30, then checkpoint preflight failed in qwen38_selected_precision.py:45. Docker supplied `2026-11-03T05:15:13.679142093Z`; Python datetime.fromisoformat rejected the nine-digit fractional timestamp. Actual local CPython3.10.22 reproduces the identical ValueError before any runtime/device import. Python3.12 accepts this form, explaining passing newer-host tests. Artifact: nov03-gpqa-original-rerun-proposal/collected/13205734959.zip checkpoint-preflight.json and run log.

Cause: cross-Python ISO parser compatibility, not a precision mismatch or model failure. Earliest inconsistent operation is timestamp parsing. Minimal candidate: validate complete timezone-aware RFC3339 form, normalize fractional seconds to datetime's six-digit microsecond representation (same effective precision as Python3.12), then parse. Keep all other marker/identity checks unchanged. Reject malformed/missing-timezone timestamps. Discriminator: reproduce on actual3.10, exercise nanoseconds and all prior malformed/wrong-policy/stale-engine guards.
