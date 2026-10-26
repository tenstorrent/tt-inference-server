# Goal-12 suite metrics validation

Enable `record_suite_metrics: true` (JSON boolean) in
`reference_config/evals/qwen38_goal12.json` on every arm being compared.
The default is false. Collection additionally requires the exact Qwen model and
membership of the suite in controls.evals. This change does not enable the flag
in existing controls, modify serving settings, or dispatch jobs.

The common helper writes `<task>_suite_metrics.json` beside Harbor's attempt
receipt and `suite_metrics.json` inside each GPQA eval-attempt directory. It
scrapes twice, outside both the attempt timer and GPQA's scorer timer. Errors
produce compact missing reasons and do not change the harness outcome. It does
not poll, profile devices, or add work to request/decode processing. HTTP reads
are bounded (2 MiB maximum, 5-second socket timeout plus a checked 5-second read
deadline); redirects are disabled to avoid forwarding authorization elsewhere.

Counter deltas describe the server interval. Nonzero running/waiting gauges at
either boundary prevent clean attribution to that suite. An API process-start
guard is not engine identity proof. Per-counter creation timestamps, when
present, detect counter recreation; decreases and missing/reset series have
null deltas. A restart that preserves API-side counters is not detectable from
these snapshots alone. Independent engine logs remain necessary for that claim.
Only whitelisted numeric metrics and bounded labels are retained, without raw
exposition, auth headers, exception text, or endpoint URLs.

Validation: 34 host tests passed using:
`uv run --no-project --with pytest --with requests python -m pytest -q tests/llm_module/test_suite_metrics.py tests/llm_module/test_attempt_timing.py`.
Tests cover guards, malformed data, defaults, outcome preservation, timeout
failure, response size bounds, credentials, timer ordering, and actual call-site
nesting. The fake-clock test measures 3 seconds inside the harness while two
5-second scrape simulations raise total wall time to 13 seconds.

An offline parse of the root-owned local-release-shared256-b32-headline
`prometheus-idle.prom` succeeded with 29 selected series. Source SHA256:
`2eeab9a84fa813121176ef9a5ee233fbf093850f3b7aa1c98c4ab1ecc5bd773c`.
That actual exposition uses `finished_reason`, including `repetition`; the parser
supports those bounded labels. This single snapshot is parser compatibility
proof, not a suite delta or proof of remote deployment identity. The collector
agent made no hardware calls or HTTP requests.
