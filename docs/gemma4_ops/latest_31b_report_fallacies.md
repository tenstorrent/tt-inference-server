# Fallacies in the Latest 31B Release Report (gemma-4-31b-it, 2026-07-01_21-26-18)

Source: `workflow_logs/reports_output/release/report_id_tt-transformers_gemma-4-31b-it_p300x2_2026-07-01_21-26-18.md`

This report ran on a real docker-managed server and produced tables, so it
*looks* trustworthy. But several numbers are misleading or outright artifacts.
Do not quote from it without the caveats below.

## 1. `score: 100.00` on GPQA Diamond is not a real accuracy
- GPQA Diamond is PhD-level science; frontier GPU models land ~50–60%. A
  **100.00** is implausible and is an artifact of the `ci-nightly` sample limit
  (a handful of questions) with `accuracy_check: N/A`, `gpu_reference_score:
  None`, `published_score: None`.
- Nothing calibrates it. Treat 100.00 as "the 1–2 sampled items happened to be
  right", **not** as model accuracy.

## 2. `terminal_bench_2 = 0.00` and `swe_bench_verified = 0.00` are false negatives
- These 0.00s are **infrastructure failures, not model quality**. In this
  report's era SWE-Bench failed because the server wasn't launched with
  `--enable-auto-tool-choice` + `--tool-call-parser gemma4` (mini-swe-agent
  sends `tool_choice:"auto"` → 400), and terminal-bench similarly didn't
  execute the agent loop correctly.
- Reporting them as an "accuracy score" of 0.00 conflates "the eval harness
  couldn't drive the model" with "the model scored zero". (This is exactly what
  the `375df057` + `enable-auto-tool-choice` spec change fixes.)

## 3. Every `accuracy_check` is `N/A` — the whole accuracy table is uncalibrated
- `gpu_reference_score: None` and `published_score: None` for all four tasks, so
  `ratio_to_reference` / `ratio_to_published` are `N/A`.
- Without a reference or tolerance check, `ifeval 89.09` / `gpqa 100.00` are raw
  numbers with no pass/fail meaning. The report's own note says the accuracy
  check depends on a GPU reference or published score — neither exists here.

## 4. `ifeval` is present but was never requested
- The requested accuracy set is GPQA Diamond, Terminal-Bench-2.0, SWE-Bench
  Verified. `ifeval` (89.09) is a default task that slipped in. Fine as a
  smoke signal, but don't present it as part of the intended eval suite.

## 5. Self-contradictory acceptance FAIL
- Acceptance fails on `benchmarks_summary.0.target_checks: Missing target_checks`
  — yet the report also states **"No performance targets defined for this model
  and device combination."**
- So it fails for missing something that, by design (EXPERIMENTAL, no enforced
  tiers), does not exist. The FAIL is circular and not a real signal.

## 6. `test_non_uniform_seeding` FAIL is a timeout, not a correctness bug
- Message: `Read timed out (read timeout=30)`. At ~127 ms/token and ~8 tok/s,
  a longer generation can blow the 30 s client timeout. This is a flaky
  latency/infra artifact, **not** a seeding-determinism defect. Presenting it in
  the parameter-conformance FAIL column is misleading (the other 24 param tests
  passed).

## 7. Benchmark sweep stops at ISL 4096 — no high-context data
- 31B on QB2 is servable to the 32768 prefill bucket; the whole point of the QB2
  work is high-ISL behavior. The sweep tops out at **ISL 4096**, so the report
  says nothing about 8K/16K/32K — the regime that actually matters here.

## 8. Means/percentiles from N=2–8 are statistically weak
- Row sample sizes: N = 8, 4, 4, 4, **2**. The `4096/128` row reports a P99 TTFT
  of 10786 ms from **N=2** — a P99 from two samples is meaningless.
- That same row's mean TTFT (6409.9 ms) is a ~6.5× jump over `2048/128`
  (982.5 ms) for only 2× the input — an anomaly (bucket padding / first-request
  warmup) that the report does not flag. A naive reader would take 6.4 s TTFT as
  representative when it is likely an outlier.

## 9. Null provenance despite a real image
- `tt_metal_commit: null`, `vllm_commit: null` even though `server_mode: docker`.
  The image tag carries the commits, but the dedicated fields being null makes
  the report look unattributable.
- `Test Timestamp: N/A` in the LLM API test metadata.

## What's actually trustworthy in this report
- The **single-user latency shape** at low ISL (TTFT ~308 ms at 128/128,
  TPOT ~125 ms, ~7.9 tok/s/user) is consistent across rows and matches
  expectations — those numbers are usable.
- The parameter-conformance tests (24/25 passing, the one FAIL being a timeout)
  indicate the sampling/penalty/stop/logprobs paths are healthy.
