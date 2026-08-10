# Milestone-0 — AIPerf benchmark path end-to-end verification (Blackhole Galaxy)

**Readiness item:** 7.1 — Verify — owner: Benchmark tooling
**Issue:** [llm-gauntlet#72](https://github.com/tenstorrent/llm-gauntlet/issues/72) · Parent: [#56](https://github.com/tenstorrent/llm-gauntlet/issues/56)
**RFP references:** requirements §G.1 (AIPerf is the pinned measurement tool), readiness §7.1. Interacts with the device config ([§5.4 / #68](m0-blackhole-galaxy-device-config.md)) and the scaling-quality sweep rule ([§5.7 / #64](m0-scaling-quality-sweep-coverage.md)).

## Problem

`--tools aiperf` is not the default, and while the AIPerf plumbing is well-factored
(`WorkflowVenvType.LLM_AIPERF`, `requirements/llm-aiperf.txt`, Python 3.11), what had
**not** been confirmed is an AIPerf benchmark run against a model served on a 32-chip
Blackhole Galaxy. The RFP pins AIPerf and every Appendix B target is set with it, so one
confirming end-to-end run is a precondition for trusting any of those numbers.

## Approach — mock server stand-in

Real Galaxy hardware is not available in this environment, so a **mock OpenAI-compatible
server** stands in for the served model, exactly as suggested on the issue. We use
[`llm-d-inference-sim`](https://github.com/llm-d/llm-d-inference-sim) — a lightweight vLLM
simulator that mimics vLLM's streaming, latency (TTFT / inter-token), and OpenAI API
without a GPU. The MacOS-aarch64 build issue noted on the issue is sidestepped by running
the **prebuilt Linux image via Docker** (`ghcr.io/llm-d/llm-d-inference-sim:v0.9.0`); no Go
toolchain is needed.

The simulator is seeded with Blackhole-Galaxy-representative latency
(`--time-to-first-token`, `--inter-token-latency`, `--max-num-seqs 128`,
`--max-model-len 262144`) so the run produces realistic metrics that flow through the real
grading logic. Seed these from the Appendix B AIPerf target sheet to make a tier pass/fail
as intended.

### What is real vs mocked

| Stage | Real / mocked |
| ----- | ------------- |
| `AIPerfDriver` — builds and runs `python -m aiperf profile` in the `LLM_AIPERF` venv | **Real** (the exact code path used on hardware) |
| Raw `profile_export_aiperf.json` artifacts | **Real** (produced by AIPerf 0.5.0) |
| `AIPerfParser.parse` → `Block` | **Real** |
| `apply_target_checks` — tiered functional/complete/target grading | **Real** |
| `report_module.generate_report` → markdown + JSON | **Real** |
| Inference server | **Mocked** (`llm-d-inference-sim`) |
| Tokenizer | Small real HF tokenizer (`Qwen/Qwen2.5-0.5B-Instruct`) stands in for the greenfield model tokenizer, which AIPerf needs to synthesize prompts and count tokens |

Only the server and the tokenizer are stand-ins; everything from driver invocation through
parsing, grading, and report rendering is the code that runs against a real Galaxy vLLM
endpoint.

## Confirming run

Reproduce with a single command (starts the mock, builds the venv, runs the path, tears down):

```bash
scripts/verify/run_aiperf_bh_galaxy_e2e.sh
```

A representative sweep of **6 points** — ISLs {128, 512, 1024} × concurrency {1, 8}, OSL 128
— was run. The sweep honors the §5.7 three-point rule: each graded concurrency level carries
≥3 distinct input lengths. Seeded latency: TTFT ≈ 60 ms, inter-token ≈ 22 ms.

Result: **6/6 sweep points ran, parsed, graded, and rendered — PASS.**

| ISL | OSL | conc | TTFT (ms) | TPOT (ms) | Tput User | Tput Decode | Target |
| --- | --- | ---- | --------- | --------- | --------- | ----------- | ------ |
| 128 | 128 | 1 | 56.7 | 21.0 | 47.8 | 46.6 | ✅ PASS |
| 512 | 128 | 1 | 59.7 | 21.3 | 46.9 | 45.6 | ✅ PASS |
| 1024 | 128 | 1 | 63.6 | 19.0 | 53.3 | 47.1 | ✅ PASS |
| 128 | 128 | 8 | 67.0 | 20.5 | 49.2 | 167.6 | ✅ PASS |
| 512 | 128 | 8 | 69.1 | 20.7 | 48.6 | 174.9 | ✅ PASS |
| 1024 | 128 | 8 | 71.3 | 19.8 | 50.8 | 121.5 | ✅ PASS |

Decode throughput scales with concurrency (≈46 tok/s at conc 1 → ≈120–175 tok/s at conc 8),
and the seeded TTFT/ITL come through the parser unchanged, confirming metric fidelity. The
tiered grader emitted functional/complete/target checks for every point.

Artifacts produced (under `workflow_logs/aiperf_bh_galaxy_e2e/`, git-ignored):

- `llm/aiperf_artifacts/bench_<isl>_<osl>_<conc>_n<n>/profile_export_aiperf.json` — raw AIPerf output (per point)
- `reports/report_<id>.md` + `reports/data/report_data_<id>.json` — rendered benchmark + tiered target-check tables
- `e2e_summary.json` — machine-readable pass/fail summary

## Confirming run — real DeepSeek-V4-Flash targets (Appendix B sheet)

The generic run above proves the path; this second run grades against the **actual
DeepSeek-V4-Flash Blackhole-Galaxy targets** from the Appendix B sheet, seeding the mock to
the target tier so the grade is meaningful. The sheet's system numbers reduce to two
concurrency-independent per-request bars (kept in
[`scripts/verify/targets_deepseek_v4_flash.json`](../../scripts/verify/targets_deepseek_v4_flash.json)):

| Metric | `target` (full) | `functional` (downrated) | Derivation |
| ------ | --------------- | ------------------------ | ---------- |
| Interactivity `tput_user` (t/s/u) | 100 | 25 | sheet "Target Interactivity" / "Downrated" (×0.25) |
| TTFT (ms) | 1410 | 2820 | `target_mean_isl × max_num_seqs / prefill_tput` = `8192 × 64 / 371,835` (full) and `/ 185,918` (downrated ×0.5) |

System context (informational, **left ungraded** so the bar does not move with the sweep
point's concurrency): 32× P150, aggregate decode ≈ 6,413 t/s, prefill 371,835 t/s,
max-num-seqs 64, target mean ISL 8,192. `complete` is omitted — the sheet defines only the
full and downrated bars.

Reproduce:

```bash
SERVED_MODEL="tenstorrent/DeepSeek-V4-Flash-0731-mock" \
  TTFT="1410ms" ITL="10ms" ISLS="4096,8192,16384" CONCURRENCIES="1,64" \
  OSL="128" NUM_PROMPTS="64" \
  TARGETS_JSON="scripts/verify/targets_deepseek_v4_flash.json" \
  OUTPUT_DIR="workflow_logs/aiperf_deepseek_v4_flash_e2e" \
  scripts/verify/run_aiperf_bh_galaxy_e2e.sh
```

The served name is a **mock stand-in** (`…-mock`) so the simulator uses its built-in
tokenizer; a real `org/model` name makes the sim require a render sidecar. The sweep centers
on the sheet's 8,192 mean ISL (`{4096, 8192, 16384}`, ≥3 input lengths per level for the §5.7
fit) at concurrency `{1, 64}` (64 = the sheet's max-num-seqs). Seeded TTFT 1410 ms, ITL 10 ms.

Result: **6/6 PASS at the `target` tier** — interactivity holds ≈108–112 t/s/u and TTFT
≈1410–1473 ms from concurrency 1 through 64:

| ISL | conc | TTFT (ms) | TPOT (ms) | Tput User | Target |
| --- | ---- | --------- | --------- | --------- | ------ |
| 4096 | 1 | 1450.1 | 8.8 | 111.2 | ✅ PASS |
| 8192 | 1 | 1399.3 | 9.0 | 111.6 | ✅ PASS |
| 16384 | 1 | 1428.1 | 9.1 | 111.4 | ✅ PASS |
| 4096 | 64 | 1473.2 | 9.3 | 109.8 | ✅ PASS |
| 8192 | 64 | 1410.8 | 9.0 | 110.1 | ✅ PASS |
| 16384 | 64 | 1449.1 | 9.4 | 108.3 | ✅ PASS |

The seeded latency comes through the real parser and tiered grader unchanged, and the
sheet-derived targets grade cleanly — confirming the DeepSeek-V4-Flash numbers are directly
consumable by the `--tools aiperf` path once real hardware measurements exist.

## Reproducible tooling (added by this issue)

- `scripts/verify/run_aiperf_bh_galaxy_e2e.sh` — starts the mock (seeded latency), builds the
  `LLM_AIPERF` venv from `requirements/llm-aiperf.txt`, runs the harness, tears down. Every
  knob (port, model, tokenizer, TTFT/ITL, targets) is overridable via env vars.
- `scripts/verify/aiperf_bh_galaxy_e2e.py` — drives the real
  `AIPerfDriver → AIPerfParser → apply_target_checks → report_module` path over a
  Blackhole-Galaxy-shaped sweep and writes the report + summary. Targets are overridable via
  `--targets-json`; the sweep grid via `--isls` / `--concurrencies` / `--osl`.
- `scripts/verify/targets_deepseek_v4_flash.json` — the DeepSeek-V4-Flash tiered targets
  derived from the Appendix B sheet, with the derivation recorded inline.

## Remaining gap for real hardware

This verifies the **software path**. The final confirming run on real silicon is:

```bash
python run.py --model <bh-galaxy-model> --device blackhole_galaxy \
  --workflow benchmarks --tools aiperf
# or, against an already-running server:
python run_workflows.py --model <bh-galaxy-model> --device blackhole_galaxy \
  --workflow benchmarks --tools aiperf --service-port 8000
```

against a real 32-chip Blackhole Galaxy vLLM endpoint, with targets and latency drawn from
the Appendix B AIPerf sheet. The mock run above gives high confidence that the tooling,
artifact shape, parser, grader, and report are correct before hardware time is spent.
