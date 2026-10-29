# Stage Review

Verdict: clean-pass

Reviewed 2026-10-29 as the independent reviewer for the bounded, unscored Qwen/Qwen3.8-27B B1 serving comparison. This verdict covers compact recurrence off/on on the retained full64, physical B1, native262144 serving pair. It does not close a new stage 9 integration, release, accuracy benchmark, concurrent-capacity gate, or SWE evaluation.

## Required Work

- None for this bounded check.

## Other Concerns

- Raw client events independently reproduce the reported three-repeat medians:

  | Input/output | Control decode TSU | Compact decode TSU | Gain | Control/compact TTFT ms |
  |---|---:|---:|---:|---:|
  | 128/128 | 37.991704 | 43.109293 | 13.470282% | 113.978937 / 115.274044 |
  | 4096/128 | 37.436828 | 42.404697 | 13.270006% | 420.004256 / 418.388795 |
  | 129/128 | 37.988560 | 43.109296 | 13.479678% | 115.126697 / 118.019816 |

  All nine measured candidate samples exceed 40 TSU. The primary end-to-end output-throughput median is separately 33.574328 to 37.501158 tokens/s. TTFT differences are small and mixed; no general prefill improvement is established.
- All 24 performance requests have 128 one-ID SSE events, matching usage, zero reported cached prompt tokens, and correct 127-interval arithmetic. All 12 input records, 12 performance trajectories/text/usage/generation settings, and six qualitative token trajectories/text/finish reasons match across arms. Warmup and measured trajectories also match within each arm.
- Actual runtime specs differ only in the compact flag. Both logs attest all 64 layers, resident B1, the selected precision, native262144 context, 72 cache blocks and 129 prepared buckets. The candidate adds 98,304 normal DRAM bytes/device at the preparation checkpoint. Preparation takes 1278.317/1289.374 seconds and is excluded from warmed measurements.

## Hard-Check Gaps

- The stable two-file exporter remains incomplete in both arms. The explicitly accepted bounded-check limitation is supported by the missing HTTP-file receipts, retained worker/client artifacts, and complete owned-resource cleanup. This review does not declare an exporter pass.
- Captures remain three, decode uses MeshTraceId(0), and all worker counters are monotonic across each 2561-row log, with one PID, B1, zero host-logit reads and zero retirements throughout the 18-request workload. The retained worker logs end at the last request release; they do not prove shutdown trace destruction. Cleanup is separately attested.
- This check does not provide new maximum-context quality, HF accuracy, cached/mixed concurrent serving, larger-batch, API conformance or release evidence. Those are outside its stated contract, not newly passed gates.

## Anomaly Ledger

- Observed anomaly: two-file stable export fails. Evidence: both final-audit helper receipts report FileNotFoundError for their exact qwen38-http filename. Affected path: post-run audit export. Control or comparison: identical failure in both arms. Likely subsystem: middleware records only /v1/chat/completions, while this probe uses /v1/completions. Investigation performed: inspected middleware, helper receipts, worker/client artifacts and cleanup reports; verified all eight comparison artifact hashes. Resolution: controlled for the expressly bounded comparison; exporter status stays incomplete.
- Observed anomaly: 2506 sampled/read results versus 2503 returned IDs. Evidence: per-request worker deltas; returned EOS-ended lists already include EOS248046. Affected path: asynchronous serving completion. Control or comparison: both arms have one extra in-flight result on each of the same three EOS-ended quality calls; all timing calls have exactly 128 samples/127 model replays. Likely subsystem: async lookahead and late-output suppression. Investigation performed: inspected actual installed model_runner source and independently confirmed AST identity of build_model_input, _finish_async_decode and _apply_sampled_tokens_to_state against retained upstream1bad source; inspected upstream async suppression code. Resolution: controlled and qualified. The separately installed async module and discarded token values were not independently captured.
- Observed anomaly: forced-length timing text includes newline and role/thinking tails. Evidence: actual 128/4096/129 performance text, with the 129 case continuing into role text. Affected path: ignore_eos timing stress. Control or comparison: exact trajectories/text in all repeats and both arms; six separate normal-EOS native-chat requests were read. Likely subsystem: continuation after natural stopping points. Investigation performed: verified generation settings and read all outputs. Resolution: controlled as timing-only evidence. Quality outputs comprise coherent haiku/French/Fibonacci EOS answers and three coherent 256-token capped prefixes. Restricted execution of generated Fibonacci passes -1, 0, 1, 2 and 8.
- Observed anomaly: earlier cancelled startup had failed cleanup and subsequent retry-owner evidence. Evidence: cancelled-run cleanup status failed/claimed-clean false; retained same-daemon absence proof and later empty root-fuser/four-device-list proof40925849632. Affected path: superseded startup. Control or comparison: replacement arms each show one adapter/preparation sequence and complete owned cleanup. Likely subsystem: cancellation/startup retry race; exact orphan disappearance cause remains unproven. Investigation performed: inspected failed cleanup and subsequent exact-owner/device-user proofs plus single-attempt source guard. Resolution: controlled for the replacement pair; failed startup contributes no performance data.

## Scope Inspected

- Skills: installed stage-review, model-bringup startup, target-workload and qualitative-check contracts.
- Source: frozen Metal f023ac0e0bcb70118f09eb994128e65dfdbc7411 model runtime, compact B1/T1/unmasked/nonprefill guard, generator/serving/sampler boundaries; TTI control df02557da5ae4ad2fe17d5958c6e6cf5cb852c26 and candidate ef8461f67c628f73441cf76d61cf37a77d3b8d95; probe, CPU test source, startup guard, middleware and final exporter. The complete measured TTI tree diff contains only the compact flag.
- Artifacts: serving-ci control40926118763 and candidate40931138892 raw calls/inputs/results, worker logs, server/runtime specs, final audits and cleanup; build40923312357 immutable-image/source/readback manifests; MATCHED_SERVING_COMPARISON.json, corrected CONTROL_REVIEW.json, prompt-format metadata and operational proofs. All 33 model-runtime manifest hashes match frozen git. Installed readback hashes verify; Sampling1D retains max_top_k32 and TP4 width128.
- Documents: isolated report reference_config/evals/qwen38_b1_serving_results.md and .json, plus prior optimized_full_model README/review. All 21 shared compact JSON sections match the independently checked comparison.
- Commands: read-only source/log inspection, git diff/show, stdlib JSON/hash/timing/counter arithmetic, AST comparison, and restricted CPU evaluation of the generated Fibonacci function. No TTNN import, device/server execution, reset, CI dispatch or implementation edit occurred. This review file is the sole reviewer file write.

## Residual Risk

- Three warmed repeats per shape from one matched host campaign do not establish broad production performance. The three capped quality outputs establish matching coherent prefixes, not complete answers or full accuracy.
- The immutable image/readback identifies vLLM0.30.0+empty, plugin0.1.0 and installed model_runner SHA256 f09e7719214d6e3e1d2b82e4abc3534b9779a1358949a9adda0625f2d1c3422c. Its hybrid/prefill patches mean dispatch1bad is not a pristine installed-source identity.
- Historical native40.673091 TSU uses a different compiled backend, prompt/trajectory, cache ownership and timing boundary. The serving/native numerical gap cannot isolate serving overhead or an additional optimization gain. The matched serving pair supplies the supported 13.270006% primary delta.
