# Remote guarded FillPad operation review

Verdict: **clean-pass for the bounded remote operation gate**. This supports proceeding to the planned serving integration test. It does not establish model startup, serving performance, trace behavior, quality, or memory fit.

## Required Work

None within the frozen operation contract.

## Evidence

- The frozen `EXPERIMENT_PLAN.json` asks for the original-stride control against guarded FillPad in image `sha256:0041fbc0...17c3b6eef`, on four devices, with 184 eligible and four fallback cases per arm. Run `39602265481` completed under source commit `847c1a824c611f12f3dcc00f57ef68b9c2bbc51e`. I checked the native JSON rather than relying on `RESULT.md`: each arm has exactly 184 eligible rows and four fallback rows. Both widths 1344/5376, all 31 nonzero remainders at physical 32, selected boundary buckets 128/1056/6656/6688/8320, alternating backing buffers, and two iterations are present.
- I independently loaded both restored `raw-operation-controls.pt` archives in a CPU-only Docker process, with network disabled and artifacts mounted read-only. For every eligible row, I reconstructed the BF16 pattern and zero mask, verified the recorded full-output digest on all four devices, and compared the 368 saved last-tile/body-row tensors to the expected slices. I checked all four fallback tensors against independently reconstructed nonzero-fill or unaligned-width expectations. Each arm has exactly 372 raw keys; no mismatch occurred, and corresponding control/candidate tensors are exactly equal. Both raw files hash to `05cc58cd3433c0b67d46c40a60982cfcc2cb47d88694b803686e6cdb13c786f4`. `raw-output-extraction.json` ties their 169,662,653-byte files to original artifact archive `12355287737.zip` and records its SHA-256; no device rerun was used.
- The native probe itself checks full backing outputs on all four devices after each operation, finite deterministic masks through exact tensor equality, in-place buffer identity, logical and padded shapes, and a metadata-only reshape. The candidate's no-miss guard is active on 172 eligible rows (80 changing-height cold hits and 92 warm hits); all 172 have unchanged program counts. The control guards all 92 warm rows with no miss. Cold program increments are 92 control versus 12 candidate.
- The build status confirms image, compiler, unity, matmul-source, dependency, and payload checks; `installed-source-pins.json` covers 19 original files. The unchanged physical-stride control binary hashes to `fb3478...`; the seven-file guarded candidate overlay builds to `01d961...`, matching the local candidate hash. I independently hashed both collected binaries and checked the per-arm input manifests. Actual loaded `_ttnncpp.so` paths and hashes match each arm's policy; both load remote `_ttnn.so` `7ce78c...` and `libtt_metal.so` `aa8b7d...`. The candidate command mounts all seven read-only source overlays and sets the canonicalization flag; the control flag is off. The copied probe matches the reviewed source.
- Four authoritative before/after ownership guard logs end in three host-PID-helper samples each, with zero foreign owners and zero visibility errors; three telemetry processes are classified as allowed. Both owned containers report completed cleanup with no errors. The wrapper checks the container ownership label before cleanup and stops only its named containers.
- Actual machine time was 214 s (3 min 34 s), below the frozen 5–18 min estimate. Wrapper time was 129.88 s; build time 26.17 s versus estimated 30–120 s; native arm time 49.47 + 33.61 = 83.08 s versus estimated 120–360 s. These intervals are operation and orchestration times, not model startup. Cold operation host wall was 18.5523 s control versus 3.0714 s candidate (6.04× in this matched microbenchmark). Warm 92-call wall increased from 0.005797 s to 0.014728 s, an 8.931 ms aggregate regression (2.54×); this cost must remain visible in serving interpretation.

## Hard-Check Gaps

None for this operation gate. Full-output values are checked by the native four-device assertions and full digests; the persisted raw tensor archive retains last-tile/body-row samples plus full fallback tensors, as specified. Sparse DRAM samples occur once per physical bucket after backing allocation and do not measure a continuous peak or combined model/KV/history reserve.

## Anomaly Ledger

- **Warm host wall regression.** Evidence: 5.797 ms control versus 14.728 ms candidate over 92 calls. Affected path: warm FillPad calls. Control: matched shapes and run image; cold path has 92 versus 12 program increments. Investigation: recalculated sums directly from all 184 eligible rows per arm. Resolution: controlled as a measured tradeoff for this operation gate; serving tests must measure its effect in context.
- **Generic collection omitted large raw archives.** Evidence: both 169,662,653-byte files were restored from verified original archive `12355287737.zip`, with extraction provenance and matching file hashes. Affected path: auditability. Investigation: independently reloaded and checked all 372 keys per arm. Resolution: fixed.
- **Initial host process visibility was incomplete.** Evidence: each ownership log contains an initial limited-view record followed by the authoritative host-PID-helper record. Affected path: hardware ownership. Investigation: inspected final three samples in each of four logs. Resolution: controlled; final checks show no foreign owner or visibility error.
- **Remote timing differs from local timing and frozen estimate.** Evidence: remote cold 18.5523→3.0714 s, whereas the frozen plan cites local 55.005→29.671 s; machine 214 s is below estimated 5–18 min. Affected path: latency extrapolation. Investigation: reconciled native, wrapper, build, and job intervals. Resolution: controlled by reporting remote operation timing only; no startup ratio inferred.

## Scope Inspected

- Goal: frozen remote `EXPERIMENT_PLAN.json` and `REVIEW_REQUEST.md`; reviewer guidance from `stage-review/SKILL.md`, narrowed to this explicitly assigned operation gate.
- Artifacts: build status and binaries, `status.json`, both native results, raw tensor archives, comparison/raw-checks, runtime policies, input manifests, loaded-runtime proofs, commands, launch statuses, extraction proof, and ownership logs under `/data/mvasiljev/gemma-fillpad-remote-ci/ci-39602265481/`.
- Code: `probe.py`, `check_raw.py`, `compare_metadata.py`, `run_device.py`, `build_remote.py`, and `owned_run.py` in the `gemma-fillpad-device-wrapper` source.
- Methods: read-only JSON/hash/coverage checks and independent CPU-only raw-tensor reconstruction. No TT hardware, server, or implementation edit was used in this review.

## Residual Risk

The accepted result proves guarded FillPad operation correctness and cache reuse at the tested shapes on the pinned remote image. Serving integration still has to validate actual model startup, kernel installation before import, target-bound JIT cache behavior, model numerics, traces, and the net cold/warm effect.
