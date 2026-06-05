# QB2 (2× P300) — missing inter-card ethernet links

**Date:** 2026-06-05 · **Affected host:** `qb2-120-p01t06` (10.32.48.16) · **Known-good comparison:** `qb2-120-p01t05` (10.32.48.15)

## TL;DR

On `p01t06` the two P300 cards have **no ethernet links between them**. On-card links are fine (chip 0↔1, chip 2↔3), but the two **cross-card** links present on the working box (`chip1↔chip2` and `chip0↔chip3`) are missing, so the 4 chips form two isolated 2-chip pairs instead of one 4-chip mesh. This blocks any 4-chip workload. Firmware and config are identical to the working box, and `tt-smi -s` looks healthy on both — only `system_health` reveals it. **Likely a missing/unseated/faulty card-to-card connection (or eth links not training) on `p01t06`.**

## Evidence (`system_health` per-channel link state)

Working `p01t05` — connected ring, every chip has 2 links:
```
0↔1, 1↔2, 2↔3, 3↔0   (cross-card links chip1↔chip2 and chip0↔chip3 are UP)
```
Affected `p01t06` — two isolated pairs, every chip has 1 link:
```
0↔1, 2↔3 only   (cross-card chip1↔chip2 and chip0↔chip3 report DOWN/unconnected)
```
Downstream effect when running a 4-chip model on `p01t06`:
```
TT_FATAL (topology_mapper.cpp:527): Graph specified in MGD could not fit in the discovered physical topology. Inter-mesh mapping failed ... (physical mesh degree {1:4}, needs {2:4})
```

## Repeatable check (any box)

```
<tt-metal>/build_Release/tools/umd/system_health 2>/dev/null | grep -E "Chip:|link UP"
```
Healthy = a `link UP ... connected to chip` that crosses the two cards (a chip in {0,1} linked to a chip in {2,3}). On `p01t06` there are none. Note: `tt-smi -s` does **not** show this (passes on both).

## Confirmed NOT the cause

- Firmware: identical (both `fw_bundle 19.8.1.0`, `ETH_FW_VERSION 0x10a01`).
- Config/software: identical; both cards individually healthy (on-card links up).
- Not transient: persists after `tt-smi -r` (tried with both tt-smi 4.1.0 and 5.2.0) and after running a fabric workload.

## Requested action

Check the card-to-card connectivity (cable/bridge/backplane) between the two P300 cards on `p01t06` that should carry the `chip1↔chip2` and `chip0↔chip3` links. If physically correct and seated, escalate as a card-to-card eth link hardware fault.

## Versions

`qb2-120-p01t06` (10.32.48.16, affected) & `qb2-120-p01t05` (10.32.48.15, working): TT-KMD 2.8.0, tt-smi 4.1.0, tt_umd 0.9.4, fw_bundle 19.8.1.0.
