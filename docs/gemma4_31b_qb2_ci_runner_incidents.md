# QB2 CI runner incidents during Gemma 4 31B agentic evaluations

Issue draft; observed 9–10 October 2026 UTC. The affected GitHub Actions pool was `bh-qb-ge` in `tenstorrent/tt-agentic-bringup-qb2`. Gemma used the `gemma4-31b-qb2` implementation on P300X2 QB2 hosts. This document concerns runner availability and startup; the failed starts below produced **no benchmark cases or model scores**.

## Summary

Three runner names needed temporary exclusion while long Terminal-Bench and SWE-bench jobs were queued:

| Runner | Observed problem | Representative evidence |
| --- | --- | --- |
| `120-qb2-p04t07` (`qb2-120-p04t07`) | A stopped Docker container was present and was not owned by the Gemma job. The CI ownership guard refused cleanup and hardware execution. | [First blocked Terminal dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37962790734): `Host contains unowned containers; refusing cleanup or hardware execution`. A [later C4 dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38010281587) was canceled during setup after landing here. |
| `120-qb2-p04t05` (`qb2-120-p04t05`) | Four-chip fabric mapping failed at model startup. Earlier jobs on this host also emitted AICLK warnings: expected 1350 MHz, observed 800 MHz, clamped by max-arbiter index 5. | [C8 Terminal startup failure](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38001414052); a [retry](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38010505443) was canceled during setup. The [C4 SWE run](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37963243142) contains the clock warning. |
| `qb2-120-p05t06` | A later four-chip fabric mapping failure occurred at model startup, after this runner had completed other Gemma work. Ownership and container checks passed. | [C4/eight-agent Terminal startup failure](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38043892436); its [exact-name hold](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38045128181) followed. |

The two fabric failures reported `TT_FATAL` at `tt_metal/fabric/topology_mapper.cpp:556`, `mapping_result.success`. The server log says the logical four-node mesh could not be mapped onto the discovered four-ASIC physical mesh: `TopologyMappingEnumerationSession: no new mapping found (all solutions exhausted or excluded)`. The logs alone do not establish whether the cause is hardware connectivity, discovery, or a mapping/software interaction. The `p04t05` warning suggests an additional clock or firmware state issue; it does not prove that the clock warning caused the fabric failure. `p05t06` is an intermittent observation, not evidence that every run on that host failed.

## Why the `hold-*` jobs were created

The shared `bh-qb-ge` label let GitHub assign a queued multi-hour model job to any eligible QB2 runner. A job sent to `p04t07` could not pass the unowned-container guard; a job sent to a runner with a fabric startup failure could spend setup time and then fail before evaluation. Repeating dispatches without changing eligibility risked wasting runner time, delaying the healthy-host jobs, and confusing infrastructure failures with model results.

The [Gemma wrapper workflow](https://github.com/tenstorrent/tt-agentic-bringup-qb2/blob/mvasiljevic/gemma-tb2-ci/.github/workflows/manual-tt-shield-dispatch.yml) therefore gained two temporary `hold-*` inputs:

- `hold-occupied-runner`: checks `docker ps -aq`. If containers exist, it sleeps on that runner and periodically checks whether it is clean. It does not remove or stop the container.
- `hold-unhealthy-runner`: compares `RUNNER_NAME` with an exact requested runner name. It exits immediately on a different runner; on the named runner it sleeps. This was used for fabric-startup failures.

The hold step is read-only and can occupy a runner for up to about 170 minutes (`timeout-minutes: 180`). Its only purpose was to take a problematic runner out of the shared pool while the pending Gemma jobs ran elsewhere. The wrapper also passed `preserve-unowned-containers: true` to Shield. We did not delete the unowned container or manually reset the devices as part of this workaround. Standard self-hosted runner hooks are outside the hold step and may run when any job starts.

Initial [occupied-runner hold](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37963641492) and [second hold](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37973784755) claimed `p04t07`. Later [exact-name `p04t07`](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38010437751) and [`p04t05`](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38010838406) holds let a [C4 Terminal retry](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38010466649) and a [C8 Terminal retry](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38010947583) run on other hosts. The holds had to be renewed when their three-hour limits approached; for example, [renewed `p04t05`](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38042229930) and [`p04t07`](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38042234212) guards were verified on their intended runners before reissuing the twenty-case Terminal scheduling job. Some queued model attempts were canceled before hardware work to place these guards first. An exact-name hold assigned to a healthy runner exited, so multiple attempts were sometimes needed to claim the intended host.

This is a temporary queue-management workaround, not a health repair. It consumes CI slots, needs manual assignment checks and renewal, and cannot ensure that an intermittently failing host remains healthy after a passing job.

## Requested infrastructure follow-up

1. Inspect the stopped unowned container on `p04t07`, determine its owner, and restore that host to the pool through the normal ownership process. Keep the preserve-unowned-container guard in place until ownership is resolved.
2. Run the appropriate four-chip fabric/connectivity health check on `p04t05` and `p05t06`, and inspect topology discovery and mapping diagnostics for the failed starts. The failure log itself suggests `./build/test/tt_metal/tt_fabric/test_system_health` if the mapping failure is unexpected.
3. Investigate the 800 MHz AICLK/max-arbiter warning on `p04t05` and establish whether the host is suitable for comparable performance measurements.
4. Provide a maintained runner quarantine or health label for the shared QB2 pool, so unhealthy or occupied machines do not require three-hour sleeping Actions jobs and repeated manual dispatches.

The broader experiment history, including the successful retries and score accounting, is in [Gemma 4 31B QB2 agentic evaluation and serving experiments](gemma4_31b_qb2_agentic.md).
