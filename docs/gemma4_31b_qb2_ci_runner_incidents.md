# QB2 CI runners need quarantine and health checks

During Gemma 4 31B agentic evaluations on 9–10 October 2026, the shared `bh-qb-ge` runner pool assigned jobs to hosts that could not reliably start the model. These failures occurred before evaluation and produced no benchmark scores.

| Runner | Observed failure | Example run |
| --- | --- | --- |
| `p04t07` | A stopped, unowned Docker container caused Shield to refuse cleanup and hardware execution: `Host contains unowned containers`. | [Blocked Terminal job](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37962790734) |
| `p04t05` | Four-chip fabric mapping failed at `topology_mapper.cpp:556`. Other runs on this host warned that AICLK was clamped to 800 MHz instead of 1350 MHz. | [Fabric failure](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38001414052); [clock warning](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37963243142) |
| `p05t06` | The same fabric mapping failure occurred after this host had completed other Gemma work; the issue appears intermittent. | [Fabric failure](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38043892436) |

To keep long evaluation jobs off these hosts, I added temporary [`hold-*` jobs](https://github.com/tenstorrent/tt-agentic-bringup-qb2/blob/mvasiljevic/gemma-tb2-ci/.github/workflows/manual-tt-shield-dispatch.yml). They occupy an affected runner for up to 170 minutes, checking either for existing containers or an exact runner name. The hold steps do not remove containers or reset devices. This required manual assignment checks, repeated dispatches, and renewals as holds expired; several model jobs were canceled before hardware work and reissued on other hosts.

**Requested:** Resolve ownership of the container on `p04t07`; check four-chip fabric connectivity and topology discovery on `p04t05` and `p05t06`; investigate the 800 MHz AICLK warning; and provide a maintained quarantine or health label so unhealthy runners are excluded without sleeping Actions jobs. The logs do not establish whether the fabric failures are hardware or software faults.
