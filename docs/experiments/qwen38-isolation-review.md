# Stage Review

Verdict: clean-pass

Scope: independent review of the bounded isolated-agent **protocol source stage**,
with a canonical-driver follow-up on 2026-10-27. It establishes no clean SWE
accuracy, grouped-model correctness, speed improvement, SWE100/500 readiness, or
Qwen release pass. Source findings are closed; repeat the bounded runtime smoke.

Current follow-up: live branch `mvasiljevic/qwen38-swe-isolated-b8-20261027`,
commit `60a0323a46167f4bb482d940e7f75002412e3017` plus the reviewed uncommitted
early-selector validation and its three regression cases. The earlier bounded source
clean-pass at base `94e098debcd31fcc2c3b6cebe13f012cfc9c0830` missed the unchanged
canonical driver, which discarded the Docker isolation selector. Its 71 passing
focused tests did not establish end-to-end propagation. Actual smoke
`40536968443` exposed that omission and was rejected; its reward 1, no exception,
and 321.24061-second envelope are not clean accuracy evidence.

Contract: preserve the frozen SWE recipe and default
public mode; disconnect setup egress before model-driven execution; retain the
host-native API; restore egress only after owned workload settlement; preserve
exact-resource teardown and closed-job reconciliation.

## Required Work

None within this bounded source review. Actual isolation and accuracy acceptance
remain gated on the smoke and matched runs below.

## Other Concerns

- Commit `60a0323a` fixes Docker propagation by forwarding only the new selector,
  preserving the previous treatment of unrelated provider kwargs. Independent
  CPU checks traced the actual controls through `EVAL_CONFIGS`, the canonical
  driver and final saved config without overriding the selector: the flag,
  owned environment, owned agent hook and API alias all survive. Docker malformed
  selectors are rejected and explicit false retains public mode. The follow-up
  guard now validates type and requires Docker before provider filtering; all
  unsupported providers fail closed when isolation is requested.
- Pinned Harbor `9973a7edc2d9950a03de6aeebafebb8f7b0a5f1e` completes agent
  installation before `run`. The wrapper detaches the public bridge before that
  call. Existing supervised exec settlement completes before `run` unwinds and
  before public egress restoration; an unsettled workload blocks restoration and
  verifier exec. Uncertain network transitions prohibit retries and preserve both
  journals for reconciliation.
- The primary bridge is explicitly internal with IPv6 disabled. The independently
  allocated public bridge has its own 32-hex attempt and matching journal filename;
  both retain the same exact owner/job. The pinned closed-job cleanup snapshot
  accepts that journal identity. Normal teardown removes verified containers,
  then the two verified empty networks, without forced foreign disconnection.
- The isolated arm avoids the old `host-gateway` overlay and rejects a nonunique
  normalized API-host mapping. The model alias maps to the primary bridge gateway;
  allocator host inventory still receives the real loopback endpoint hostname.
  Host-network serving is opt-in. Agent kwargs, generation policy and timeouts
  remain unchanged; public behavior has regression coverage.

## Hard-Check Gaps

- Before collecting clean accuracy, repeat the actual pinned Harbor path on one
  original B8/block256/medium task. Establish model API access and failed external
  GitHub/PR retrieval during the agent phase, plus journal transitions
  `setup_egress=[true,false,true]`, verifier completion, and exact cleanup. The
  existing CPU Docker proof exercises the network primitive, not the complete
  adapter or vLLM endpoint. Record the effective source/config/image identities.
- After that smoke, collect matched isolated control/candidate SWE8 runs before
  advancing to SWE25/100/500. Earlier public trials cannot supply clean accuracy
  or a comparable speed control for the changed network protocol. The full-model
  strict checks and broader serving/release gates remain separate.
- Expanded CPU checks encountered the pre-existing
  `test_occupancy32_manifest_and_handoff_exact_identities` failure: this pilot's
  committed controls select eight tasks, while the test requires 32. Both values
  already differ at reviewed HEAD. This unrelated historical check is not green;
  it was not altered to make the isolation review pass.
- Expanded follow-up checks found another pre-existing assertion:
  `test_harbor_checkout_uses_runtime_branch` expects `tt-inference-server`, while
  the source pins `9973a7ed...`. It already fails at `60a0323a^`; it was not
  changed for this protocol review.

## Anomaly Ledger

- Observed anomaly: the first intended isolated smoke ran on a public task bridge.
  Evidence: run `40536968443` saved config omits `owned_agent_isolation`; its sole
  task-network journal has no `internal` flag, parent setup attempt or network
  transitions. Affected path: canonical eval-to-Harbor driver. Control: actual
  current controls through the corrected driver serialize true and both owned
  hooks. Likely subsystem: Docker kwargs filtering. Investigation: inspected raw
  config/journal and commit `60a0323a`; reran canonical CPU regressions.
  Resolution: Docker source path fixed; smoke rejected and must be repeated.
- Observed anomaly: the canonical driver still discarded true for supported
  non-Docker providers, hiding isolation from writer validation. Evidence: Daytona
  serialized with empty kwargs and ordinary MiniSwe. Affected path: provider
  filtering. Control: actual Docker selector propagation. Likely subsystem:
  validation ordering. Investigation: reproduced, confirmed the pinned Harbor enum,
  reviewed the early guard and reran canonical tests. Resolution: fixed; non-Boolean
  values and true for Daytona/Kubernetes are rejected before output/filtering.
- Observed anomaly: cancellation during final egress restoration returned the
  successful agent result. Evidence: CPU reproduction with the real wrapper and
  barrier; affected path: `OwnedMiniSweAgent.run` finally block. Control:
  cancellation after parent success while restoration awaits an event. Likely
  subsystem: discarded barrier cancellation flag. Investigation: reproduced,
  reviewed owner fix, reran regression. Resolution: fixed; repeated cancellation
  now waits for restoration and then raises `CancelledError`.
- Observed anomaly: isolation was accepted for Terminus/custom agent imports
  without the isolation hook, retaining public setup egress. Evidence: generated
  config with `owned_agent_isolation=true` and no owned MiniSwe hook. Affected
  path: Harbor config writer. Control: standard MiniSwe path. Likely subsystem:
  missing configuration guard. Investigation: reproduced and reviewed owner fix.
  Resolution: fixed; only Boolean opt-in with owned Docker and standard MiniSwe
  without an explicit agent import is accepted; rejection tests pass.
- Observed anomaly: upstream reference material reached successful public SWE8
  trials. Evidence: medium run `40518335859`, returned SymPy PR diffs at trajectory
  step 21 (`18698`) and step 44 (`21612`), plus Sphinx cloned history and fix commit
  `333e7a447` at steps 39–42 (`8459`). Affected path: public agent networking.
  Control: primitive CPU proof retains API access while blocking external TCP.
  Likely subsystem: benchmark protocol. Investigation: inspected actual commands
  and returned observations, not only summary counts. Resolution: controlled for
  reporting: raw 6/8 is reference-exposed and excluded from clean accuracy;
  isolated scored evidence is pending.

## Scope Inspected

- Skills: installed `stage-review/SKILL.md`, `model-bringup/SKILL.md`, and
  `model-bringup/references/target-workload.md` from package version `0.1.21`.
  Applicable workload is multi-turn SWE with valid task completion before speed;
  this patch changes protocol, not memory/precision/trace behavior.
- Code: `agentic_owned_networks/{allocator,docker_environment,mini_swe_agent,
  execution_scope,exec_authority,exec_supervisor}.py`, `llm_module/agentic/harbor.py`,
  `llm_module/drivers/agentic.py`, `test_module/llm_tests/agentic_eval_tests.py`,
  `reference_config/evals/qwen38_goal12.{py,json}`, and
  `workflows/run_docker_server.py`; relevant new and existing tests.
- Evidence root: `/home/mvasiljev/build/qwen38-swe/`; inspected `STATE.json`,
  `internal-network-proof.json`, `probe_internal_network.py`, read-only
  `shield-owned-cleanup.py`, medium summary and selected raw trajectories.
  Inspected the exact pinned Harbor setup/run/exec methods from local Git objects.
- Follow-up evidence: `isolated-smoke-40536968443` saved Harbor config and task
  journal, `isolated-smoke-summary.json`, dispatch inputs, and committed protocol
  notes. Narrow canonical-driver/Qwen suite: **78 passed** in 0.29s. Adding
  `tests/workflows/test_eval_config_harbor_env.py`: **100 passed, 1 pre-existing
  failure** described above. Manual CPU checks covered actual controls,
  malformed Docker flags, explicit public mode and the unsupported-provider
  reproduction. Final guard/source review reran the canonical-driver/Qwen suite:
  **81 passed** in 0.26s; `git diff --check` clean. No Docker command was run in
  either canonical-driver follow-up.
- Commands: read-only Git/source/artifact inspection; CPU cancellation/config
  reproductions; `git diff --check` (clean); initial focused pytest run **71 passed**
  in 1.37s. Selected files: owned network, environment, settlement, isolation,
  phase cancellation, Compose normalization, Harbor dataset/recipe, Qwen isolated
  network and audit-mount tests. Interpreter:
  `/home/mvasiljev/tt-metal/python_env/bin/python`, with local `test-deps` on
  `PYTHONPATH`. Compose checks were config-only. No hardware, server, Docker
  mutation, CI dispatch, model-source edit or upload was performed by the reviewer.

## Residual Risk

The CPU proof covers one external IPv4 TCP destination and a simple host HTTP
listener. Actual task images, proxy settings and the production API remain to be
exercised by the smoke. This is a cooperative benchmark network restriction;
host services, DNS and pre-existing repository contents require their own evidence
before any stronger security or contamination-free claim. This review does not
assess the active full-model experiment.
