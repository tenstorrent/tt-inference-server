# Qwen SWE validation protocol

The network-enabled medium SWE8 pilot40518335859 returned6/8 verifier passes,
all8 without exceptions or missing responses, in3102.957499 seconds of trial
envelope. It saved124375 completion tokens and8914897 prompt tokens
(8687616 cached). All320 audited requests selected medium reasoning, and all
8 task checksums match the historical frozen cohort. This is a raw scaffold
result, not clean accuracy: three successful trials retrieved upstream fixes.

| Task | Raw reward | Observed reference access |
|---|---:|---|
|astropy13398|0|Downloaded matching upstream golden patch/PR diff|
|astropy13453|1|No external fix retrieval found in reviewed actions|
|astropy14309|1|Read repository history around the base change; no future fix retrieval found|
|astropy7166|0|No external retrieval found in reviewed actions|
|scikit-learn13135|1|No external retrieval found in reviewed actions|
|sphinx8459|1|Cloned upstream history and read the matching fix commit|
|sympy18698|1|Read matching PR files and raw diff|
|sympy21612|1|Read matching PR diff and changed implementation accordingly|

Do not turn the unexposed successes into a post-selected accuracy denominator.
The earlier network-enabled runs remain performance/scaffold baselines. No
matched GPU Verified reference exists here; the published61.7 figure is Pro.

The optional owned-agent isolation protocol retains a per-trial internal Docker
bridge throughout. A separately owned public bridge permits package installation
before the agent and verification afterward. The installed mini-swe-agent wrapper
detaches public egress before model-driven execution and restores it only after
the supervised workload settles. Model API requests reach a host-native vLLM
listener through the internal bridge gateway. Both bridge identities, phases,
labels and closed-job journals remain explicit; uncertain mutation or unsettled
workloads block further transitions. Other agents/import paths cannot enable
this mode silently. Normal public runs keep their existing behavior.

This changes benchmark protocol and must be applied to both comparison arms.
Keep the model/prompt, frozen tasks, generation settings and deadlines fixed.
The code/CPU stage is reviewed separately from actual Harbor execution and
model acceptance; no clean score or speedup follows from source review.

Local CPU-only Docker proof: setup/verifier external TCP succeeds; agent phase
external TCP fails while the mock host API stays reachable. Both owned test
containers/networks were removed. Actual Compose normalization showed that
extra_hosts merges duplicate hostname entries instead of replacing them; the
isolated path therefore omits the legacy host-gateway mapping and validates its
new mapping uniquely. A cancellation first delivered during egress restoration
must finish restoration and still propagate cancellation. Focused regression
coverage and independent review are recorded in qwen38-isolation-review.md.

Next gate: one-task Harbor smoke using the original B8/256 configuration, then
matched isolated SWE8 control/candidate before25/100/500. Check API success,
network transition journals, complete task denominator, exceptions, missing
responses, verifier rewards and trajectories. This protocol does not change
model context262144 or grant a reference-score parity claim.

Raw logs, trajectories, CSVs, Tracy captures and probe dumps stay local/CI;
Git contains code and compact documentation only.

The first attempted isolated CI smoke40536968443 completed its one task
(reward1, no exception,321.24061s) but failed the protocol gate: its saved
Harbor config had no isolation flag and its bridges were public. It is not a
clean score. The canonical agentic driver previously forwarded environment
kwargs only for Kubernetes and discarded them for Docker. It now forwards only
the owned-agent isolation selector for Docker; unrelated provider kwargs retain
their previous behavior. The regression test builds the actual Qwen eval task,
passes it through build_harbor_config, then serializes the final Harbor config.
The one-task smoke must be repeated and actual network journals inspected before
running matched isolated SWE8 arms.
