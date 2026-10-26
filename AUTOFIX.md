# Owned agent execution settlement

Contract: a timed-out installed SWE agent must stop inside its owned task container before verifier execution; a canceled nonstructural Compose exec must not falsely taint daemon structure. Build/up/network uncertainty remains fail closed.

Diagnosis: AUTODEBUG.md traces Harbor 0.23 AgentTimeoutError through cancellation of the untimed Docker exec collector. All 128 examined SWE matrix trials retained a secondary teardown OwnershipError. Host CLI termination does not establish remote workload termination.

Changes: context-local verified exec classification; mini-SWE run-only supervision; Linux subreaper with PID/starttime/boot identity and pidfd signaling; bounded cancellation-resistant settlement; strict receipt validation and verifier block on unproven settlement. Readonly helper mount and known host/daemon authority exclusions precede exec. Agent settings and primary timeout outcome preserved.

Validation: 71 focused lifecycle/authority/supervisor/adapter fixtures passed, plus 8 Harbor configuration tests. Real CPU Docker helper: 3/3 cases, including detached doublefork and blocked loopback HTTP. CLI cancellation left work running; explicit settlement removed recorded descendants. Actual adapter with exact pinned Harbor base methods: repeated cancellation settled before verifier and exact-ID teardown; readonly helper/source hash/default private IPC verified.

Limits: full Harbor package is absent locally; adapter CPU fixture bypasses constructor/start/network allocation. CPU HTTP is loopback, not vLLM cancellation. Fresh real task qualification required before scoring. Cooperative supervision and known authority exclusions are not a hostile-code sandbox. Existing structural uncertainty is never cleared.
