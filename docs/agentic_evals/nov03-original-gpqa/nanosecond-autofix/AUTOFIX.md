# Python3.10 Docker timestamp compatibility repair

Diagnosis: AUTODEBUG.md. Actual CPython3.10.22 rejects the exact nine-digit fractional timestamp from CI41582292198; newer3.12 accepted it, masking the host-version incompatibility.

Minimal production change: qwen38_selected_precision.docker_timestamp_epoch validates complete timezone-aware RFC3339Nano, normalizes fractional seconds to datetime microseconds (same truncation as3.12), and preserves timezone. verify_adapter_marker retains all generator class, full64 layer, policy, current EngineCore PID and generation comparisons. Malformed/naive timestamps still fail. No server/model/limit/clock change.

Validation: actual local CPython3.10.22 reproduces original ValueError, then3 stdlib tests pass with exact failing nine-digit +00:00, nine-digit Z, nonzero timezone,1–9fractional digits, malformed inputs and full marker identity negatives.35 focused host timestamp/selected-precision/checkpoint/default native-route tests pass. Reconstructed host replay uses the retained actual CI marker body plus the exact Docker timestamp from traceback and passes full policy/class/layer/PID checks; artifact validation.json labels reconstruction explicitly. No new device/model request. One preexisting test assumed live coordinator admission remained STOP; it now supplies its own blocked receipt fixture so the test remains valid after coordinator admission.

Regenerate approved executable-source map for source/test edits; selected-control hash unchanged. The failed CI had no generation events and stopped before hashing/samples. Actual retry runtime is not yet confirmed; publication/re-admission/dispatch remain coordinator-owned, under a new explicitly funded attempt, never a reset of the failed run's recorded clock.
