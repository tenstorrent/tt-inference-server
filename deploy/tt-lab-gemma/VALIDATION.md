# Fresh Gemma validation — September 23, 2026

Branch `review/gemma-native` preserves the Gemma integration on normal
upstream history, with the shared persistent tt-lab runner prerequisites
in a separate commit. Original dirty snapshots were left untouched.

Validated server source: `c2510566` (following commits add reports only).
The companion native source is `6536f9692192cb6c1cebe9849f2e8d9d60113f5a` in
`tenstorrent/tt-lab`, branch `review/gemma-native`. Its freshly built full
30-layer native executable has SHA256
`996fbeccb68752ab48ea842d77163bd4aecd23930aa868cda5e5732970f07317`.

Fresh checks passed:

- 4 Gemma adapter and 20 GPT-OSS protocol tests, run in separate processes.
- 86 shared service/chat/scheduler and 13 device-worker tests, with
  `MODEL_RUNNER=llm_test MODEL_SERVICE=llm IS_GALAXY=false DEVICE_IDS='(0)'`
  and `PYTHONPATH=.` from `tt-media-server`.
- `benchmark.py --long --full-context`: factual/math/code responses,
  streaming, repeat/reset behavior, invalid parameters, and exact
  4088-input/eight-output context-boundary coverage.
- Separate nonstreaming arithmetic request: 391; healthy readiness/canary;
  zero worker errors/restarts and service restarts.
- Native full-model bitwise silicon/proxy check and 30-layer upstream CPU
  comparison; details and raw evidence are in the
  [native validation report](https://github.com/tenstorrent/tt-lab/blob/review/gemma-native/docs/gemma-review-validation.md).

The review version is running on `http://127.0.0.1:8001` in the transient
user unit `ttlab-gemma-review-validation.service`. It loads this checkout's
`server.env` as literal key/value pairs, overrides `TT_LAB_BINARY` to
`/home/ttuser/workspace/tt-lab-gemma-review/_out/tt-lab`, uses the local
Transformers overlay and server virtualenv, and launches `uvicorn main:app`
from this review checkout's `tt-media-server`. This tests the reviewed code,
not the older executable in the original snapshot. The unit is not enabled
across reboot. Stop it with:

```sh
systemctl --user stop ttlab-gemma-review-validation.service
```

The original Gemma service remains inactive and disabled. Do not start it
while the review instance owns card 1/port 8001. No GPT-OSS service/card was
modified. The committed `start.sh` and `server.env` retain their original
machine-local deployment paths; adjust those for another checkout or host.

Only the supported 32-worker text-serving configuration was validated.
Unfinished 64-worker native diagnostics, broader quality evaluation, and
the historical faster-prefill targets are not claimed complete.
