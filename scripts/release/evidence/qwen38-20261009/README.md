# Qwen BFP8 container qualification handoff, October 9, 2026

Image manifest:
`sha256:79f7b4469a6ec2bcce5204399b37b2aeced8be7f260dd98f7007ad41f8813055`.
Model source: `20619e008a236aaf393937b222a60a5b03e49cdc`.
The image identity and corrected real-entrypoint probe passed. The probe's
receipt explicitly records that no hardware was opened and no server started.
It supplied the required model/device CLI arguments, exercised the actual TTIS
wrapper, and intercepted only the final vLLM launch. No image rebuild was needed
for the externally mounted probe correction.

`image-hardware-control-v2/launch.json` contains the exact persistent launch,
controller hash and evaluator hash. `evaluation-command.json` retains the
full-198 OpenBench command. The first waiting-only v1 supervisor was replaced
after verifying its exact invocation and `hardware_opened: false`; neither
supervisor interrupted the active performance queue.

The v2 supervisor waits for that queue's exact invocation to terminate cleanly,
then obtains `/tmp/tt-device.lock` before resetting the Galaxy once and launching
the immutable image. It checks all eight worker bindings and BFP8 precision,
streaming/nonstreaming, multi-turn chat, concurrency 128, and a synthetic tool
round trip. Full OpenBench then runs against the same resident container.

Limits: nine-hour predecessor wait, 90-minute container readiness, 30-minute API
suite, 300 seconds per tool request, 9,000-second evaluation, 15-hour service
lifetime. Container resources are 256 GiB RAM and 16 CPUs; the host supervisor
has 4 GiB and four CPUs. A systemd stop hook removes only the task-labelled
container. Task-local caches are removable; weights and the root filesystem are
read-only. The job survives disconnect, but does not resume after a reboot.

The new evaluator explicitly sets both the HTTP client's 7,000-second timeout
and zero SDK retries. These were absent from the first OpenBench run, which
timed out eight requests. Its output is retained separately in the model branch;
the rerun neither selects failed questions only nor combines scores across runs.

`live-observation.json` and the status JSONs are point-in-time snapshots, not
completion claims. Inspect the live service and final receipts before reporting
success. This queue does not qualify SJC3 Helm, Shield CI or agentic benchmarks.
Local startup/ownership/evaluation-command regressions: **43 passed**.
