# Isolated LongBench context diagnostic

This branch replays 24 fixed examples twice against the packaged QB2 server. It does not qualify a release. Do not merge its workflow override into the release candidate.

The cases are the shortest, median and longest prompt from each of eight selected LongBench tasks. Selection uses prompt length, never the score. One arm sends the original final 1,535 tokens; the other sends the full prompt. Both run serially with the original asynchronous completion payload (`stop=[]`, temperature 0, seed 42, up to 512 output tokens). No chat wrapper or special token is added. The 24 examples are a diagnostic subset of the 3,668-example LongBench evaluation.

Shield downloads the fixed source artifact through normal CI ownership. Before inference, the helper verifies source hashes, sample IDs, tokenizer revision and file hash, both payload hashes, the immutable server image, checkpoint and observed generator/precision/configuration. It writes every response, usage and finish reason before grading transport validity. It stops on the first transport or identity error and does not retry or resume requests. `COMPLETE.json` records successful transport completion, not accuracy acceptance.

Normal server startup, reporting, artifact upload and cleanup remain in use. The standard accuracy tasks are explicitly marked skipped on this branch. Frozen performance references and full accuracy thresholds remain unchanged in the release candidate.

Offline verification, without server or hardware:

```sh
python diagnostics/longbench_context.py --source /path/to/downloaded/artifact \
  --tokenizer /path/to/pinned/tokenizer.json --output /new/output/path --prepare-only
```

To analyze a completed diagnostic, load `responses.jsonl` and use the pinned harness's task-specific metrics and original references from the source samples. Compare paired per-example scores and token usage. Treat original concurrent-run responses only as observational controls; the new serial pair isolates prompt length. A subset gain does not establish full-dataset accuracy or GPU-reference comparability. Revalidate the full affected evaluation after a justified repair.
