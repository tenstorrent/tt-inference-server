# Tests on a real server and device

Most of these tests send HTTP requests to an already running `tt_media_server_cpp` with a
real runner and wait for real generation (about 80 s per job on a T3K with 4
steps). They do not start the server themselves: the server must be
`model_ready`.

## Starting the server (T3K, Wan2.2)

From `tt-media-server/cpp_server`:

```bash
source <tt-metal>/python_env/bin/activate
MODEL_SERVICE=video \
MODEL=Wan2.2-T2V-A14B-Diffusers \
MODEL_RUNNER=tt-wan2.2 \
DEVICE=t3k \
DEVICE_IDS="(0,1,2,3)" \
TT_DIT_CACHE_DIR=<path to the tt_dit weight cache> \
TT_METAL_HOME=<tt-metal> \
TT_VIDEO_TRACED=0 \
HF_HUB_OFFLINE=1 \
OPENAI_API_KEY=<token> \
./build/tt_media_server_cpp -p 8000
```

Notes:
- `DEVICE_IDS="(0,1,2,3)"` is required on a T3K. The default `(0)` gives the worker a
  single card, and the Wan mesh (2x4, 8 chips) cannot be opened.
- `MODEL=Wan2.2-T2V-A14B-Diffusers` is required. Without it the Python settings pick a
  galaxy configuration with a 1x1 mesh.
- `TT_DIT_CACHE_DIR` should point at an existing weight cache. Otherwise the weights are
  built from PyTorch at every start, which is slow.
- If the start fails with `Read unexpected run_mailbox value`, the chips are in a stale
  state and need `tt-smi -r`.
- On a T3K, Wan2.2 runs untraced by default (traced output is noise there, unlike
  the Python server's default). `TT_VIDEO_TRACED=0` in the command above is therefore
  not needed on a T3K, but it makes the setting explicit; `TT_VIDEO_TRACED=1` forces
  tracing.
- With one `DEVICE_IDS` group (the usual T3K launch) the model runs on a thread of the
  server process, not in a worker process; only several groups give worker processes
  (see `docs/VIDEO_SERVICE.md`, "Execution modes"). "No worker left" then means "no
  server left".
- The runner is C++; its only Python is the adapter in `python/cpp_dit_adapters` (only
  tt-metal is needed), so `TT_PYTHON_PATH` is not used. `TT_VIDEO_PYTHON_DIR` overrides
  where the adapters are looked up (default `cpp_server/python`).
- The race test needs the extra variable `TT_MEDIA_TEST_PRECLAIM_DELAY_MS=1500`. It is a
  test hook and must not be set for normal use.

## Folders

| Folder | Topic |
|---|---|
| `cancel/` | cancelling jobs: in the queue, while running, the race with `CLAIMED` |
| `jobs/` | job queue, `download` and `Range`, repeatability, sync mode, validation |
| (shared) `common.py` | helpers: HTTP calls, waiting for statuses, reading the log, the PASS/FAIL list. Not a test |

The tests that use the mock runner (`server_cases.py`, `contract_cases.py`,
`run_server_tests.sh`) are in `tests/video/mock/`.

## Shared parameters

`--base-url` (default `http://127.0.0.1:8000`), `--api-key`, `--steps`
(`num_inference_steps`, default 4), `--server-log`, `--payload-dir`
(`/tmp/tt-video-ipc-<pid>`; by default the directory of the process that listens on
`--base-url`, which is found with `ss`, so a second server on the machine is not picked), `--output-dir`
(the server's `TT_VIDEO_OUTPUT_DIR`), `--download-dir`.

Checks that need the log or the directories of the server are skipped (SKIP) without
those parameters, not failed. The exit code is 0 only if no check failed.

## Adding a new test

1. Create a script in the folder that matches the topic (or a new folder if the topic
   does not fit anywhere).
2. At the top put `sys.path.insert(0, <folder above the script>)` and then
   `import common as c` (the same pattern as in the existing scripts).
3. Use `c.add_common_args(ap)` and `c.setup(args)` at the start and `c.finish()` at the end.
4. In the docstring write the command to run it, and add the script to the table in
   `tests/video/README.md`.
