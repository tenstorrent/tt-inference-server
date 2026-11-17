# glm-router: the router in front of GLM-5.3-Flash on Galaxy

This directory is the router that serves GLM-5.3-Flash on OpenRouter today (2026-11-16). One router process sits in
front of four Blackhole Galaxies, and each Galaxy runs one `glm-tt serve` on port 8000.

## What runs in production

| | |
| --- | --- |
| Router | [vllm-project/router](https://github.com/vllm-project/router) at commit `3451e66` (vllm_router_rs 0.1.17) plus three patches, applied in this order: `router-media-placeholders.patch` (sha256 `583e9eeb...`), `router-reuseport.patch` (`07424e7f...`), `router-retry-exclude.patch` (`ddf95b3a...`). Nothing else. |
| Toolchain | rustc / cargo 1.99.1, `cargo build --release --locked --bin vllm-router` |
| Binary | sha256 `68ad05bd50a82d107ed228f6892656b6c955507b38642c08d4c756c46d5df1d8`. A rebuild at the original build path and CARGO_HOME reproduces it bit for bit; at another path the code is the same but the sha256 differs, because cargo hashes the path into its metadata. Your build from `build-router.sh` will therefore have a different sha256. |
| Policy | `cache_aware`, cache threshold 0.3, imbalance abs 8 (the slots per Galaxy) and rel 2, 5 retries, request timeout 7200 s, health check `/health` every 10 s (3 failures) |
| Switches | `VLLM_ROUTER_MEDIA_PLACEHOLDERS=1`, `VLLM_ROUTER_RANDOM_TIES=1`, `VLLM_ROUTER_REUSEPORT=1`, `VLLM_ROUTER_RETRY_EXCLUDE_FAILED=1` |
| Handoff | Two loop sessions share the client port (SO_REUSEPORT); `glm-router.sh handoff` starts the next router beside the live one, then the old one finishes its streams and exits. The metrics port alternates between `METRICS_PORT` and `METRICS_PORT_B` at each handoff. |
| Auth | Each request's bearer key is checked against the Galaxies' `/v1/models` (cached). The router adds no key of its own. |

What the patches do (each is off while its switch is unset; with all four off the binary routes exactly as stock
0.1.17):

- **Media placeholders** (`VLLM_ROUTER_MEDIA_PLACEHOLDERS=1`). In stock 0.1.17, `cache_aware` keys its prefix tree on
  the token ids from the serve's render endpoint, which a chat with images does not get. Those requests fell back to
  least-loaded routing. With the switch, each image becomes a placeholder key, so a conversation with images stays on
  the Galaxy that holds its cache.
- **Random ties** (`VLLM_ROUTER_RANDOM_TIES=1`, same patch). Least-loaded ties are broken at random instead of always
  going to the first worker.
- **Port sharing** (`VLLM_ROUTER_REUSEPORT=1`, set by `glm-router.sh` from `REUSEPORT=1`). The listener gets
  SO_REUSEADDR and SO_REUSEPORT, so a new router binds the same port while the old one drains: a handoff refuses no
  connection.
- **Retry-exclude** (`VLLM_ROUTER_RETRY_EXCLUDE_FAILED=1`). After an attempt answers a retryable status (408, 429,
  500, 502, 503, 504), the worker goes on the request's failed list, and the retry picks among the other workers.
  When every worker has failed it, it picks from all again. First choice, the request body, streaming and the retry
  budget are unchanged. The patch touches only `src/routers/http/router.rs`.

## Retries on a full Galaxy: fixed

Stock 0.1.17 remembers failed workers only for the `kv_aware` policy. With `cache_aware`, when the Galaxy holding a
conversation answered 429 (all its slots busy), every retry went back to the same Galaxy, and the client got a 429
while another Galaxy was free. Retry-exclude fixes that: in tests with three always-full workers and one real serve,
the stock router returned 429 for 60% of requests and the patched one for none, with no dropped or duplicated
request and every stream intact.

**Cache-share caveat.** A retry that leaves the owner prefills cold on another Galaxy, and the conversation's next
turns can follow it there. How much cached share that costs depends on how often the owner frees a slot within the
stock retry window: about 1 point in tests with 8 slots per worker and an admission cap at 75% of slots, 6-13 points
with 2 slots per worker under heavy 429 pressure, and much more when load runs above total capacity. Keep the
admission cap in front of the router at or below 75% of the total slots, and watch each Galaxy's cached share after
turning it on. Turning it off is one env line and a handoff.

## Files

- `build-router.sh`: builds the binary above into `PREFIX/bin` (user-level rustup, no system change).
- `router-media-placeholders.patch`, `router-reuseport.patch`, `router-retry-exclude.patch`: the three patches on
  top of 0.1.17, applied in that order.
- `glm-router.sh`: run, start, stop, retire, handoff, metrics, status, check, add/remove/drain a worker. It reads
  its settings from `glm-router.env`.
- `glm-router.env.example`: the settings production uses, with addresses and paths as placeholders.

## Run it

```
./build-router.sh                       # -> ~/glm-router/bin/vllm-router (prints its sha256)
cp glm-router.env.example ~/glm-router/glm-router.env   # set BIND, WORKERS_FILE, KEY_FILE, VALIDATION_URLS
printf 'http://<galaxy-1-address>:8000\nhttp://<galaxy-2-address>:8000\n' > ~/glm-router/workers
./glm-router.sh start
./glm-router.sh check
```

To replace the running router (a new binary or new settings) with no refused connection, edit the env file and run
`./glm-router.sh handoff`; then read the new metrics port with `./glm-router.sh metrics`. A router started without
`REUSEPORT=1` cannot share its port, so its first handoff refuses connections for the new router's start time.

Every router start or handoff begins with empty prefix trees: until conversations come back, turns route by least
load.

The serve itself and its launcher are in `../src/run_glm_tt.py`, `../src/glm_tt_entrypoint.py` and
`../tt-lab-glm.Dockerfile`. The engine source and the prebuilt `glm-tt` that production runs are published separately.
