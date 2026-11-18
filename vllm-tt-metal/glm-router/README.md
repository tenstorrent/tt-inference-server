# glm-router: the router in front of GLM-5.3-Flash on Galaxy

This directory is the router that serves GLM-5.3-Flash on OpenRouter (state of 2026-11-17). One router process sits in
front of four Blackhole Galaxies, and each Galaxy runs one `glm-tt serve` on port 8000.

Two builds are here. The router that carries production traffic today is the **media-placeholders-only** build (one
patch). The **route3** build (three patches: media placeholders, port sharing, retry-exclude) and its handoff
tooling (`glm-router.sh handoff`) are provided and tested, and running idle beside the production router, but they do
not carry traffic yet.

## What runs in production today

| | |
| --- | --- |
| Router | [vllm-project/router](https://github.com/vllm-project/router) at commit `3451e66` (vllm_router_rs 0.1.17) plus `router-media-placeholders.patch` (sha256 `583e9eeb...`). Nothing else. |
| Toolchain | rustc / cargo 1.99.1, `cargo build --release --locked --bin vllm-router` |
| Binary | sha256 `2a98718e...`. A rebuild at the original build path and CARGO_HOME reproduces it bit for bit; at another path the code is the same but the sha256 differs, because cargo hashes the path into its metadata. Your build from `build-router.sh` will therefore have a different sha256. |
| Policy | `cache_aware`, cache threshold 0.3, imbalance abs 8 (the slots per Galaxy) and rel 2, 5 retries, request timeout 7200 s, health check `/health` every 10 s (3 failures) |
| Switches | `VLLM_ROUTER_MEDIA_PLACEHOLDERS=1`, `VLLM_ROUTER_RANDOM_TIES=1` |
| Auth | Each request's bearer key is checked against the Galaxies' `/v1/models` (cached). The router adds no key of its own. |

## The route3 build (provided and tested, not serving yet)

route3 is the same router at `3451e66` plus three patches, applied in this order: `router-media-placeholders.patch`
(`583e9eeb...`), `router-reuseport.patch` (`07424e7f...`), `router-retry-exclude.patch` (`ddf95b3a...`). Its binary's
sha256 is `68ad05bd...` (same reproducibility note as above). It runs with all four switches on
(`VLLM_ROUTER_MEDIA_PLACEHOLDERS`, `VLLM_ROUTER_RANDOM_TIES`, `VLLM_ROUTER_REUSEPORT`,
`VLLM_ROUTER_RETRY_EXCLUDE_FAILED`). It has been built and tested, and runs idle on its own port beside the production
router; production traffic has not been moved to it. With it, two loop sessions share the client port
(SO_REUSEPORT): `glm-router.sh handoff` starts the next router beside the live one, then the old one finishes its
streams and exits, and the metrics port alternates between `METRICS_PORT` and `METRICS_PORT_B` at each handoff.

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

## Retries on a full Galaxy: a known limitation of the serving router, fixed in route3

Stock 0.1.17 remembers failed workers only for the `kv_aware` policy. With `cache_aware`, when the Galaxy holding a
conversation answered 429 (all its slots busy), every retry goes back to the same Galaxy, and the client gets a 429
while another Galaxy is free. This is still the behavior of the router serving today. Retry-exclude, in route3,
fixes that once route3 takes traffic: in tests with three always-full workers and one real serve,
the stock router returned 429 for 60% of requests and the patched one for none, with no dropped or duplicated
request and every stream intact.

**Cache-share caveat.** A retry that leaves the owner prefills cold on another Galaxy, and the conversation's next
turns can follow it there. How much cached share that costs depends on how often the owner frees a slot within the
stock retry window: about 1 point in tests with 8 slots per worker and an admission cap at 75% of slots, 6-13 points
with 2 slots per worker under heavy 429 pressure, and much more when load runs above total capacity. Keep the
admission cap in front of the router at or below 75% of the total slots, and watch each Galaxy's cached share after
turning it on. Turning it off is one env line and a handoff.

(This caveat applies when route3 is switched on; the router serving today does not exclude failed workers.)

## Files

- `build-router.sh`: builds a binary into `PREFIX/bin` (user-level rustup, no system change). `PATCHES=placeholders`
  (the default) is the production router today; `PATCHES=route3` adds port sharing and retry-exclude.
- `router-media-placeholders.patch`, `router-reuseport.patch`, `router-retry-exclude.patch`: the patches on top of
  0.1.17; production today applies the first only, route3 all three in that order.
- `glm-router.sh`: run, start, stop, retire, handoff, metrics, status, check, add/remove/drain a worker. It reads
  its settings from `glm-router.env`.
- `glm-router.env.example`: the settings production uses today, with addresses and paths as placeholders, and the
  extra route3 lines marked.

## Run it

```
./build-router.sh                       # production's media-placeholders build -> ~/glm-router/bin/vllm-router (prints its sha256)
PATCHES=route3 ./build-router.sh        # the three-patch build instead (adds port sharing and retry-exclude)
cp glm-router.env.example ~/glm-router/glm-router.env   # set BIND, WORKERS_FILE, KEY_FILE, VALIDATION_URLS
printf 'http://<galaxy-1-address>:8000\nhttp://<galaxy-2-address>:8000\n' > ~/glm-router/workers
./glm-router.sh start
./glm-router.sh check
```

To replace the running router (a new binary or new settings) with no refused connection, build with `PATCHES=route3`,
set `REUSEPORT=1` in the env file, and run `./glm-router.sh handoff` (the live router must itself share the port;
route3 does, the production router today does not); then read the new metrics port with `./glm-router.sh metrics`. A router started without
`REUSEPORT=1` cannot share its port, so its first handoff refuses connections for the new router's start time.

Every router start or handoff begins with empty prefix trees: until conversations come back, turns route by least
load.

The serve itself and its launcher are in `../src/run_glm_tt.py`, `../src/glm_tt_entrypoint.py` and
`../tt-lab-glm.Dockerfile`. The engine source and the prebuilt `glm-tt` that production runs are published separately.
