# glm-router: the router in front of GLM-5.3-Flash on Galaxy

This directory is the router that serves GLM-5.3-Flash on OpenRouter today. One router process sits in front of four
Blackhole Galaxies, and each Galaxy runs one `glm-tt serve` on port 8000.

## What runs in production

| | |
| --- | --- |
| Router | [vllm-project/router](https://github.com/vllm-project/router) at commit `3451e66` (vllm_router_rs 0.1.17) plus `router-media-placeholders.patch` (sha256 `583e9eeb...`), nothing else |
| Toolchain | rustc / cargo 1.99.1, `cargo build --release --bin vllm-router` |
| Binary | sha256 `2a98718e3178278d79f520bf8b4093a308d7c8ec77534eb59d011133b688f896` (64,930,896 bytes). A rebuild at the original build path reproduces it bit for bit; at another path the code is the same but the sha256 differs, because cargo hashes the path into its metadata. |
| Policy | `cache_aware`, cache threshold 0.3, imbalance abs 8 (the slots per Galaxy) and rel 2, 5 retries of a 429 |
| Switches | `VLLM_ROUTER_MEDIA_PLACEHOLDERS=1`, `VLLM_ROUTER_RANDOM_TIES=1` |
| Auth | Each request's bearer key is checked against the Galaxies' `/v1/models` (cached). The router adds no key of its own. |

What the patch does:

- **Media placeholders.** In stock 0.1.17, `cache_aware` keys its prefix tree on the token ids from the serve's render
  endpoint, which a chat with images does not get. Those requests fell back to least-loaded routing. With
  `VLLM_ROUTER_MEDIA_PLACEHOLDERS=1`, each image becomes a placeholder key, so a conversation with images stays on the
  Galaxy that holds its cache.
- **Random ties.** With `VLLM_ROUTER_RANDOM_TIES=1`, least-loaded ties are broken at random instead of always going to the
  first worker.

## Files

- `build-router.sh`: builds the binary above into `PREFIX/bin` (user-level rustup, no system change).
- `router-media-placeholders.patch`: the one patch on top of 0.1.17.
- `glm-router.sh`: start, stop, status, check, add/remove/drain a worker and a zero-drop handoff between two router
  instances. It reads its settings from `glm-router.env`.
- `glm-router.env.example`: the settings production uses, with addresses and paths as placeholders.

## Run it

```
./build-router.sh                       # -> ~/glm-router/bin/vllm-router
cp glm-router.env.example ~/glm-router/glm-router.env   # set BIND, WORKERS_FILE, KEY_FILE; uncomment the two switches
printf 'http://<galaxy-1-address>:8000\n...' > ~/glm-router/workers
./glm-router.sh start
./glm-router.sh check
```

## Known limitation

When the chosen Galaxy answers 429 (full), the router's retries can pick the same Galaxy again, so a client can get a
429 while another Galaxy is free. In 0.1.17 only the `kv_aware` policy remembers which worker failed. A patch that
excludes failed workers on retry is being tested; it is not in this build.

The serve itself and its launcher are in `../glm_tt_entrypoint.py` and `../tt-lab-glm.Dockerfile`. The engine source and
the prebuilt `glm-tt` that production runs are published separately.
