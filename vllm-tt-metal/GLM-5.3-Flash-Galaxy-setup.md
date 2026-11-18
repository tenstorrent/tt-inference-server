# GLM-5.3-Flash on Galaxy: setup as production runs it

This is the setup serving GLM-5.3-Flash on OpenRouter as of 2026-11-17: four Blackhole Galaxies, each running one
`glm-tt serve`, behind one router.

## What runs

| | |
| --- | --- |
| Engine | tt-lab build t7b (glx-tools t7b, engine source `ee8aaa78`, `tt-lab` sha256 `c8c11c89...`). Source: [tenstorrent/regen-glm5.3-flash, branch glm-5.3-flash-galaxy](https://github.com/tenstorrent/regen-glm5.3-flash/tree/glm-5.3-flash-galaxy). Binary and weights: the Hugging Face repo `tenstorrent/GLM-5.3-Flash-TTQ`: folder `glx-tools-t7b/` (`tt-lab`, its firmware built in, `images/` for reference, SHA256SUMS, licenses), run through the loader and C/C++ libraries in `glm-tt-ship2-e492dca3/runtime/` as production does; plus the plan0h TTQ in five parts, the MTP sidecar and the vocabulary. |
| Layout per Galaxy | 4 groups x 8 chips x 2 users = 8 slots, MTP, 262,144-position context for every request, `--overload 429 --admit-wait 1 --retry-after 1`, in-process group restart on failure; `GET /tt-liveness` answers 200 |
| Session store | shared by the four groups: 100 GB of RAM per group (`--session-ram-gb 100`) and a 512 GB SSD tier per group (`--session-ssd-gb 512`) with its files in `--session-dir` on a local NVMe. The SSD files need 4 x 512 GB = 2 TB of free disk on top of the weights. |
| Media caches | image cache 16 GB (`TT_LAB_SERVE_MEDIA_CACHE_MB=16384`), video cache on a local disk (`TT_LAB_SERVE_VIDEO_CACHE_DIR`). Both act only with the glm-vision worker (`--mm-encoder`), which the package does not include. |
| Tool calls | `TT_LAB_GLM_TOOLS_STRICT=1`, `TT_LAB_GLM_TOOL_REPAIR=1` |
| Router | vllm-project/router 3451e66 + `router-media-placeholders.patch` only, `cache_aware` thr 0.3, abs 8, rel 2, 5 retries, `VLLM_ROUTER_MEDIA_PLACEHOLDERS=1`, `VLLM_ROUTER_RANDOM_TIES=1`. A three-patch build (`PATCHES=route3`: adds port sharing and retry-exclude) is provided and tested but not serving yet; see [glm-router/README.md](glm-router/README.md) |
| Admission | a cap in front of the router at 24 concurrent requests (75% of the 32 slots) |

## 1. Serve on each Galaxy

Through `run.py` (the dev catalog row carries the production settings):

```
python3 run.py --dev-mode --model GLM-5.3-Flash --tt-device blackhole_galaxy --workflow server --docker-server
```

Host setup downloads `tenstorrent/GLM-5.3-Flash-TTQ` into the persistent volume (about 190 GB). The launcher,
`src/run_glm_tt.py`, checks `glx-tools-t7b/tt-lab` and `glm-tt-ship2-e492dca3/runtime/` against their folders'
`SHA256SUMS` (`GLM_TT_PACKAGE`, `GLM_TT_BINARY`, `GLM_TT_RUNTIME_PACKAGE`), stages them together, prints the Galaxy's
Ethernet link map and starts `tt-lab serve` through that loader on port 8000, with the bearer key from `VLLM_API_KEY`
or `API_KEY`. `/health` answers 200 once all four
groups serve (about 3-5 minutes). It creates the session directory and the video cache directory. The host needs 600 GB of RAM for the page-cached TTQ and the session store, and about 2.2 TB of free disk on the cache volume (200 GB of weights, 2 TB of session SSD files).

Under Kubernetes, the Helm chart runs the same row with the `tt-lab-glm` image (`tt-lab-glm.Dockerfile`,
`src/glm_tt_entrypoint.py`): `models.GLM-5.3-Flash.media.blackhole_galaxy.tt_lab_glm` in
`charts/tt-inference-server/values.yaml`. The image digest there is still a placeholder until the image is published.

If serve exits 3, a group halted mid-command: reset the chips (`tt-smi -glx_reset`), delete the dirty marker the
launcher names, then restart.

## 2. Build the router

On the router host (any Linux host that reaches every Galaxy's port 8000):

```
cd vllm-tt-metal/glm-router
./build-router.sh ~/glm-router         # rustc 1.99.1 under ~/glm-router/rust; the media-placeholders patch only (production today)
PATCHES=route3 ./build-router.sh ~/glm-router   # the three-patch build: adds port sharing and retry-exclude
```

It prints the binary's sha256. That differs from the one production runs (`2a98718e...` for the default build,
`68ad05bd...` for route3), because cargo hashes the build path and CARGO_HOME into the binary; the code is the same.

## 3. Start the router

```
cp glm-router.env.example ~/glm-router/glm-router.env
# edit: BIND (and BIND_APPROVED=1 to listen beyond loopback), WORKERS_FILE, VALIDATION_URLS, KEY_FILE, ALERT_CMD
printf 'http://<galaxy-%s-address>:8000\n' 1 2 3 4 > ~/glm-router/workers
./glm-router.sh start                  # tmux session INSTANCE, restarted 2 s after any exit
./glm-router.sh check                  # router /health, worker list, each worker's /health and render time
```

Point your ingress (with the admission cap) at `BIND:PORT`.

## 4. Hand off to a new router binary or new settings

The no-refusal handoff below needs the port-sharing patch, so it works with a router built with `PATCHES=route3` and
started with `REUSEPORT=1`. The route3 build and this handoff tooling are provided and tested, but the production
router today is the media-placeholders-only build and route3 does not carry production traffic yet.

With `REUSEPORT=1` the live router and the next one share the client port, so nothing is refused:

```
# edit ~/glm-router/glm-router.env (e.g. ROUTER_BIN, or a switch)
./glm-router.sh handoff                # the new router starts in INSTANCE-b on METRICS_PORT_B; the old one finishes
                                       # its streams and exits without restart
./glm-router.sh metrics                # the live metrics port (it alternates at each handoff)
./glm-router.sh status
```

Rollback is the same: restore the previous env file and run `handoff` again. A router that was started without
`REUSEPORT=1` cannot share its port; its first handoff refuses connections for the new router's start time (about
0.3 s plus the wait for every worker's `/health`). Moving traffic to a router on a different port is a change in your
ingress; confirm it moved by watching `vllm_router_requests_total{route="/v1/chat/completions"}` grow on the new
router's metrics port and stop on the old one's.

## 5. Take a Galaxy out and back

```
./glm-router.sh drain http://<galaxy-2-address>:8000      # no new requests; waits for its running ones to end
# ... stop serve, update or reset, start serve, wait for /health ...
./glm-router.sh undrain http://<galaxy-2-address>:8000
```

The other Galaxies keep serving throughout. Draining drops that Galaxy's prefix tree in the router, so its
conversations' next turns prefill again elsewhere.

## 6. The per-server manifest

Every production server runs a read-only sidecar, `glm_manifest.py` (a Python 3 standard-library script), in its own
tmux session on port 8001. It serves `/manifest` (JSON) and `/manifest.md`, which name the code the server runs (repo,
branch, commit, binary hashes), its uptime and live stats, and a scorecard against the regen spec it is linked to
(`models/glm-5.3-flash/openrouter/spec.yaml` in tenstorrent/tt-regen-spec). The sidecar only reads: it does not touch
the serve, the chips or the router.

The tool is in [tenstorrent/regen-glm5.3-flash, branch glm-5.3-flash-galaxy](https://github.com/tenstorrent/regen-glm5.3-flash/tree/glm-5.3-flash-galaxy)
under [`tools/cluster/glx-manifest/`](https://github.com/tenstorrent/regen-glm5.3-flash/tree/glm-5.3-flash-galaxy/tools/cluster/glx-manifest): `glm_manifest.py`,
`glm-manifest-ctl.sh` (start, stop, restart, status), `glm-manifest` (query client), `glm-manifest-spec-update`,
`write-provenance.sh`, `glm-manifest.conf.example` and `tests/`. Its design is in
[`notes/glm-glx-manifest.md`](https://github.com/tenstorrent/regen-glm5.3-flash/blob/glm-5.3-flash-galaxy/notes/glm-glx-manifest.md) on that branch.

Quick start, on each server:

```
cp glm-manifest.conf.example glm-manifest.conf   # set HOST_NAME, BIND (the host's own address), SERVE_URL, the dirs
./glm-manifest-ctl.sh start                       # tmux session glm-manifest; then GET http://<BIND>:8001/manifest or /manifest.md
```

The spec copy goes in `<MANIFEST_DIR>/spec/` (`spec.yaml` plus `spec.meta.json`). The servers hold no GitHub
credentials: `glm-manifest-spec-update` refreshes the copy from GitHub on a machine where `gh` works and installs it on
the servers; the sidecar re-reads it without a restart. The fleet configuration (host names and addresses) is local to
the operator and not published: recreate it from `glm-manifest.conf.example`.
