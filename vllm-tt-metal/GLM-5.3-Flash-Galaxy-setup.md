# GLM-5.3-Flash on Galaxy: setup as production runs it

This is the setup serving GLM-5.3-Flash on OpenRouter as of 2026-11-16: four Blackhole Galaxies, each running one
`glm-tt serve`, behind one router.

## What runs

| | |
| --- | --- |
| Engine | glm-tt build t6 (tt-lab glx-tools t6, engine source `b12c169e`). Source: [tenstorrent/regen-glm5.3-flash, branch glm-5.3-flash-galaxy](https://github.com/tenstorrent/regen-glm5.3-flash/tree/glm-5.3-flash-galaxy). Prebuilt package and weights: the Hugging Face repo `tenstorrent/GLM-5.3-Flash-TTQ` (package folder `glm-tt-t6-b12c169e/`, the plan0h TTQ in five parts, the MTP sidecar, the vocabulary). |
| Layout per Galaxy | 4 groups x 8 chips x 2 users = 8 slots, MTP, 262,144-position context for every request, `--overload 429 --admit-wait 1 --retry-after 1`, in-process group restart on failure |
| Session store | RAM only, shared by the four groups, 100 GB per group (`--session-ram-gb 100`); no SSD tier |
| Media caches | image cache 16 GB (`TT_LAB_SERVE_MEDIA_CACHE_MB=16384`), video cache on a local disk (`TT_LAB_SERVE_VIDEO_CACHE_DIR`). Both act only with the glm-vision worker (`--mm-encoder`), which the package does not include. |
| Tool calls | `TT_LAB_GLM_TOOLS_STRICT=1`, `TT_LAB_GLM_TOOL_REPAIR=1` |
| Router | vllm-project/router 3451e66 + three patches, `cache_aware` thr 0.3, abs 8, rel 2, 5 retries, all four switches on; see [glm-router/README.md](glm-router/README.md) |
| Admission | a cap in front of the router at 24 concurrent requests (75% of the 32 slots) |

## 1. Serve on each Galaxy

Through `run.py` (the dev catalog row carries the production settings):

```
python3 run.py --dev-mode --model GLM-5.3-Flash --tt-device blackhole_galaxy --workflow server --docker-server
```

Host setup downloads `tenstorrent/GLM-5.3-Flash-TTQ` into the persistent volume (about 190 GB); the launcher,
`src/run_glm_tt.py`, verifies the package against its `SHA256SUMS`, prints the Galaxy's Ethernet link map and starts
`glm-tt serve` on port 8000 with the bearer key from `VLLM_API_KEY` or `API_KEY`. `/health` answers 200 once all four
groups serve (about 3-5 minutes). The host needs 600 GB of RAM for the page-cached TTQ and the session store.

Under Kubernetes, the Helm chart runs the same row with the `tt-lab-glm` image (`tt-lab-glm.Dockerfile`,
`src/glm_tt_entrypoint.py`): `models.GLM-5.3-Flash.media.blackhole_galaxy.tt_lab_glm` in
`charts/tt-inference-server/values.yaml`. The image digest there is still a placeholder until the image is published.

If serve exits 3, a group halted mid-command: reset the chips (`tt-smi -glx_reset`), delete the dirty marker the
launcher names, then restart.

## 2. Build the router

On the router host (any Linux host that reaches every Galaxy's port 8000):

```
cd vllm-tt-metal/glm-router
./build-router.sh ~/glm-router         # rustc 1.99.1 under ~/glm-router/rust; applies the three patches in order
```

It prints the binary's sha256. That differs from production's `68ad05bd...`, because cargo hashes the build path and
CARGO_HOME into the binary; the code is the same.

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
