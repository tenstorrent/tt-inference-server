# Running the Qwen3.6-27B server (tt-inference-server, Blackhole Galaxy)

How to serve Qwen3.6-27B via tt-inference-server using the prebuilt docker image.

## Prerequisites

- **Branch**: tt-inference-server checked out at `ssinghal/qwen36-long-context`
  (carries the Qwen3.6 entry in `workflows/model_specs/dev/llm.yaml` + benchmark/eval config).
- **Docker access**: be in the `docker` group, or prefix every docker/`run.py` command with
  `sg docker -c "..."`.
- **Device**: Blackhole Galaxy up, firmware bundle **19.10.0.0** (`eth_fw 1.10.1`). Verify:
  `tt-smi -s` → `board_type: tt-galaxy-bh`, `dram_status: true`. (Older firmware causes
  nondeterministic hangs.)
- **Image**: the final self-contained image, which has tt-metal + the Python padfix +
  `tt_vllm_plugin` baked in:
  ```
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.15.0-qwen36
  ```
  > The bare base image (no `tt_vllm_plugin`) fails at startup with a `qwen3_5` AutoConfig
  > error. If you only have the base, build the full image once:
  > `tt-inference-server/scripts/build_qwen36_image.sh`

## Launch command (same for warm or cold start)

```bash
cd <repo>/tt-inference-server          # branch ssinghal/qwen36-long-context

python3 run.py --workflow server \
  --model Qwen3.6-27B \
  --tt-device blackhole_galaxy \
  --docker-server \
  --dev-mode \
  --no-auth \
  --host-volume "$(pwd)/persistent_volume" \
  --override-docker-image ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.15.0-qwen36
```
(Prefix with `sg docker -c "..."` if not in the docker group.)

The command launches a **detached** container and returns immediately, printing the
container ID. Watch startup with `docker logs -f <id>`.

### Required (non-default) flags — why
| Flag | Why |
|---|---|
| `--dev-mode` | Resolves the spec from `workflows/model_specs/dev/llm.yaml` (Qwen3.6 `override_tt_config`: `trace_mode: true`, `sample_on_device_mode: decode_only`, `fabric_config: FABRIC_1D_RING`, `QWEN36_*` env). Without it the prod catalog has no Qwen3.6 entry → fails. |
| `--override-docker-image` | `run.py` otherwise derives a non-existent `-release-…:0.11.1-fa021238662-8f36910` tag from the spec's commit hashes. The override points it at the real image. |
| `--no-auth` | Skips the JWT requirement. Otherwise set `JWT_SECRET` in `.env`. |
| `--host-volume` | Bind-mounts the weights/cache volume (parent dir of the version-keyed `volume_id_*`). Mutually exclusive with `--host-hf-cache`. |

## Case A — warm start (weights + TT cache already present)

The volume must hold `volume_id_qwen3_6_galaxy-Qwen3.6-27B-v<version>/` with `weights/` +
`tt_metal_cache/`. Make it writable by the container's uid 1000, and symlink the version
name if the cache version differs from the image version:
```bash
chmod -R a+rwX persistent_volume/
# only if your cache dir version != image version (e.g. cache v0.11.1, image v0.15.0):
ln -sfn volume_id_qwen3_6_galaxy-Qwen3.6-27B-v0.11.1 \
        persistent_volume/volume_id_qwen3_6_galaxy-Qwen3.6-27B-v0.15.0
```
Then run the launch command. Warm startup ≈ **5–6 min** (load 64 layers + warmup).

## Case B — cold start (no HF weights, no TT cache; download on first run)

The server auto-downloads on first run: `snapshot_download(repo_id="Qwen/Qwen3.6-27B", …)`
into the volume, and the TT tensor/kernel cache is generated on first warmup.

```bash
# 1. HF token (REQUIRED — without it the download 401s; accept any model gating on HF first)
echo 'HF_TOKEN=hf_xxxxxxxx' >> .env

# 2. empty, writable volume with enough free disk (~80–100 GB: fp16 + repack + tt cache)
mkdir -p persistent_volume && chmod -R a+rwX persistent_volume
```
Then run the launch command. No symlink needed — `run.py` creates the correctly
version-keyed dir and downloads into it (persists for next time).
**First run is long: ~30–60+ min** (download tens of GB → repack → first tensor/kernel
cache gen → warmup). Watch `docker logs -f <id>` for `Downloading weights from
Qwen/Qwen3.6-27B …`, then `cache_monitor` first-run activity, then `vLLM service is healthy`.
Every subsequent run is a ~5–6 min warm start.

## Verify
```bash
curl -sf http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.6-27B","prompt":"The capital of France is","max_tokens":12,"temperature":0}'
# expect: " Paris."
```

## Stop
```bash
docker stop <container_id>      # printed by run.py, or from `docker ps`
```

## Notes / known limitations
- **Prefill bucketing**: all prompts ≤4096 tokens are padded to the 4096 bucket
  (`get_padded_prefill_len`); larger prompts pad to the next power of two up to 262144 (256k).
- **256k**: use input length ≤ ~240000 so the prompt + chat-template expansion stays under
  `max_model_len` 262144 (it pads up to the 262144 bucket).
- **Performance** (batch-1): pure decode ≈ 23.5 tok/s (flat to ~32k ISL), ~18 tok/s @256k.
  End-to-end `output_throughput` is lower because OSL=128 amortizes prefill.
- **Decode quality caveat**: coherent on short/easy prompts; **degrades on hard reasoning /
  math content** (deterministic repetition collapse) — an open decode-forward issue. Don't
  rely on it for reasoning benchmarks yet.
