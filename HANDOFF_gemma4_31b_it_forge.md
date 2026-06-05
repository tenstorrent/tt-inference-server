# Handoff: gemma-4-31b-it as a Forge LLM (tensor-parallel) on P300X2

**Goal:** serve `gemma-4-31b-it` through `run.py` as a Forge LLM, 512 seq len / concurrency 1, on
QB2 (2× P300 = `p300x2`, 4 chips, `(1,4)` tensor-parallel mesh) via the tt-media-server forge image.

**Status:** integration **complete and correct**; verified end-to-end up to fabric init. Blocked
only by hardware on the box it was developed on (`qb2-120-p01t06`) — its two P300 cards have no
inter-card ethernet links, so the 4-chip mesh can't form. **This branch should run as-is on a box
whose 4 chips form a connected mesh (e.g. the working `qb2-120-p01t05`).**

## What this branch adds

1. `workflows/model_specs/prod/cnn.yaml` — new Forge LLM spec entry, `weights: google/gemma-4-31b-it`,
   `impl: forge_vllm_plugin`, `inference_engine: FORGE`, device `P300X2`, `max_context: 512`,
   `max_concurrency: 1`, plus a `TT_MESH_GRAPH_DESC_PATH` env override (see learnings).
2. `run_server_gemma4_31b.sh` — launcher (port 8013, `--tt-device p300x2 --engine forge
   --impl forge-vllm-plugin --workflow server --docker-server --no-auth`, with
   `--override-docker-image` = the tt-shield forge image).

## How to run (on the working machine)

```bash
# prereqs: HF_TOKEN exported (gemma-4 is gated); `docker login ghcr.io` with read:packages on tenstorrent/tt-shield
git checkout <this-branch>
./run_server_gemma4_31b.sh
# watch:
docker logs -f $(docker ps -q --filter "publish=8013")
```
Success path in the container log: `Using custom mesh graph descriptor: ...p300_x2...` →
`Physical Mesh ... Degree histogram: {2:4}` → weights load → compile → `Model warmup completed` →
`Uvicorn running on http://0.0.0.0:8000`.

Smoke test once up:
```bash
curl -s localhost:8013/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-31B-it","prompt":"Hello, my name is","max_tokens":32}' | jq
```

## Key learnings (why the spec looks the way it does)

- **Runner is correct and is the TP one.** Server selects `vllm_forge_gemma4_31b` (tt-media-server
  `ModelRunners.VLLMForge_GEMMA4_31B`), which pins `(…, P300X2)` to `max_model_length=512,
  max_num_seqs=1, max_num_batched_tokens=2560`, mesh `(1,4)`, `enable_tensor_parallel=True`,
  `use_2d_mesh=False`, `experimental_weight_dtype="bfp_bf8"`. It's the analogue of the Llama-70B TP
  runner (`VLLMForge_LLAMA_70B`). All of that config is server-side; the tt-inference-server spec
  only routes `MODEL`+`DEVICE` to it.
- **`model_name` MUST be lowercase `gemma-4-31b-it`.** `model_name = Path(weights).name`, and the
  server resolves `ModelNames(MODEL)` with an **exact, case-sensitive** lookup. `ModelNames.GEMMA_4_31B_IT
  == "gemma-4-31b-it"`. An uppercase `B` (the real HF repo is `google/gemma-4-31B-it`) would raise
  ValueError on the server. The container loads the real `google/gemma-4-31B-it` repo internally from
  its own constants, so the lowercase repo string is host-side naming only (no host HF download
  happens without `--host-volume`/`--host-hf-cache`/`--host-weights-dir`).
- **`TT_MESH_GRAPH_DESC_PATH` must point at `p300_x2`.** The gemma runner does NOT auto-set it (only
  whisper/tts do), and the image's `_BH_DEVICE_MESH_DESCRIPTORS` wrongly maps `p300x2 -> p150`. So it
  otherwise inherits a stale single-chip `p150` descriptor from the host `.env` → StableHLO
  `unknown mesh: @mesh` at compile. The spec pins the `p300_x2` descriptor as an `-e` var (overrides
  `--env-file`). Path inside the image:
  `/home/container_app_user/app/server/venv-worker/lib/python3.12/site-packages/pjrt_plugin_tt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto`
- **Docker image (forge, tt-shield, has gemma-4 support):**
  `ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-forge:97aea20e33246455fa81b1d2bc5270093415ee23_fd9296b_79608294867`
- **Inefficiency to optionally fix:** the container re-downloads weights (~8 min) every launch
  because HF's cache (`~/.cache/huggingface`) is outside the mounted `cache_root` volume. Pass
  `--host-hf-cache <dir>` to persist it across runs.

## Reference (known-good)

tt-xla CI test `test_vllm_tp_benchmark[gemma4-31b-it-tp]` (mirrors
`test_tensor_parallel_generation_bhqb_gemma4_31b`) runs this model on a healthy 4-chip Blackhole
QuietBox with bf16. A 2-chip mesh is NOT enough (31B bf16 OOMs); needs the full 4-chip mesh.

## The hardware blocker on p01t06 (not a code issue)

`qb2-120-p01t06` (10.32.48.16) has on-card eth links only (chip 0↔1, 2↔3); the cross-card links
`chip1↔chip2` and `chip0↔chip3` are down/absent → fabric degree `{1:4}` (two isolated pairs) instead
of `{2:4}`. Persists after `tt-smi -r` (4.1.0 and 5.2.0) and after running a fabric workload.
Firmware/config identical to the working `p01t05` (10.32.48.15, degree `{2:4}`). Filed for IT in
`QB2_P300_ETH_FABRIC_HANG_REPORT.md`. Repeatable check (any box):
```
<tt-metal>/build_Release/tools/umd/system_health 2>/dev/null | grep -E "Chip:|link UP"
```
Healthy = a `link UP ... connected to chip` crossing the {0,1} and {2,3} card pairs. `tt-smi -s`
does NOT reveal this (passes on both boxes).

## Qwen3-32B (added as a Forge LLM the same way — follow-up commit)

Qwen3-32B is wired as a Forge LLM TP model mirroring gemma. **Server-side support was added to
this repo's tt-media-server** (it did not exist before):
- `tt-media-server/tt_model_runners/vllm_forge_qwen_32b.py` — `VLLMForgeQwen32BRunner` (text-only
  copy of the gemma runner; same TP additional_config: `enable_tensor_parallel`, `use_2d_mesh=False`,
  `bfp_bf8`, `optimization_level=0`, `cpu_sampling`).
- `tt-media-server/config/constants.py` — `SupportedModels.QWEN_3_32B`, `ModelNames.QWEN_3_32B
  ("Qwen3-32B")`, `ModelRunners.VLLMForge_QWEN_32B`, LLM service map, model-names map, and a
  `(VLLMForge_QWEN_32B, P300X2)` config (mesh (1,4), 512/1, `DEVICE_IDS_4_GROUP`).
- `tt-media-server/tt_model_runners/runner_fabric.py` — runner mapping.
- `workflows/model_specs/prod/cnn.yaml` — spec `weights: Qwen/Qwen3-32B` (basename `Qwen3-32B`
  matches `ModelNames.QWEN_3_32B`; no lowercase issue unlike gemma), P300X2, 512/1, same p300_x2
  mesh-descriptor override.
- `run_server_qwen3_32b_forge.sh` — launcher (port 8014).

**CRITICAL for testing:** the prebuilt gemma image (`97aea20…`) does **NOT** contain the
`VLLMForge_QWEN_32B` runner (it only knows Qwen3-4B/8B, no `QWEN_3_32B`; its generic `(VLLMForge,
P300X2)` config is single-chip `(1,1)`, not TP). So `run_server_qwen3_32b_forge.sh` against that
image will fail the server's `ModelNames("Qwen3-32B")` lookup. To test, you need a forge image
**built from this repo's updated tt-media-server**, OR dev-mount the updated tt-media-server into
the container. Set `IMAGE=` in the launch script accordingly. (Reference that it can work at all:
tt-xla `test_vllm_tp_benchmark[qwen3-32b-qb2-tp]`, `use_2d_mesh=False`.)

Note: Qwen3-32B also already runs on P300X2 via the **native** path (`prod/llm.yaml`,
`impl: tt_transformers`, `inference_engine: VLLM`, v0.14.0) — independent of this forge work.

## Next session TODO (on the working machine)

1. Confirm `system_health` shows a connected 4-chip mesh (cross-card links UP).
2. `./run_server_gemma4_31b.sh`; confirm `Degree histogram: {2:4}` and warmup completes.
3. Smoke-test with the curl above; confirm coherent output.
4. (Optional) add `--host-hf-cache` to skip the weight re-download while iterating.
5. If it serves cleanly, this is PR-ready (the spec + script). Consider whether the entry belongs in
   `prod/cnn.yaml` (current) or `dev/cnn.yaml` given EXPERIMENTAL status.
