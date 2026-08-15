# Qwen3.6-27B Tenstorrent Support on BH QuietBox 2

#### Useful links

- [BH QuietBox 2 details](https://tenstorrent.com/hardware/tt-quietbox)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Qwen3.6-27B` is also supported on hardware:

- [BH LoudBox](Qwen3.6-27B_p150x8.md)

## Quickstart - Deploy Qwen3.6-27B Inference Server on BH QuietBox 2

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal/README.md) inference engine.

**docker run command**

```bash
docker run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --ipc host \
  --publish 8000:8000 \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume volume_id_Qwen3.6-27B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.18.0-c49bb76-6b4a3a7 \
  --model Qwen3.6-27B \
  --tt-device p300x2
```

> **Caution (build pin):** the pinned `0.18.0-c49bb76-6b4a3a7` image carries the "coherent but incorrect" generation bug for this model ([tt-metal#49513](https://github.com/tenstorrent/tt-metal/issues/49513)). Use a 0.19.0+ build that includes [tt-metal#48861](https://github.com/tenstorrent/tt-metal/pull/48861) (chunked-GDN C++ op) and commit `c355c15b` (weight loading).

**via run.py command**

```bash
python3 run.py --model Qwen3.6-27B --device p300x2 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 4 @ 262144 ctx (see note below) |
| Max Context Length | 262144 |
| Implementation Code | [qwen36-blackhole](https://github.com/tenstorrent/tt-metal/tree/c49bb76/models/demos/blackhole/qwen36) |
| tt-metal Commit | `c49bb76` |
| vLLM Commit | `6b4a3a7` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.18.0-c49bb76-6b4a3a7` |

> **Batch size note:** independently verified on a TT-QuietBox 2 (4×P150): `max_num_seqs=4` at full 262144 ctx decodes 4 requests concurrently (61.87 tok/s aggregate @4; 4-prompt concurrent correctness gate 4/4), while `max_num_seqs=8` at 262144 ctx OOMs in device DRAM (`bank_manager.cpp:462`). Batch=8 is available via the 64k-context spec from [#4706](https://github.com/tenstorrent/tt-inference-server/pull/4706).
