# gemma-3-4b-it Tenstorrent Support on n300

gemma-3-4b-it is also supported on:

- [n150](gemma-3-4b-it_n150.md)

#### Back links

- [n300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other vlm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy gemma-3-4b-it Inference Server on n300

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal-llama3/README.md) inference engine.

The default model weights for this implementation is `gemma-3-4b-it`, the following weights are supported as well:

- `medgemma-4b-it`

To use these weights simply swap `gemma-3-4b-it` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model gemma-3-4b-it --device n300 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it), [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/c254ee3/models/tt_transformers) |
| tt-metal Commit | `c254ee3` |
| vLLM Commit | `c4f2327` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-c254ee3-c4f2327` |
