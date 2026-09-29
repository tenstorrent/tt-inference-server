# Llama-3.1-8B-Instruct Tenstorrent Support on P150

#### Useful links

- [P150 details](https://tenstorrent.com/hardware/blackhole)
- [Search other training models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy Llama-3.1-8B-Instruct Inference Server on p150

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server (forge plugin)](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model meta-llama/Llama-3.1-8B-Instruct --device p150 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Implementation Code | [trainer-training-lora](https://github.com/tenstorrent/tt-inference-server/tree/a3a9fb4/tt-media-server/tt_model_runners/forge_training_runners/trainer_training_lora_runner.py) |
| tt-metal Commit | `a3a9fb4` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server-forge:0.22.0-a3a9fb4` |
