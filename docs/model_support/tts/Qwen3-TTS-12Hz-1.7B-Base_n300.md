# Qwen3-TTS-12Hz-1.7B-Base Tenstorrent Support on N300

#### Useful links

- [N300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other tts models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Qwen3-TTS-12Hz-1.7B-Base` is also supported on hardware:

- [N150](Qwen3-TTS-12Hz-1.7B-Base_n150.md)

## Quickstart - Deploy Qwen3-TTS-12Hz-1.7B-Base Inference Server on n300

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

Dev catalog: use `--dev-mode`. Japanese text is auto-detected. Default voice is jim.

**via run.py command**

```bash
python3 run.py --model Qwen3-TTS-12Hz-1.7B-Base --device n300 --workflow server --dev-mode
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) |
| Model Status | 🔵 Functional |
| Max Batch Size | 1 |
| Device mesh | (1, 2) TP=2 |
| Implementation Code | [qwen3-tts](https://github.com/tenstorrent/tt-metal/tree/888379c/models/demos/qwen3_tts) |
| tt-metal Commit | `888379c` |
| Sample rate | 24000 Hz |
