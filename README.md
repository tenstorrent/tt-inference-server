[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tenstorrent/tt-inference-server)

# tt-inference-server

`tt-inference-server` is the fastest way to deploy and test models for serving inference on Tenstorrent hardware.

## Quickstart guide

On first-run please see the [prerequisites](docs/prerequisites.md) guide for general Tenstorrent hardware and software setup.

For the specific quickstart guide and details for your model, select your model and hardware configuration in Model Support pages and tables below. Alternatively you can see all models supported for your given Tenstorrent hardware.

<!-- MODEL_SUPPORT_START -->
### Models by Model Type

Browse models by type:

- [LLM Models](docs/model_support/llm_models.md) - Large Language Models
- [CNN Models](docs/model_support/cnn_models.md) - Convolutional Neural Networks
- [Audio Models](docs/model_support/audio_models.md) - Speech-to-text models
- [Image Models](docs/model_support/image_models.md) - Image generation models
- [Embedding Models](docs/model_support/embedding_models.md) - Text embedding models
- [TTS Models](docs/model_support/tts_models.md) - Text-to-speech models
- [Video Models](docs/model_support/video_models.md) - Video generation models
- [VLM Models](docs/model_support/vlm_models.md) - Vision-Language Models

### Models by Hardware

Browse models by hardware type:

- [n150 (N150)](docs/model_support/models_by_hardware.md#n150-n150)
- [n300 (N300)](docs/model_support/models_by_hardware.md#n300-n300)
- [TT-LoudBox (T3K)](docs/model_support/models_by_hardware.md#tt-loudbox-t3k)
- [Tenstorrent Galaxy (GALAXY)](docs/model_support/models_by_hardware.md#tenstorrent-galaxy-galaxy)
- [Tenstorrent Galaxy (GALAXY_T3K)](docs/model_support/models_by_hardware.md#tenstorrent-galaxy-galaxy_t3k)
- [p100 (P100)](docs/model_support/models_by_hardware.md#p100-p100)
- [p150 (P150)](docs/model_support/models_by_hardware.md#p150-p150)
- [4xp150 (P150X4)](docs/model_support/models_by_hardware.md#4xp150-p150x4)
- [8xp150 (P150X8)](docs/model_support/models_by_hardware.md#8xp150-p150x8)
<!-- MODEL_SUPPORT_END -->

# Workflow automation in tt-inference-server

For details on the workflow automation for:
- deploying inference servers
- running E2E performance benchmarks
- running accuracy evals

See:
- [Model Readiness Workflows User Guide](docs/workflows_user_guide.md).
- [workflows/README.md](workflows/README.md)

## Benchmarking

For more details see [benchmarking/README.md](benchmarking/README.md)

## Evals

For more details see [evals/README.md](evals/README.md)

## Development

Developer documentation: [docs/README.md](docs/README.md)

Release documentation: [scripts/release/README.md](scripts/release/README.md)

If you encounter setup or stability problems with any model please [file an issue](https://github.com/tenstorrent/tt-inference-server/issues/new?template=Blank+issue) and our team will address it.
