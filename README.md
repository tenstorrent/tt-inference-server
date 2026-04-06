[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tenstorrent/tt-inference-server)

# tt-inference-server
<!-- test for tt-studio -->

`tt-inference-server` is the fastest way to deploy and test models for serving inference on Tenstorrent hardware.

## Quickstart guide

On first-run please see the [prerequisites](docs/prerequisites.md) guide for general Tenstorrent hardware and software setup.

For the specific quickstart guide and details for your model, select your model and hardware configuration in Model Support pages and tables below. Alternatively you can see all models supported for your given Tenstorrent hardware.

<!-- MODEL_SUPPORT_START -->
### Models by Model Type

Browse models by type:

- [LLM Models](docs/model_support/llm/README.md) - Large Language Models
- [VLM Models](docs/model_support/vlm/README.md) - Vision-Language Models
- [Video Models](docs/model_support/video/README.md) - Video generation models
- [Image Models](docs/model_support/image/README.md) - Image generation models
- [Audio Models](docs/model_support/audio/README.md) - Speech-to-text models
- [TTS Models](docs/model_support/tts/README.md) - Text-to-speech models
- [Embedding Models](docs/model_support/embedding/README.md) - Text embedding models
- [CNN Models](docs/model_support/cnn/README.md) - Convolutional Neural Networks

### Models by Hardware Configuration

Browse models by hardware:

- [WH Galaxy](docs/model_support/models_by_hardware.md#wh-galaxy)
- [BH LoudBox](docs/model_support/models_by_hardware.md#bh-loudbox)
- [BH QuietBox](docs/model_support/models_by_hardware.md#bh-quietbox)
- [p150](docs/model_support/models_by_hardware.md#p150)
- [p100](docs/model_support/models_by_hardware.md#p100)
- [WH LoudBox/QuietBox](docs/model_support/models_by_hardware.md#wh-loudboxquietbox)
- [n300](docs/model_support/models_by_hardware.md#n300)
- [n150](docs/model_support/models_by_hardware.md#n150)
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
