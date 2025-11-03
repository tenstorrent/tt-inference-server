# How to run a model on a specific tt-metal (or other dependency)

## Using Docker for tt-metal

1. Add the commit as the tt-metal for your model in `model_spec.py`

Open the file `model_spec.py` and edit the ModelSpecTemplate for the model you want to run on specific commit, for example with LLama 3.3 70B on LoudBox/QuietBox (T3K:)
```
ModelSpecTemplate(
        weights=[
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.1-70B",
            "meta-llama/Llama-3.1-70B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ],
        impl=tt_transformers_impl,
        system_requirements=SystemRequirements(
            firmware=VersionRequirement(
                specifier=">=18.8.0",
                mode=VersionMode.SUGGESTED,
            ),
            kmd=VersionRequirement(
                specifier=">=2.2.0",
                mode=VersionMode.SUGGESTED,
            ),
        ),
        tt_metal_commit="9b67e09",
        vllm_commit="a91b644",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
                env_vars={
                    "MAX_PREFILL_CHUNK_SIZE": "32",
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
                },
            ),
        ],
        status=ModelStatusTypes.FUNCTIONAL,
    ),
```

You would need to edit `tt_metal_commit="9b67e09"` to be `tt_metal_commit="<my-specific-commit>"`

2. Build tt-metal docker image with that commit:
```bash
python3 scripts/build_docker_images.py --build-metal-commit <my-specific-commit>
```
