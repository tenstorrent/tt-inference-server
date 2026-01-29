"""
SGLang TT-Metal Plugin Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sglang-tt-plugin",
    version="0.1.0",
    description="Tenstorrent plugin for SGLang",
    long_description=long_description,
    packages=find_packages(),
    classifiers=[""],
    python_requires=">=3.8",
    install_requires=[
        # NOTE: torch is NOT included here - user must install CPU PyTorch FIRST
        # See INSTALLATION.md for correct installation order
        "ttnn",
        "transformers",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-asyncio",
        ],
    },
    entry_points={
        "console_scripts": [
            "sglang-tt-server = sglang_tt_plugin.scripts.launch_tt_server:main",
        ],
        "sglang.models": [
            "tt_llama = sglang_tt_plugin.models.tt_llm:TTLlamaForCausalLM",
            "tt_qwen = sglang_tt_plugin.models.tt_llm:TTQwenForCausalLM",
            "tt_mistral = sglang_tt_plugin.models.tt_llm:TTMistralForCausalLM",
            "tt_gptoss = sglang_tt_plugin.models.tt_llm:TTGptOssForCausalLM",
        ],
    },
    package_data={
        "sglang_tt_plugin": ["*.py"],
    },
)