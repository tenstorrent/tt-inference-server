"""
SGLang TT-Metal Plugin Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sglang-tt-plugin",
    version="0.1.0",
    author="TT-Metal Integration Team",
    author_email="your-email@example.com",
    description="Tenstorrent TT plugin for SGLang",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/sglang-tt-plugin",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
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
            "tt_llama = sglang_tt_plugin.models.tt_llama:TTLlamaForCausalLM",
        ],
    },
    package_data={
        "sglang_tt_plugin": ["*.py"],
    },
)