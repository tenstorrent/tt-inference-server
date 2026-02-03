# Prerequisites

## Hardware 

Install and configure Tenstorrent hardware following the official documentation: https://docs.tenstorrent.com

## System Software

Install system software using `tt-installer`, see guide at: https://docs.tenstorrent.com/getting-started/README.html#running-the-installer-script

This should install:
- Firmware: (https://github.com/tenstorrent/tt-zephyr-platforms)
- KMD (https://github.com/tenstorrent/tt-kmd)
- tt-smi: https://github.com/tenstorrent/tt-smi
- Hugepages
- tt-topology: https://github.com/tenstorrent/tt-topology (only needed on WH TT-LoudBox or TT-QuietBox, set `mesh` topology)

For more information see: See: https://docs.tenstorrent.com/getting-started/README#install-system-software-dependencies

Verify your setup post-install, check expected devices are detected and available with correct firmware and tt-kmd:
```bash
# you will need to source the associated python venv, by default:
source ~/.tenstorrent-venv/bin/activate
tt-smi
```

If you have any difficulties installing system software using tt-installer, please file an issue in https://github.com/tenstorrent/tt-installer/issues with relevant logs.

If running on a TT-LoudBox or TT-QuietBox, you will also need to set up `mesh` topology, see https://github.com/tenstorrent/tt-topology?tab=readme-ov-file#mesh

## tt-inference-server Software Requirements

- **Docker**: Required for running inference server Docker images. Full Podman support is experimental and not guaranteed. It is recommended to 
- **Python 3.8+**: this is the python version used to run `run.py`. Other versions will likely work, but 3.8 and 3.10 are tested with.

## HuggingFace Access

Many models require a HuggingFace token for gated model weights:

1. Create a token at https://huggingface.co/settings/tokens
2. Accept terms for required models (e.g., [Llama 3](https://huggingface.co/meta-llama))
3. The token will be requested on first run of `run.py`


## [optional] CPU performance setting

For peak performance for some models CPU frequency profile is recommended. If you cannot do this for your setup, it is optional and can be skipped, though performance may be slightly lower than otherwise expected for some models.

```
sudo apt-get update && sudo apt-get install -y linux-tools-generic
# enable perf mode
sudo cpupower frequency-set -g performance

# disable perf mode (if desired later)
# sudo cpupower frequency-set -g ondemand
```
