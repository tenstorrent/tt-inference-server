# tt-smi Docker Image

This document provides instructions for building and running the `tt-smi` utility within a lightweight, self-contained Docker container. The build process is automated via a Python script.

## Prerequisites

Before you begin, ensure you have the following installed on your host machine:
* **Docker Engine**: To build and run the container.
* **Python 3.8+**: To run the build automation script.

## Building the Docker Image

The build process is managed by the Python script located at `scripts/build_smi_docker.py`. This script handles versioning, tagging, and invoking Docker with the correct arguments.

The script should be run from the root of the repository.

### Build Arguments

The script accepts the following command-line arguments:

| Argument                  | Description                                                                                             | Default                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------- |
| `--build-tt-smi-version`  | The specific version of the `tt-smi` pip package to install (e.g., "1.2.3"). If omitted, the latest version will be installed. | `""` (latest)                 |
| `--force-build`           | A flag to force a rebuild of the image, even if an image with the same tag already exists locally or remotely. | `False`                       |
| `--push`                  | A flag to push the image to the `ghcr.io/tenstorrent` registry after a successful build.                 | `False`                       |
| `--ubuntu-version`        | The version of Ubuntu to use as the base.                                                              | `"22.04"`                     |

### Examples

#### 1. Build an Image with a Specific `tt-smi` Version

This is the recommended approach for creating reproducible builds.

```bash
python3 scripts/build_smi_docker.py --build-tt-smi-version "1.2.3"
```
This command will create a Docker image with the following name and tag:
`ghcr.io/tenstorrent/tt-inference-server/tt-smi-ubuntu-22.04-amd64:1.2.3`

#### 2. Build an Image with the Latest `tt-smi` Version

If you don't specify a version, the script will install the latest available version from PyPI and tag the image accordingly.

```bash
python3 scripts/build_smi_docker.py
```
This command will create a Docker image with the following name and tag:
`ghcr.io/tenstorrent/tt-inference-server/tt-smi-ubuntu-22.04-amd64:latest`

#### 3. Force a Rebuild and Push to the Registry

To rebuild an existing image and push it to the remote container registry:
```bash
python3 scripts/build_smi_docker.py --build-tt-smi-version "1.2.3" --force-build --push
```

## Running the tt-smi Container

Once the image is built, you can execute `tt-smi` commands using `docker run`.

> **IMPORTANT**: The `tt-smi` tool needs direct access to the Tenstorrent hardware on the host machine. You must map the device nodes into the container using the `--device` flag. The most common devices are `/dev/tenstorrent` and `/dev/ipmi0`.

### Examples

#### 1. View the Help Menu

You can run the container without any arguments to see the default help output. No device mapping is needed for this command.
```bash
docker run --rm -it ghcr.io/tenstorrent/tt-inference-server/tt-smi-ubuntu-22.04-amd64:1.2.3
```

#### 2. List All Devices

To list Tenstorrent devices, you must map the `/dev/tenstorrent` devices.
```bash
docker run --rm -it --device=/dev/tenstorrent ghcr.io/tenstorrent/tt-inference-server/tt-smi-ubuntu-22.04-amd64:1.2.3 -l
```

#### 3. Perform a Galaxy Reset

Some systems, like the Tenstorrent Galaxy products, require access to both the Tenstorrent devices and the IPMI device.
```bash
docker run --rm -it \
  --device=/dev/tenstorrent \
  --device=/dev/ipmi0 \
  ghcr.io/tenstorrent/tt-inference-server/tt-smi-ubuntu-22.04-amd64:1.2.3 -glx_reset
```

## Image Naming Convention

The build script generates image names using a standardized format:

`<registry>/<image-name>-<os>-<version>-<arch>:<tt-smi-version>`

For example: `ghcr.io/tenstorrent/tt-inference-server/tt-smi-ubuntu-22.04-amd64:1.2.3`

- **Registry**: `ghcr.io`
- **Image Name**: `tenstorrent/tt-inference-server/tt-smi`
- **OS/Version/Arch**: `ubuntu-22.04-amd64`
- **Tag (tt-smi Version)**: `1.2.3` or `latest`
