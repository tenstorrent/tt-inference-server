#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

set -euo pipefail  # Exit on error, print commands, unset variables treated as errors, and exit on pipeline failure

check_image_exists_remote() {
    local image_tag="$1"
    if docker manifest inspect "${image_tag}" > /dev/null 2>&1; then
        echo "✅ The image exists on GHCR: ${image_tag}"
        return 0
    else
        echo "The image does NOT exist on GHCR: ${image_tag}"
        return 1
    fi
}

check_image_exists_local() {
    local image_tag="$1"
    if docker inspect --type=image "${image_tag}" > /dev/null 2>&1; then
        echo "✅ The image exists locally: ${image_tag}"
        return 0
    else
        echo "The image does NOT exist locally: ${image_tag}"
        return 1
    fi
}

# ==============================================================================
# Main script logic
# ==============================================================================

# Ensure script is being executed, not sourced
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "⛔ Error: This script is being sourced. Please make execute it:"
    echo "chmod +x ./build_docker.sh && ./build_docker.sh"
    set +euo pipefail  # Unset 'set -euo pipefail' when sourcing so it doesnt exit or mess up sourcing shell
    return 1;  # 'return' works when sourced; 'exit' would terminate the shell
fi

# defaults
force_build=false
build=false
push_images=false
UBUNTU_VERSION="20.04"
CONTAINER_APP_UID=1000
# ------------------------------------------------------------------------------
# Process CLI options
# ------------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --force-build)
            force_build=true
            build=true
            ;;
        --build)
            build=true
            ;;
        --push)
            push_images=true
            ;;
        --ubuntu-version)
            if [ $# -lt 2 ]; then
                echo "Error: --ubuntu-version requires a value."
                exit 1
            fi
            UBUNTU_VERSION="$2"
            shift
            ;;
        --container-uid)
            if [ $# -lt 2 ]; then
                echo "Error: --ubuntu-version requires a value."
                exit 1
            fi
            CONTAINER_APP_UID="$2"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

repo_root=$(git rev-parse --show-toplevel)


# validation
# Check if PWD ends with the expected suffix
expected_suffix="tt-inference-server"
if [[ "$repo_root" != *"$expected_suffix" ]]; then
    echo "Error: Script must be run tt-inference-server repo root, the found repo root is: '$repo_root'."
    exit 1
fi
if [[ "$UBUNTU_VERSION" != "22.04" && "$UBUNTU_VERSION" != "20.04" ]]; then
    echo "Error: Unsupported UBUNTU_VERSION: $UBUNTU_VERSION. Only 22.04 and 20.04 are supported."
    exit 1
fi
if [[ "$CONTAINER_APP_UID" =~ ^[0-9]+$ ]] && (( $CONTAINER_APP_UID >= 1000 && $CONTAINER_APP_UID < 60000 )); then
    echo "CONTAINER_APP_UID=${CONTAINER_APP_UID} is within expected range."
else
    echo "CONTAINER_APP_UID=${CONTAINER_APP_UID} is not a number or outside expected range of 1000 to 59999."
fi
cd "$repo_root"

# build image vars
UBUNTU_VERSION="${UBUNTU_VERSION}"
OS_VERSION="ubuntu-${UBUNTU_VERSION}-amd64"
TT_METAL_COMMIT_SHA_OR_TAG=v0.56.0-rc39
TT_METAL_COMMIT_DOCKER_TAG=${TT_METAL_COMMIT_SHA_OR_TAG:0:12}
TT_VLLM_COMMIT_SHA_OR_TAG=3429acf14e46436948db6865b90178c6375d0217
TT_VLLM_COMMIT_DOCKER_TAG=${TT_VLLM_COMMIT_SHA_OR_TAG:0:12}
CONTAINER_APP_UID="${CONTAINER_APP_UID}"
IMAGE_VERSION=$(cat VERSION)
# TODO: use this local source build of tt-metal dev image until
# published images have source builds or without .whl installed
TT_METAL_DOCKERFILE_URL=local/tt-metal/tt-metalium/${OS_VERSION}:${TT_METAL_COMMIT_SHA_OR_TAG}


cloud_image_tag=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-cloud-${OS_VERSION}:${IMAGE_VERSION}-${TT_METAL_COMMIT_DOCKER_TAG}-${TT_VLLM_COMMIT_DOCKER_TAG}
dev_image_tag=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-${OS_VERSION}:${IMAGE_VERSION}-${TT_METAL_COMMIT_DOCKER_TAG}-${TT_VLLM_COMMIT_DOCKER_TAG}

# Initialize flags for whether to build each image locally.
build_cloud_image=true
build_dev_image=true

if [ "$force_build" = true ]; then
    echo "Force build option provided (--force-build). Skipping remote image checks; both all images will be built locally."
else
    # Check for the images independently, negating check_image_exists return
    if check_image_exists_remote "${cloud_image_tag}" || check_image_exists_local "${cloud_image_tag}"; then
        build_cloud_image=false
    fi
    if check_image_exists_remote "${cloud_image_tag}" || check_image_exists_local "${cloud_image_tag}"; then
        build_dev_image=false
    fi
fi

if [ "$build" = true ]; then

    echo "using TT_METAL_DOCKERFILE_URL: ${TT_METAL_DOCKERFILE_URL}"

    if ! check_image_exists_local "${TT_METAL_DOCKERFILE_URL}"; then
        echo "Image ${TT_METAL_DOCKERFILE_URL} does not exist, building it ..."
        # build tt-metal base-image
        mkdir -p temp_docker_build_dir
        cd temp_docker_build_dir
        git clone --depth 1 https://github.com/tenstorrent/tt-metal.git
        cd tt-metal
        git fetch --depth 1 origin tag ${TT_METAL_COMMIT_SHA_OR_TAG}
        git checkout ${TT_METAL_COMMIT_SHA_OR_TAG}
        docker build \
            -t local/tt-metal/tt-metalium/${OS_VERSION}:${TT_METAL_COMMIT_SHA_OR_TAG} \
            --build-arg UBUNTU_VERSION=${UBUNTU_VERSION} \
            --target ci-build \
            -f dockerfile/Dockerfile .
        cd "$repo_root"
        rm -rf temp_docker_build_dir
    fi
    
    # build cloud deploy image
    if [ "$build_cloud_image" = true ]; then
        echo "building: ${cloud_image_tag}"
        cd "$repo_root"
        docker build \
        -t ${cloud_image_tag} \
        --build-arg TT_METAL_DOCKERFILE_URL="${TT_METAL_DOCKERFILE_URL}" \
        --build-arg TT_METAL_COMMIT_SHA_OR_TAG="${TT_METAL_COMMIT_SHA_OR_TAG}" \
        --build-arg TT_VLLM_COMMIT_SHA_OR_TAG="${TT_VLLM_COMMIT_SHA_OR_TAG}" \
        --build-arg CONTAINER_APP_UID="${CONTAINER_APP_UID}" \
        . -f vllm-tt-metal-llama3/vllm.tt-metal.src.cloud.Dockerfile
    else
        echo "skipping, build_cloud_image=${build_cloud_image}"
    fi

    # build dev image
    if [ "$build_dev_image" = true ]; then
        echo "building: ${dev_image_tag}"
        docker build \
        -t "${dev_image_tag}" \
        --build-arg CLOUD_DOCKERFILE_URL="${cloud_image_tag}" \
        . -f vllm-tt-metal-llama3/vllm.tt-metal.src.dev.Dockerfile

        echo "✅ built images:"
        echo "${cloud_image_tag}"
        echo "${dev_image_tag}"
    else
        echo "skipping, build_dev_image=${build_dev_image}"
    fi
else
    echo "to build images use (--build)"
    echo "check mode completed"
fi

# ------------------------------------------------------------------------------
# Push images to Docker Hub if requested
# ------------------------------------------------------------------------------

if [ "${push_images}" = true ]; then
    echo "Pushing images to Docker Hub..."
    if check_image_exists_local "${cloud_image_tag}" && ! check_image_exists_remote "${cloud_image_tag}"; then
        docker push "${cloud_image_tag}"
    fi

    if check_image_exists_local "${dev_image_tag}" && ! check_image_exists_remote "${dev_image_tag}"; then
        docker push "${dev_image_tag}"
    fi
fi

echo "✅ build_docker.sh completed successfully!"
