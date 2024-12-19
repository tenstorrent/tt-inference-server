#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

set -euo pipefail  # Exit on error, print commands, unset variables treated as errors, and exit on pipeline failure

# Function to display usage information
usage() {
    echo "Usage: $0 <model_type>"
    echo "Available model types:"
    echo "  llama-3.3-70b-instruct"
    echo "  llama-3.2-11b-vision-instruct"
    echo "  llama-3.1-70b-instruct"
    echo "  llama-3.1-70b"
    echo "  llama-3.1-8b-instruct"
    echo "  llama-3.1-8b"
    echo "  llama-3-70b-instruct"
    echo "  llama-3-70b"
    echo "  llama-3-8b-instruct"
    echo "  llama-3-8b"
    echo
    exit 1
}

# globals
readonly REPO_ROOT=$(dirname "$(realpath "$0")")

check_and_prompt_env_file() {
    local MODEL_NAME_KEY="MODEL_NAME"
    local MODEL_NAME=""
    # Check if .env file exists
    if [[ -f "${ENV_FILE}" ]]; then
        # Extract the MODEL_NAME value from .env
        echo "found ENV_FILE: ${ENV_FILE}"
        FOUND_MODEL_NAME=$(grep "^$MODEL_NAME_KEY=" "$ENV_FILE" | cut -d '=' -f2) || FOUND_MODEL_NAME=""
        # If MODEL_NAME is found, display it
        if [[ -n "$FOUND_MODEL_NAME" ]]; then
            echo "The existing file ${ENV_FILE} contains MODEL_NAME: $FOUND_MODEL_NAME"
            # Prompt the user to overwrite or exit
            local choice=""
            read -p "Do you want to overwrite the existing file ${ENV_FILE}? (y/n) [default: y]:" choice
            choice=${choice:-y}
            # Handle user's choice
            case "$choice" in
                y|Y )
                    echo "Overwriting the ${ENV_FILE} file ..."
                    # Logic to overwrite .env goes here
                    OVERWRITE_ENV=true
                    ;;
                n|N )
                    OVERWRITE_ENV=false
                    ;;
                * )
                    echo "⛔ Invalid option. Exiting."
                    exit 1
                    ;;
            esac
        else
            echo "MODEL_NAME not found in ${ENV_FILE}. Overwritting."
            OVERWRITE_ENV=true
        fi
    else
        echo "${ENV_FILE} does not exist. Proceeding to create a new one."
        OVERWRITE_ENV=true
    fi
}

get_hf_env_vars() {
    # get HF_TOKEN
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "HF_TOKEN environment variable is not set. Please set it before running the script."
        read -r -s -p "Enter your HF_TOKEN: " input_hf_token
        echo
        if [ -z "${input_hf_token:-}" ]; then
            echo "⛔ HF_TOKEN cannot be empty. Please try again."
            exit 1
        elif [[ ! "$input_hf_token" == hf_* ]]; then
            echo "⛔ HF_TOKEN must start with 'hf_'. Please try again."
            exit 1
        fi
        HF_TOKEN=${input_hf_token}
        echo "✅ HF_TOKEN set."
    fi
    # get HF_HOME
    if [ -z "${HF_HOME:-}" ]; then
        echo "HF_HOME environment variable is not set. Please set it before running the script."
        read -r -p "Enter your HF_HOME [default: $HOME/.cache/huggingface]:" input_hf_home
        echo
        input_hf_home=${input_hf_home:-"$HOME/.cache/huggingface"}
        if [ ! -d "$input_hf_home" ] || [ ! -w "$input_hf_home" ]; then
            echo "⛔ HF_HOME must be a valid directory and writable by the user. Please try again."
            exit 1
        fi
        HF_HOME=${input_hf_home}
        echo "✅ HF_HOME set."
    fi
}

# Function to set environment variables based on the model selection and write them to .env
setup_model_environment() {
    # Set environment variables based on the model selection
    # note: MODEL_NAME is the lower cased basename of the HF repo ID
    case "$1" in
        "llama-3.3-70b-instruct")
        MODEL_NAME="llama-3.3-70b-instruct"
        HF_MODEL_REPO_ID="meta-llama/Llama-3.3-70B-Instruct"
        META_MODEL_NAME=""
        META_DIR_FILTER=""
        REPACKED=1
        ;;
        "llama-3.2-11b-vision-instruct")
        MODEL_NAME="llama-3.2-11b-vision-instruct"
        HF_MODEL_REPO_ID="meta-llama/Llama-3.2-11B-Vision-Instruct"
        META_MODEL_NAME=""
        META_DIR_FILTER=""
        REPACKED=0
        ;;
        "llama-3.1-70b-instruct")
        MODEL_NAME="llama-3.1-70b-instruct"
        HF_MODEL_REPO_ID="meta-llama/Llama-3.1-70B-Instruct"
        META_MODEL_NAME="Meta-Llama-3.1-70B-Instruct"
        META_DIR_FILTER="llama3_1"
        REPACKED=1
        ;;
        "llama-3.1-70b")
        MODEL_NAME="llama-3.1-70b"
        HF_MODEL_REPO_ID="meta-llama/Llama-3.1-70B"
        META_MODEL_NAME="Meta-Llama-3.1-70B"
        META_DIR_FILTER="llama3_1"
        REPACKED=1
        ;;
        "llama-3.1-8b-instruct")
        MODEL_NAME="llama-3.1-8b-instruct"
        HF_MODEL_REPO_ID="meta-llama/Llama-3.1-8B-Instruct"
        META_MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
        META_DIR_FILTER="llama3_1"
        REPACKED=0
        ;;
        "llama-3.1-8b")
        MODEL_NAME="llama-3.1-8b"
        HF_MODEL_REPO_ID="meta-llama/Llama-3.1-8B"
        META_MODEL_NAME="Meta-Llama-3.1-8B"
        META_DIR_FILTER="llama3_1"
        REPACKED=0
        ;;
        "llama-3-70b-instruct")
        MODEL_NAME="llama-3-70b-instruct"
        HF_MODEL_REPO_ID="meta-llama/Llama-3-70B-Instruct"
        META_MODEL_NAME="Meta-Llama-3-70B-Instruct"
        META_DIR_FILTER="llama3"
        REPACKED=1
        ;;
        "llama-3-70b")
        MODEL_NAME="llama-3-70b"
        HF_MODEL_REPO_ID="meta-llama/Llama-3-70B"
        META_MODEL_NAME="Meta-Llama-3-70B"
        META_DIR_FILTER="llama3"
        REPACKED=1
        ;;
        "llama-3-8b-instruct")
        MODEL_NAME="llama-3-8b-instruct"
        HF_MODEL_REPO_ID="meta-llama/Llama-3-8B-Instruct"
        META_MODEL_NAME="Meta-Llama-3-8B-Instruct"
        META_DIR_FILTER="llama3"
        REPACKED=0
        ;;
        "llama-3-8b")
        MODEL_NAME="llama-3-8b"
        HF_MODEL_REPO_ID="meta-llama/Llama-3-8B"
        META_MODEL_NAME="Meta-Llama-3-8B"
        META_DIR_FILTER="llama3"
        REPACKED=0
        ;;
        *)
        echo "⛔ Invalid model choice."
        usage
        exit 1
        ;;
    esac
    # Initialize OVERWRITE_ENV
    OVERWRITE_ENV=false

    # Set default values for environment variables
    DEFAULT_PERSISTENT_VOLUME_ROOT=${REPO_ROOT}/persistent_volume
    MODEL_ENV_DIR="${DEFAULT_PERSISTENT_VOLUME_ROOT}/model_envs"
    
    mkdir -p ${MODEL_ENV_DIR}
    ENV_FILE="${MODEL_ENV_DIR}/${MODEL_NAME}.env"
    export ENV_FILE
    check_and_prompt_env_file


    if [ "$OVERWRITE_ENV" = false ]; then
        echo "✅ using existing .env file: ${ENV_FILE}."
        return 0
    fi
    # Safely handle potentially unset environment variables using default values
    PERSISTENT_VOLUME_ROOT=${PERSISTENT_VOLUME_ROOT:-$DEFAULT_PERSISTENT_VOLUME_ROOT}
    # Prompt user for PERSISTENT_VOLUME_ROOT if not already set or use default
    read -r -p "Enter your PERSISTENT_VOLUME_ROOT [default: ${DEFAULT_PERSISTENT_VOLUME_ROOT}]: " INPUT_PERSISTENT_VOLUME_ROOT
    PERSISTENT_VOLUME_ROOT=${INPUT_PERSISTENT_VOLUME_ROOT:-$PERSISTENT_VOLUME_ROOT}
    echo # move to a new line after input   
    # Set environment variables with defaults if not already set
    PERSISTENT_VOLUME=${PERSISTENT_VOLUME_ROOT}/volume_id_tt-metal-${MODEL_NAME}v0.0.1
    

    read -p "Use 🤗 Hugging Face authorization token for downloading models? Alternative is direct authorization from Meta. (y/n) [default: y]: " input_use_hf_token
    choice_use_hf_token=${input_use_hf_token:-"y"}
    echo # move to a new line after input
    # Handle user's choice
    case "$choice_use_hf_token" in
        y|Y )
            echo "Using 🤗 Hugging Face Token."
            get_hf_env_vars
            # default location for HF e.g. ~/.cache/huggingface/models/meta-llama/Llama-3.3-70B-Instruct
            # LLAMA_WEIGHTS_DIR=${HF_HOME}/local_dir/${HF_MODEL_REPO_ID}
            WEIGHTS_DIR=${PERSISTENT_VOLUME}/model_weights/${MODEL_NAME}
            ;;
        n|N )
            if [ -z "${META_DIR_FILTER:-}" ]; then
                echo "⛔ MODEL_NAME=${MODEL_NAME} does not support using direct Meta authorization model download. Please use Hugging Face method."
            fi
            echo "Using direct authorization from Meta. You will need their URL Authorization token, typically from their website or email."
            # Prompt user for LLAMA_REPO if not already set or use default
            read -r -p "Enter the path where you want to clone the Llama model repository [default: ${LLAMA_REPO}]: " INPUT_LLAMA_REPO
            LLAMA_REPO=${INPUT_LLAMA_REPO:-$LLAMA_REPO}
            LLAMA_DIR=${LLAMA_DIR:-${LLAMA_REPO}/models/${META_DIR_FILTER}}
            LLAMA_WEIGHTS_DIR=${LLAMA_WEIGHTS_DIR:-${LLAMA_DIR}/${META_MODEL_NAME}}
            echo  # move to a new line after input
            ;;
        * )
            echo "⛔ Invalid option. Exiting."
            exit 1
            ;;
    esac

    # Prompt user for JWT_SECRET securely
    read -sp "Enter your JWT_SECRET: " JWT_SECRET
    echo  # move to a new line after input
    # Verify the JWT_SECRET is not empty
    if [ -z "${JWT_SECRET:-}" ]; then
        echo "⛔ JWT_SECRET cannot be empty. Please try again."
        exit 1
    fi

    if [ "${REPACKED}" -eq 1 ]; then
        echo "REPACKED is enabled."
        REPACKED_STR="repacked-"
    else
        echo "REPACKED is disabled."
        REPACKED_STR=""
    fi

    # Write environment variables to .env file
    echo "Writing environment variables to ${ENV_FILE} ..."
    cat > ${ENV_FILE} <<EOF
# Environment variables for the model setup
USE_HF_DOWNLOAD=$choice_use_hf_token
MODEL_NAME=$MODEL_NAME
META_MODEL_NAME=$META_MODEL_NAME
HF_MODEL_REPO_ID=$HF_MODEL_REPO_ID
HOST_HF_HOME=${HF_HOME:-""}
# host paths
LLAMA_REPO=${LLAMA_REPO:-""}
LLAMA_DIR=${LLAMA_DIR:-""}
LLAMA_WEIGHTS_DIR=${LLAMA_WEIGHTS_DIR:-""}
PERSISTENT_VOLUME_ROOT=$PERSISTENT_VOLUME_ROOT
PERSISTENT_VOLUME=$PERSISTENT_VOLUME
WEIGHTS_DIR=${WEIGHTS_DIR:-""}
# container paths
REPACKED=${REPACKED}
REPACKED_STR=${REPACKED_STR}
CACHE_ROOT=/home/user/cache_root
HF_HOME=/home/user/cache_root/huggingface
MODEL_WEIGHTS_ID=id_${REPACKED_STR}$MODEL_NAME
MODEL_WEIGHTS_PATH=/home/user/cache_root/model_weights/${REPACKED_STR}$MODEL_NAME
LLAMA_VERSION=llama3
TT_METAL_ASYNC_DEVICE_QUEUE=1
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
SERVICE_PORT=7000
LLAMA3_CKPT_DIR=/home/user/cache_root/model_weights/${REPACKED_STR}$MODEL_NAME
LLAMA3_TOKENIZER_PATH=/home/user/cache_root/model_weights/${REPACKED_STR}$MODEL_NAME/tokenizer.model
LLAMA3_CACHE_PATH=/home/user/cache_root/tt_metal_cache/cache_${REPACKED_STR}$MODEL_NAME
# These are secrets and must be stored securely for production environments
JWT_SECRET=$JWT_SECRET
HF_TOKEN=${HF_TOKEN:-""}
EOF

    echo "Environment variables written to: ${ENV_FILE}"
    echo "✅ setup_model_environment completed!"
}

# Function to load environment variables from .env file
load_env() {
    if [ -f ${ENV_FILE} ]; then
        echo "Sourcing environment variables from ${ENV_FILE} file..."
        source ${ENV_FILE}
    else
        echo "⛔ ${ENV_FILE} file not found. Please run the setup first."
        exit 1
    fi
}

# SUDO PORTION: Encapsulated in a function to handle all sudo-requiring tasks
setup_permissions() {
    # Load environment variables from .env
    load_env

    echo "Running sudo-required commands..."
    # Create group 'dockermount' if it doesn't exist
    if ! getent group dockermount > /dev/null 2>&1; then
        echo "Creating group 'dockermount' ..."
        sudo groupadd dockermount
    else
        echo "Group 'dockermount' already exists."
    fi

    # Add host user to 'dockermount' group
    echo "Adding user: '$USER' to 'dockermount' group ..."
    sudo usermod -aG dockermount "$USER"

    # Get container user with UID 1000 and add to group
    CONTAINER_UID=1000
    CONTAINER_USER=$(getent passwd ${CONTAINER_UID} | cut -d: -f1)
    if [ -n "$CONTAINER_USER" ]; then
        echo "Adding container user: '$CONTAINER_USER' (UID ${CONTAINER_UID}) to 'dockermount' group ..."
        sudo usermod -aG dockermount "$CONTAINER_USER"
    else
        echo "No user found with UID ${CONTAINER_UID}."
    fi

    # Set file ownership and permissions
    echo "Setting file ownership and permissions for container and host access ..."
    if [ ! -d "${PERSISTENT_VOLUME}" ]; then
        # if the user point the PERSISTENT_VOLUME
        sudo mkdir -p "${PERSISTENT_VOLUME}"
    fi
    sudo chown -R ${CONTAINER_UID}:dockermount "${PERSISTENT_VOLUME}"
    sudo chmod -R 775 "${PERSISTENT_VOLUME}"

    echo "✅ setup_permissions completed!"
}

# Shared function for repacking weights
repack_weights() {
    local source_dir="$1"
    local target_dir="$2"

    # Create target directory if it doesn't exist
    mkdir -p "${target_dir}"

    # Copy required files
    cp "${source_dir}/tokenizer.model" "${target_dir}/tokenizer.model"
    cp "${source_dir}/params.json" "${target_dir}/params.json"
    
    # Set up Python environment for repacking
    VENV_NAME=".venv_repack"
    echo "Setting up python venv for repacking: ${VENV_NAME}"
    python3 -m venv ${VENV_NAME}
    source ${VENV_NAME}/bin/activate
    pip install --upgrade setuptools wheel pip==21.2.4 tqdm
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.1
    
    # Download repacking script
    curl -O https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/models/demos/t3000/llama2_70b/scripts/repack_weights.py
    
    echo "Repacking weights..."
    python repack_weights.py "${source_dir}" "${target_dir}" 5
    
    # Cleanup
    deactivate
    rm -rf ${VENV_NAME} repack_weights.py
    
    echo "✅ Weight repacking completed!"
}

setup_weights_meta() {
    # Step 1: Set up Llama model repository path
    echo "Using repository path: $LLAMA_REPO"

    # Step 2: Clone the repository (if it doesn't already exist)
    if [ ! -d "$LLAMA_REPO" ]; then
        echo "Cloning the Llama repository to: $LLAMA_REPO"
        git clone https://github.com/meta-llama/llama-models.git "$LLAMA_REPO"
        cd "$LLAMA_REPO"
        # checkout commit before ./download.sh was removed
        git checkout 685ac4c107c75ce8c291248710bf990a876e1623
    else
        echo "🔔 Llama repository already exists at $LLAMA_REPO"
    fi

    # Step 3: Check if weights are already downloaded
    if [ -d "${LLAMA_WEIGHTS_DIR}" ] && [ "$(ls -A "${LLAMA_WEIGHTS_DIR}")" ]; then
        echo "Weights already downloaded at ${LLAMA_WEIGHTS_DIR}"
        echo "Skipping download."
    else
        # Step 4: Run the download script and select models
        echo "Running the download script to download models at ${LLAMA_DIR}/download.sh ..."
        cd "$LLAMA_DIR"
        ./download.sh
        cd -
    fi

    # Step 5: Copy weights to persistent volume
    echo "Setting up persistent volume root: ${PERSISTENT_VOLUME}"
    mkdir -p "${PERSISTENT_VOLUME}/model_weights/"

    if [ "${REPACKED}" -eq 1 ]; then
        WEIGHTS_DIR="${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME}"
        repack_weights "${LLAMA_WEIGHTS_DIR}" "${WEIGHTS_DIR}"
    else
        WEIGHTS_DIR="${PERSISTENT_VOLUME}/model_weights/${MODEL_NAME}"
        cp -rf "${LLAMA_WEIGHTS_DIR}" "${WEIGHTS_DIR}"
    fi

    echo "using weights directory: ${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME}"
    echo "✅ setup_weights_meta completed!"
}

setup_weights_huggingface() {
    # Step 1: Verify HF_TOKEN and HF_HOME are set
    if [ -z "${HF_TOKEN:-}" ] || [ -z "${HOST_HF_HOME:-}" ]; then
        echo "⛔ HF_TOKEN or HF_HOME not set. Please ensure both environment variables are set."
        exit 1
    fi

    # Step 2: Set up persistent volume root
    echo "Setting up persistent volume root: ${PERSISTENT_VOLUME}"
    mkdir -p "${PERSISTENT_VOLUME}/model_weights/"

    # Step 3: Create python virtual environment for huggingface downloads
    VENV_NAME=".venv_hf_setup"
    echo "Setting up python venv for Hugging Face downloads: ${VENV_NAME}"
    python3 -m venv ${VENV_NAME}
    source ${VENV_NAME}/bin/activate

    # Step 4: Install required packages
    pip install --upgrade pip setuptools wheel
    pip install "huggingface_hub[cli]"

    # Step 5: Download model using huggingface-cli
    echo "Downloading model from Hugging Face Hub..."
    # stop timeout issue: https://huggingface.co/docs/huggingface_hub/en/guides/cli#download-timeout
    export HF_HUB_DOWNLOAD_TIMEOUT=60
    # using default HF naming convention for model weights
    huggingface-cli download "${HF_MODEL_REPO_ID}" \
        original/params.json \
        original/tokenizer.model \
        original/consolidated.* \
        --cache-dir="${HOST_HF_HOME}" \
        --token="${HF_TOKEN}"

    # symlinks are broken for huggingface-cli download with --local-dir option
    # see: https://github.com/huggingface/huggingface_hub/pull/2223
    # to use symlinks, find most recent snapshot and create symlink to that
    mkdir -p "${WEIGHTS_DIR}"
    LOCAL_REPO_NAME=$(echo "${HF_MODEL_REPO_ID}" | sed 's|/|--|g')
    SNAPSHOT_DIR="${HOST_HF_HOME}/models--${LOCAL_REPO_NAME}/snapshots"
    # note: ls -td will sort by modification date descending, potential edge case
    # if desired snapshot is not most recent modified or ls sorts differently
    MOST_RECENT_SNAPSHOT=$(ls -td -- ${SNAPSHOT_DIR}/* | head -n 1)
    echo "create symlink: ${MOST_RECENT_SNAPSHOT}/original/ -> ${WEIGHTS_DIR}"
    for item in ${MOST_RECENT_SNAPSHOT}/original/*; do
        ln -s "$item" "${WEIGHTS_DIR}"
    done

    # Step 6: Process and copy weights
    if [ "${REPACKED}" -eq 1 ]; then
        REPACKED_WEIGHTS_DIR="${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME}"
        mkdir -p "${REPACKED_WEIGHTS_DIR}"
        repack_weights "${WEIGHTS_DIR}" "${REPACKED_WEIGHTS_DIR}"
    fi

    # Step 7: Cleanup
    deactivate
    rm -rf ${VENV_NAME}

    echo "using weights directory: ${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME}"
    echo "✅ setup_weights_huggingface completed!"
}

setup_tt_metal_cache() {
    # check if tt_metal_cache already exists
    TT_METAL_CACHE_DIR="${PERSISTENT_VOLUME}/tt_metal_cache/cache_${REPACKED_STR}$MODEL_NAME"
    if [ -d "${TT_METAL_CACHE_DIR}" ]; then
        echo "✅ tt_metal_cache already exists at: ${TT_METAL_CACHE_DIR}."
        return 0
    fi

    # create tt_metal_cache directory
    mkdir -p "${TT_METAL_CACHE_DIR}"
    echo "✅ setup_tt_metal_cache completed!"
}

setup_weights() {
    # Step 1: Load environment variables from .env file
    load_env

    # check if model weights already exist
    if [ -d "${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME}" ]; then
        echo "Model weights already exist at: ${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME}"
        echo "contents:"
        echo
        echo "$(ls -lh ${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME})"
        echo
        echo "If directory does not have correct weights, to re-download or copy the model weights delete the directory."
        echo "🔔 check if directory contents are correct."
    else
        # Determine which setup method to use based on HF_TOKEN presence
        if [ "${USE_HF_DOWNLOAD}" == "y" ]; then
            setup_weights_huggingface
        else
            setup_weights_meta
        fi
    fi
    
    echo "create tt-metal cache dir: ${LLAMA3_CACHE_PATH}"
    mkdir -p "${PERSISTENT_VOLUME}/tt_metal_cache/cache_${REPACKED_STR}${MODEL_NAME}"
}

# ==============================================================================
# Main script logic
# ==============================================================================

# Ensure script is being executed, not sourced
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "⛔ Error: This script is being sourced. Please make execute it:"
    echo "chmod +x ./setup.sh && ./setup.sh"
    set +euo pipefail  # Unset 'set -euo pipefail' when sourcing so it doesnt exit or mess up sourcing shell
    return 1;  # 'return' works when sourced; 'exit' would terminate the shell
fi

if [ $# -lt 1 ]; then
    usage
fi

# Set up environment variables for the chosen model
MODEL_TYPE=$1
setup_model_environment "$MODEL_TYPE"
setup_weights
setup_tt_metal_cache
# Call the script again with sudo to execute the sudo-required commands
echo "Switching to sudo portion to set file permissions and complete setup."
setup_permissions
