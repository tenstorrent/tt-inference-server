#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

set -euo pipefail  # Exit on error, print commands, unset variables treated as errors, and exit on pipeline failure

# Function to display usage information
usage() {
    echo "Usage: $0 <model_type>"
    echo "Available model types:"
    echo "  Qwen2.5-72B-Instruct"
    echo "  Qwen2.5-7B-Instruct"
    echo "  DeepSeek-R1-Distill-Llama-70B"
    echo "  Llama-3.3-70B-Instruct"
    echo "  Llama-3.2-11B-Vision-Instruct"
    echo "  Llama-3.2-3B-Instruct"
    echo "  Llama-3.2-1B-Instruct"
    echo "  Llama-3.1-70B-Instruct"
    echo "  Llama-3.1-70B"
    echo "  Llama-3.1-8B-Instruct"
    echo "  Llama-3.1-8B"
    echo "  Llama-3-70B-Instruct"
    echo "  Llama-3-70B"
    echo "  Llama-3-8B-Instruct"
    echo "  Llama-3-8B"
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

# Returns:
# 0 - Everything is OK
# 1 - The string doesn't match token's expected format
# 2 - The token is rejected by Hugging Face
# 3 - The token doesn't have access to the model
check_hf_access() {
    local input_hf_token="$1"

    # Check token format
    if [ -z "${input_hf_token:-}" ]; then
        echo "⛔ HF_TOKEN cannot be empty. Please try again."
        return 1
    elif [[ "$input_hf_token" != hf_* ]]; then
        echo "⛔ HF_TOKEN must start with 'hf_'. Please try again."
        return 1
    fi

    # Check if Hugging Face approves the token
    local whoami_url="https://huggingface.co/api/whoami-v2"
    response=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer ${input_hf_token}" "${whoami_url}")

    if [ "$response" -ne 200 ]; then
        echo "⛔ HF_TOKEN rejected by 🤗 Hugging Face. Please check your token and try again."
        return 2
    fi

    # To confirm if the token has access to the model we need to try to download a file
    local model_url="https://huggingface.co/api/models/${HF_MODEL_REPO_ID}"
    model_files=$(curl -s -H "Authorization: Bearer ${input_hf_token}" "${model_url}" | grep -o '"rfilename":"[^"]*"' | cut -d'"' -f4)
    if [ -z "$model_files" ]; then
        # this should never happen for the models supported by this script
        echo "⛔ No files found in the model repository. HF_MODEL_REPO_ID=${HF_MODEL_REPO_ID}. Does your HF_TOKEN have access?"
        exit 1
    fi

    # Check the header for the first file
    # If the token can't access the model the response will have this:
    #     x-error-code: GatedRepo
    first_file=$(echo "$model_files" | head -n 1)
    response_headers=$(curl -s -H "Authorization: Bearer ${input_hf_token}" -I "https://huggingface.co/${HF_MODEL_REPO_ID}/resolve/main/${first_file}")
    x_error_code=$(echo "$response_headers" | grep -i "^x-error-code" | awk -F': ' '{print $2}' | tr -d '\r' || echo "")
    if [ -n "$x_error_code" ]; then
        echo "⛔ The model is gated and you don't have access."
        return 3
    fi

    echo "✅ HF_TOKEN is valid and has access to the model."
    return 0
}

get_hf_env_vars() {
    # get HF_TOKEN
    if [ -z "${HF_TOKEN:-}" ]; then
        read -r -s -p "Enter your HF_TOKEN: " input_hf_token
        echo

        check_hf_access $input_hf_token
        if [ $? -ne 0 ]; then
            echo "⛔ Error occurred during HF_TOKEN validation. Please check the token and try again."
            exit 1
        fi

        HF_TOKEN=${input_hf_token}
        echo "✅ HF_TOKEN set."
    fi
    # get HF_HOME
    if [ -z "${HF_HOME:-}" ]; then
        echo "HF_HOME environment variable is not set. Please set it before running the script."
        read -e -r -p "Enter your HF_HOME [default: $HOME/.cache/huggingface]:" input_hf_home
        echo
        input_hf_home=${input_hf_home:-"$HOME/.cache/huggingface"}
        if [ ! -d "$input_hf_home" ]; then
            mkdir -p "$input_hf_home" 2>/dev/null || {
                echo "⛔ Failed to create HF_HOME directory. Please check permissions and try again."
                echo "Entered input was HF_HOME:= ${input_hf_home}, is this correct for your system?"
                exit 1
            }
        fi
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
    # note: MODEL_NAME is the directory name for the model weights
    case "$1" in
        "Qwen2.5-72B-Instruct")
        IMPL_ID="tt-metal"
        MODEL_NAME="Qwen2.5-72B-Instruct"
        HF_MODEL_REPO_ID="Qwen/Qwen2.5-72B-Instruct"
        META_MODEL_NAME=""
        META_DIR_FILTER=""
        REPACKED=0
        ;;
        "Qwen2.5-7B-Instruct")
        IMPL_ID="tt-metal"
        MODEL_NAME="Qwen2.5-7B-Instruct"
        HF_MODEL_REPO_ID="Qwen/Qwen2.5-7B-Instruct"
        META_MODEL_NAME=""
        META_DIR_FILTER=""
        REPACKED=0
        ;;
        "DeepSeek-R1-Distill-Llama-70B")
        IMPL_ID="tt-metal"
        MODEL_NAME="DeepSeek-R1-Distill-Llama-70B"
        HF_MODEL_REPO_ID="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        META_MODEL_NAME=""
        META_DIR_FILTER=""
        REPACKED=0
        ;;
        "Llama-3.3-70B-Instruct")
        IMPL_ID="tt-metal"
        MODEL_NAME="Llama-3.3-70B-Instruct"
        HF_MODEL_REPO_ID="meta-llama/Llama-3.3-70B-Instruct"
        META_MODEL_NAME=""
        META_DIR_FILTER=""
        REPACKED=1
        ;;
        "Llama-3.2-11B-Vision-Instruct")
        IMPL_ID="tt-metal"
        MODEL_NAME="Llama-3.2-11B-Vision-Instruct"
        HF_MODEL_REPO_ID="meta-llama/Llama-3.2-11B-Vision-Instruct"
        META_MODEL_NAME=""
        META_DIR_FILTER=""
        REPACKED=0
        ;;
        "Llama-3.2-3B-Instruct")
        IMPL_ID="tt-metal"
        MODEL_NAME="Llama-3.2-3B-Instruct"
        HF_MODEL_REPO_ID="meta-llama/Llama-3.2-3B-Instruct"
        META_MODEL_NAME=""
        META_DIR_FILTER=""
        REPACKED=0
        ;;
        "Llama-3.2-1B-Instruct")
        IMPL_ID="tt-metal"
        MODEL_NAME="Llama-3.2-1B-Instruct"
        HF_MODEL_REPO_ID="meta-llama/Llama-3.2-1B-Instruct"
        META_MODEL_NAME=""
        META_DIR_FILTER=""
        REPACKED=0
        ;;
        "Llama-3.1-70B-Instruct")
        IMPL_ID="tt-metal"
        MODEL_NAME="Llama-3.1-70B-Instruct"
        HF_MODEL_REPO_ID="meta-llama/Llama-3.1-70B-Instruct"
        META_MODEL_NAME="Meta-Llama-3.1-70B-Instruct"
        META_DIR_FILTER="llama3_1"
        REPACKED=1
        ;;
        "Llama-3.1-70B")
        IMPL_ID="tt-metal"
        MODEL_NAME="Llama-3.1-70B"
        HF_MODEL_REPO_ID="meta-llama/Llama-3.1-70B"
        META_MODEL_NAME="Meta-Llama-3.1-70B"
        META_DIR_FILTER="llama3_1"
        REPACKED=1
        ;;
        "Llama-3.1-8B-Instruct")
        IMPL_ID="tt-metal"
        MODEL_NAME="Llama-3.1-8B-Instruct"
        HF_MODEL_REPO_ID="meta-llama/Llama-3.1-8B-Instruct"
        META_MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
        META_DIR_FILTER="llama3_1"
        REPACKED=0
        ;;
        "Llama-3.1-8B")
        IMPL_ID="tt-metal"
        MODEL_NAME="Llama-3.1-8B"
        HF_MODEL_REPO_ID="meta-llama/Llama-3.1-8B"
        META_MODEL_NAME="Meta-Llama-3.1-8B"
        META_DIR_FILTER="llama3_1"
        REPACKED=0
        ;;
        "Llama-3-70B-Instruct")
        IMPL_ID="tt-metal"
        MODEL_NAME="Llama-3-70B-Instruct"
        HF_MODEL_REPO_ID="meta-llama/Llama-3-70B-Instruct"
        META_MODEL_NAME="Meta-Llama-3-70B-Instruct"
        META_DIR_FILTER="llama3"
        REPACKED=1
        ;;
        "Llama-3-70B")
        IMPL_ID="tt-metal"
        MODEL_NAME="Llama-3-70B"
        HF_MODEL_REPO_ID="meta-llama/Llama-3-70B"
        META_MODEL_NAME="Meta-Llama-3-70B"
        META_DIR_FILTER="llama3"
        REPACKED=1
        ;;
        "Llama-3-8B-Instruct")
        IMPL_ID="tt-metal"
        MODEL_NAME="Llama-3-8B-Instruct"
        HF_MODEL_REPO_ID="meta-llama/Llama-3-8B-Instruct"
        META_MODEL_NAME="Meta-Llama-3-8B-Instruct"
        META_DIR_FILTER="llama3"
        REPACKED=0
        ;;
        "Llama-3-8B")
        IMPL_ID="tt-metal"
        MODEL_NAME="Llama-3-8B"
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

    # Set default values for environment variables
    DEFAULT_PERSISTENT_VOLUME_ROOT=${REPO_ROOT}/persistent_volume
    # Safely handle potentially unset environment variables using default values
    PERSISTENT_VOLUME_ROOT=${PERSISTENT_VOLUME_ROOT:-$DEFAULT_PERSISTENT_VOLUME_ROOT}
    # Prompt user for PERSISTENT_VOLUME_ROOT if not already set or use default
    read -e -r -p "Enter your PERSISTENT_VOLUME_ROOT [default: ${DEFAULT_PERSISTENT_VOLUME_ROOT}]: " INPUT_PERSISTENT_VOLUME_ROOT
    PERSISTENT_VOLUME_ROOT=${INPUT_PERSISTENT_VOLUME_ROOT:-$PERSISTENT_VOLUME_ROOT}
    echo # move to a new line after input   
    # Set environment variables with defaults if not already set
    MODEL_VERSION="0.0.1"
    MODEL_ID="id_${IMPL_ID}-${MODEL_NAME}-v${MODEL_VERSION}"
    PERSISTENT_VOLUME="${PERSISTENT_VOLUME_ROOT}/volume_${MODEL_ID}"

    # Initialize OVERWRITE_ENV
    OVERWRITE_ENV=false
    MODEL_ENV_DIR="${PERSISTENT_VOLUME_ROOT}/model_envs"
    mkdir -p ${MODEL_ENV_DIR}
    ENV_FILE="${MODEL_ENV_DIR}/${MODEL_NAME}.env"
    export ENV_FILE
    check_and_prompt_env_file

    if [ "$OVERWRITE_ENV" = false ]; then
        echo "✅ using existing .env file: ${ENV_FILE}."
        return 0
    fi

    read -p $'How do you want to provide a model?\n1) Download from 🤗 Hugging Face (default)\n2) Download from Meta\n3) Local folder\nEnter your choice: ' input_model_source
    choice_model_source=${input_model_source:-"1"}
    echo # move to a new line after input
    # Handle user's choice
    case "$choice_model_source" in
        1 )
            echo "Using 🤗 Hugging Face Token."
            get_hf_env_vars
            ;;
        2 )
            if [ -z "${META_DIR_FILTER:-}" ]; then
                echo "⛔ MODEL_NAME=${MODEL_NAME} does not support using direct Meta authorization model download. Please use Hugging Face method."
            fi
            echo "Using direct authorization from Meta. You will need their URL Authorization token, typically from their website or email."
            # Prompt user for LLAMA_REPO if not already set or use default
            read -e -r -p "Enter the path where you want to clone the Llama model repository [default: ${LLAMA_REPO}]: " INPUT_LLAMA_REPO
            LLAMA_REPO=${INPUT_LLAMA_REPO:-$LLAMA_REPO}
            LLAMA_MODELS_DIR=${LLAMA_MODELS_DIR:-${LLAMA_REPO}/models/${META_DIR_FILTER}}
            LLAMA_WEIGHTS_DIR=${LLAMA_WEIGHTS_DIR:-${LLAMA_MODELS_DIR}/${META_MODEL_NAME}}
            echo  # move to a new line after input
            ;;
        3 )
            if [ -n "${LLAMA_DIR:-}" ]; then
                # If LLAMA_DIR environment variable is set, it usually points to the folder with weights
                LLAMA_WEIGHTS_DIR=${LLAMA_DIR}
            else
                # Else prompt user for the path
                read -e -r -p "Provide the path to the Llama model directory: " input_llama_dir
                LLAMA_WEIGHTS_DIR=${input_llama_dir}
            fi
            echo "Using local folder: ${LLAMA_WEIGHTS_DIR}"
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

    CONTAINER_APP_USERNAME="container_app_user"
    CONTAINER_HOME="/home/${CONTAINER_APP_USERNAME}"
    CACHE_ROOT="${CONTAINER_HOME}/cache_root"
    MODEL_WEIGHTS_PATH="${CACHE_ROOT}/model_weights/${REPACKED_STR}$MODEL_NAME"
    # Write environment variables to .env file
    echo "Writing environment variables to ${ENV_FILE} ..."
    cat > ${ENV_FILE} <<EOF
# Environment variables for the model setup
MODEL_SOURCE=$choice_model_source
HF_MODEL_REPO_ID=$HF_MODEL_REPO_ID
MODEL_NAME=$MODEL_NAME
MODEL_VERSION=${MODEL_VERSION}
IMPL_ID=${IMPL_ID}
MODEL_ID=${MODEL_ID}
META_MODEL_NAME=$META_MODEL_NAME
REPACKED=${REPACKED}
REPACKED_STR=${REPACKED_STR}
# model runtime variables
SERVICE_PORT=7000
# host paths
HOST_HF_HOME=${HF_HOME:-""}
LLAMA_REPO=${LLAMA_REPO:-""}
LLAMA_MODELS_DIR=${LLAMA_MODELS_DIR:-""}
LLAMA_WEIGHTS_DIR=${LLAMA_WEIGHTS_DIR:-""}
PERSISTENT_VOLUME_ROOT=$PERSISTENT_VOLUME_ROOT
PERSISTENT_VOLUME=$PERSISTENT_VOLUME
# container paths
CACHE_ROOT=${CACHE_ROOT}
HF_HOME=${CACHE_ROOT}/huggingface
MODEL_WEIGHTS_ID=id_${REPACKED_STR}$MODEL_NAME
MODEL_WEIGHTS_PATH=${MODEL_WEIGHTS_PATH}
LLAMA_DIR=${MODEL_WEIGHTS_PATH}
LLAMA3_CKPT_DIR=${MODEL_WEIGHTS_PATH}
LLAMA3_TOKENIZER_PATH=${MODEL_WEIGHTS_PATH}/tokenizer.model
LLAMA3_CACHE_PATH=${CACHE_ROOT}/tt_metal_cache/cache_${REPACKED_STR}$MODEL_NAME
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
        echo "Running the download script to download models at ${LLAMA_DIR}/download.sh ..."
        cd "$LLAMA_DIR"
        ./download.sh
        cd -
    fi

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

    if [ "${HF_MODEL_REPO_ID}" = "Qwen/Qwen2.5-72B-Instruct" ] || [ "${HF_MODEL_REPO_ID}" = "Qwen/Qwen2.5-7B-Instruct" ]; then
        # download full repo
        HF_REPO_PATH_FILTER="*"
        huggingface-cli download "${HF_MODEL_REPO_ID}" 
    elif [ "${HF_MODEL_REPO_ID}" = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" ]; then
        # download full repo
        HF_REPO_PATH_FILTER="*"
        huggingface-cli download "${HF_MODEL_REPO_ID}" 
    else
        HF_REPO_PATH_FILTER="original/*"
        # using default Llama original convention for model weights
        huggingface-cli download "${HF_MODEL_REPO_ID}" \
            original/params.json \
            original/tokenizer.model \
            original/consolidated.* \
            --cache-dir="${HOST_HF_HOME}" \
            --token="${HF_TOKEN}"
    fi

    if [ $? -ne 0 ]; then
        echo "⛔ Error occured during: huggingface-cli download ${HF_MODEL_REPO_ID}"
        echo "🔔 check for common issues:"
        echo "  1. 401 Unauthorized error occurred."
        echo "    For example:"
        echo "      huggingface_hub.errors.GatedRepoError: 401 Client Error. Cannot access gated repo"
        echo "      ❗ In this case, go to the repo URL in your web browser and click through the access request form."
        echo "  2. check correct HF_TOKEN is set in the .env file: ${ENV_FILE}"
        exit 1
    fi

    # symlinks are broken for huggingface-cli download with --local-dir option
    # see: https://github.com/huggingface/huggingface_hub/pull/2223
    # to use symlinks, find most recent snapshot and create symlink to that
    WEIGHTS_DIR=${PERSISTENT_VOLUME}/model_weights/${MODEL_NAME}
    mkdir -p "${WEIGHTS_DIR}"
    LOCAL_REPO_NAME=$(echo "${HF_MODEL_REPO_ID}" | sed 's|/|--|g')
    SNAPSHOT_DIR="${HOST_HF_HOME}/models--${LOCAL_REPO_NAME}/snapshots"
    if [ ! -d "$SNAPSHOT_DIR" ]; then
        echo "Primary snapshot directory not found at: $SNAPSHOT_DIR"
        # Try alternative path
        SNAPSHOT_DIR="${HOST_HF_HOME}/hub/models--${LOCAL_REPO_NAME}/snapshots"
        if [ ! -d "$SNAPSHOT_DIR" ]; then
            echo "⛔ Error: Alternative snapshot directory not found either at ${SNAPSHOT_DIR}."
            exit 1
        fi
        echo "Using alternative snapshot directory: $SNAPSHOT_DIR"
    fi
    # note: ls -td will sort by modification date descending, potential edge case
    # if desired snapshot is not most recent modified or ls sorts differently
    MOST_RECENT_SNAPSHOT=$(ls -td -- ${SNAPSHOT_DIR}/* | head -n 1)
    # Note: do not quote the pattern or globbing wont work
    for item in ${MOST_RECENT_SNAPSHOT}/${HF_REPO_PATH_FILTER}; do
        if [ "${REPACKED}" -eq 1 ]; then
            echo "create symlink to: ${item} in ${WEIGHTS_DIR}"
            ln -s "$item" "${WEIGHTS_DIR}"
        else
            # if not repacking, need to make weights accessible in container
            echo "copying ${item} to ${WEIGHTS_DIR} ..."
            mkdir -p "${WEIGHTS_DIR}"
            if [ -L "$item" ]; then
                # Get the linked blob and copy it to the destination with the name of the link
                cp -L "$item" "${WEIGHTS_DIR}/$(basename "$item")"
            else
                cp -rf "$item" "${WEIGHTS_DIR}"
            fi
        fi
    done

    if [ "${HF_MODEL_REPO_ID}" = "meta-llama/Llama-3.2-11B-Vision-Instruct" ]; then
        # tt-metal impl expects models with naming: consolidated.xx.pth
        # this convention is followed in all models expect Llama-3.2-11B-Vision-Instruct
        mv "${WEIGHTS_DIR}/consolidated.pth" "${WEIGHTS_DIR}/consolidated.00.pth"  
    fi

    # Step 6: Cleanup HF setup venv
    echo "Cleanup HF setup venv ..."
    deactivate
    rm -rf ${VENV_NAME}
    
    # Step 7: Process and copy weights
    if [ "${REPACKED}" -eq 1 ]; then
        echo "repacking weights ..."
        REPACKED_WEIGHTS_DIR="${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME}"
        mkdir -p "${REPACKED_WEIGHTS_DIR}"
        repack_weights "${WEIGHTS_DIR}" "${REPACKED_WEIGHTS_DIR}"
    fi

    echo "using weights directory: ${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME}"
    echo "✅ setup_weights_huggingface completed!"
}

setup_weights_local() {
    WEIGHTS_DIR=${PERSISTENT_VOLUME}/model_weights/${MODEL_NAME}
    echo "copy weights: ${LLAMA_WEIGHTS_DIR} -> ${WEIGHTS_DIR}"
    mkdir -p "${WEIGHTS_DIR}"
    for item in ${LLAMA_WEIGHTS_DIR}/*; do
        if [ -L "$item" ]; then
            # Get the linked file and copy it to the destination with the name of the link
            target=$(readlink "$item")
            cp -L "$item" "${WEIGHTS_DIR}/$(basename "$item")"
        else
            cp -rf "$item" "${WEIGHTS_DIR}"
        fi
    done
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
    load_env
    # do cache setup first incase there is network issue downloading weights.
    # TODO: deprecate setup_tt_metal_cache, new model impl uses model_weights dir for tt-metal cache
    setup_tt_metal_cache
    # check if model weights already exist
    if [ -d "${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME}" ]; then
        echo "Model weights already exist at: ${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME}"
        echo "🔔 check if directory contents are correct."
        echo "contents:"
        echo "ls -lh ${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME}"
        echo "$(ls -lh ${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME})"
        echo
        echo "If directory does not have correct weights, to re-download or copy the model weights delete the directory."
    else
        echo "Setting up persistent volume root: ${PERSISTENT_VOLUME}"
        mkdir -p "${PERSISTENT_VOLUME}/model_weights/"
        case "$MODEL_SOURCE" in
            1 )
                setup_weights_huggingface
                ;;
            2 )
                setup_weights_meta
                ;;
            3 )
                setup_weights_local
                ;;
            * )
                echo "⛔ Invalid model source. Exiting."
                exit 1
                ;;
        esac
    fi
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
