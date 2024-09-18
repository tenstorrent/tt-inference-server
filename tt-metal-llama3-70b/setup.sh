#!/bin/bash

set -euo pipefail  # Exit on error, print commands, unset variables treated as errors, and exit on pipeline failure

# Function to display usage information
usage() {
    echo "Usage: $0 <model_type>"
    echo "Available model types:"
    echo "  llama-3.1-70b-instruct"
    echo "  llama-3.1-70b"
    echo "  llama-3.1-8b-instruct"
    echo "  llama-3.1-8b"
    echo "  llama-3-70b-instruct"
    echo "  llama-3-70b"
    echo "  llama-3-8b-instruct"
    echo "  llama-3-8b"
    echo
    echo "Options:"
    echo "  setup_permissions      Run the script to set file permissions after first run setup (requires sudo)."
    exit 1
}

check_and_prompt_env_file() {
    local ENV_FILE=".env"
    local MODEL_NAME_KEY="MODEL_NAME"
    local MODEL_NAME=""
    
    # Check if .env file exists
    if [[ -f "$ENV_FILE" ]]; then
        # Extract the MODEL_NAME value from .env
        FOUND_MODEL_NAME=$(grep "^$MODEL_NAME_KEY=" "$ENV_FILE" | cut -d '=' -f2)

        # If MODEL_NAME is found, display it
        if [[ -n "$FOUND_MODEL_NAME" ]]; then
            echo "The existing .env file contains MODEL_NAME: $FOUND_MODEL_NAME"
            # Prompt the user to overwrite or exit
            local choice=""
            read -p "Do you want to overwrite the existing .env file? (y/n) [default: y]:" choice
            choice=${choice:-y}
            # Handle user's choice
            case "$choice" in
                y|Y )
                    echo "Overwriting the .env file ..."
                    # Logic to overwrite .env goes here
                    OVERWRITE_ENV=true
                    ;;
                n|N )
                    echo "✅ using existing .env file."
                    OVERWRITE_ENV=false
                    ;;
                * )
                    echo "⛔ Invalid option. Exiting."
                    exit 1
                    ;;
            esac
        else
            echo "MODEL_NAME not found in .env file. Overwritting."
            OVERWRITE_ENV=true
        fi
        
    else
        echo ".env file does not exist. Proceeding to create a new one."
        OVERWRITE_ENV=true
    fi
}


# Function to set environment variables based on the model selection and write them to .env
setup_model_environment() {
    # Set default values for environment variables
    REPO_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
    DEFAULT_PERSISTENT_VOLUME_ROOT=${REPO_ROOT}/persistent_volume
    DEFAULT_LLAMA3_1_REPO=~/llama-models
    # Set environment variables based on the model selection
    case "$1" in
      "llama-3.1-70b-instruct")
      MODEL_NAME="llama-3.1-70b-instruct"
      META_MODEL_NAME="Meta-Llama-3.1-70B-Instruct"
      META_DIR_FILTER="llama3_1"
      REPACKED=1
      ;;
      "llama-3.1-70b")
      MODEL_NAME="llama-3.1-70b"
      META_MODEL_NAME="Meta-Llama-3.1-70B"
      META_DIR_FILTER="llama3_1"
      REPACKED=1
      ;;
      "llama-3.1-8b-instruct")
      MODEL_NAME="llama-3.1-8b-instruct"
      META_MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
      META_DIR_FILTER="llama3_1"
      REPACKED=0
      ;;
      "llama-3.1-8b")
      MODEL_NAME="llama-3.1-8b"
      META_MODEL_NAME="Meta-Llama-3.1-8B"
      META_DIR_FILTER="llama3_1"
      REPACKED=0
      ;;
      "llama-3-70b-instruct")
      MODEL_NAME="llama-3-70b-instruct"
      META_MODEL_NAME="Meta-Llama-3-70B-Instruct"
      META_DIR_FILTER="llama3"
      REPACKED=1
      ;;
      "llama-3-70b")
      MODEL_NAME="llama-3-70b"
      META_MODEL_NAME="Meta-Llama-3-70B"
      META_DIR_FILTER="llama3"
      REPACKED=1
      ;;
      "llama-3-8b-instruct")
      MODEL_NAME="llama-3-8b-instruct"
      META_MODEL_NAME="Meta-Llama-3-8B-Instruct"
      META_DIR_FILTER="llama3"
      REPACKED=0
      ;;
      "llama-3-8b")
      MODEL_NAME="llama-3-8b"
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

    check_and_prompt_env_file

    if [ "$OVERWRITE_ENV" = false ]; then
        echo "✅ using existing .env file."
        return 0
    fi

    # Safely handle potentially unset environment variables using default values
    PERSISTENT_VOLUME_ROOT=${PERSISTENT_VOLUME_ROOT:-$DEFAULT_PERSISTENT_VOLUME_ROOT}
    LLAMA3_1_REPO=${LLAMA3_1_REPO:-$DEFAULT_LLAMA3_1_REPO}

    # Prompt user for PERSISTENT_VOLUME_ROOT if not already set or use default
    read -p "Enter your PERSISTENT_VOLUME_ROOT [default: ${PERSISTENT_VOLUME_ROOT}]: " INPUT_PERSISTENT_VOLUME_ROOT
    PERSISTENT_VOLUME_ROOT=${INPUT_PERSISTENT_VOLUME_ROOT:-$PERSISTENT_VOLUME_ROOT}
    echo
    # Prompt user for LLAMA3_1_REPO if not already set or use default
    read -p "Enter the path where you want to clone the Llama model repository [default: ${LLAMA3_1_REPO}]: " INPUT_LLAMA3_1_REPO
    LLAMA3_1_REPO=${INPUT_LLAMA3_1_REPO:-$LLAMA3_1_REPO}
    echo  # move to a new line after input

    # Set environment variables with defaults if not already set
    LLAMA3_1_DIR=${LLAMA3_1_DIR:-${LLAMA3_1_REPO}/models/${META_DIR_FILTER}}
    LLAMA3_1_WEIGHTS_DIR=${LLAMA3_1_WEIGHTS_DIR:-${LLAMA3_1_DIR}/${META_MODEL_NAME}}
    PERSISTENT_VOLUME=${PERSISTENT_VOLUME:-${PERSISTENT_VOLUME_ROOT}/volume_id_tt-metal-${MODEL_NAME}v0.0.1}

    # Prompt user for JWT_SECRET securely
    read -sp "Enter your JWT_SECRET: " JWT_SECRET
    echo  # move to a new line after input
    # Verify the JWT_SECRET is not empty
    if [ -z "$JWT_SECRET" ]; then
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
    echo "Writing environment variables to .env file..."
    cat > .env <<EOF
MODEL_NAME=$MODEL_NAME
META_MODEL_NAME=$META_MODEL_NAME
# host paths
LLAMA3_1_REPO=$LLAMA3_1_REPO
LLAMA3_1_DIR=$LLAMA3_1_DIR
LLAMA3_1_WEIGHTS_DIR=$LLAMA3_1_WEIGHTS_DIR
PERSISTENT_VOLUME_ROOT=$PERSISTENT_VOLUME_ROOT
PERSISTENT_VOLUME=$PERSISTENT_VOLUME
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
EOF

    echo "Environment variables written to .env file."
    echo "✅ setup_model_environment completed!"
}

# Function to load environment variables from .env file
load_env() {
    if [ -f .env ]; then
        echo "Sourcing environment variables from .env file..."
        source .env
    else
        echo "⛔ .env file not found. Please run the setup first."
        exit 1
    fi
}

# SUDO PORTION: Encapsulated in a function to handle all sudo-requiring tasks
setup_permissions() {
    echo "Running sudo-required commands..."
    # Check if the script is being run as root
    if [ "$EUID" -ne 0 ]; then
        echo "⛔ Please run as root or use: sudo $0 setup_permissions"
        exit 1
    fi

    # Load environment variables from .env
    load_env

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
    CONTAINER_USER=$(getent passwd 1000 | cut -d: -f1)
    if [ -n "$CONTAINER_USER" ]; then
        echo "Adding container user: '$CONTAINER_USER' (UID 1000) to 'dockermount' group ..."
        sudo usermod -aG dockermount "$CONTAINER_USER"
    else
        echo "No user found with UID 1000."
    fi

    # Set file ownership and permissions
    echo "Setting file ownership and permissions for container and host access ..."
    sudo chown -R ${CONTAINER_USER}:dockermount "${PERSISTENT_VOLUME}"
    sudo chmod -R 775 "${PERSISTENT_VOLUME}"

    echo "✅ setup_permissions completed!"
}

setup_weights() {
    # St`ep 1: Load environment variables from .env file
    load_env

    # check if model weights already exist
    if [ -d "${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME}" ]; then
        echo "Model weights already exist at: ${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME}."
        echo "contents:"
        echo
        echo "$(ls -lh ${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME})"
        echo
        echo "⛔ Skipping model weights setup. Delete the directory to re-download or copy the model weights."
        exit 1
    fi

    # TODO: support HF_TOKEN for downloading models
    # Step 2: Set up Llama model repository path
    echo "Using repository path: $LLAMA3_1_REPO"

    # Step 3: Clone the repository (if it doesn't already exist)
    if [ ! -d "$LLAMA3_1_REPO" ]; then
        echo "Cloning the Llama repository to: $LLAMA3_1_REPO"
        git clone https://github.com/meta-llama/llama-models.git "$LLAMA3_1_REPO"
    else
        echo "🔔 Llama repository already exists at $LLAMA3_1_REPO"
    fi

    # Step 4: Check if weights are already downloaded
    if [ -d "${LLAMA3_1_WEIGHTS_DIR}" ] && [ "$(ls -A "${LLAMA3_1_WEIGHTS_DIR}")" ]; then
        echo "Weights already downloaded at ${LLAMA3_1_WEIGHTS_DIR}."
        echo "Skipping download."
    else
        # Step 5: Run the download script and select models
        echo "Running the download script to download models at ${LLAMA3_1_DIR}/download.sh ..."
        cd "$LLAMA3_1_DIR"
        ./download.sh
        cd -
    fi

    # Step 6: Set up persistent volume root
    echo "Setting up persistent volume root."
    mkdir -p "${PERSISTENT_VOLUME}/model_weights/"

    # Step 7: Create directories for weights, tokenizer, and params

    # Step 8: Copy weights, tokenizer, and params
    echo "Copying model weights, tokenizer, and params to the persistent volume."
    cp -r "${LLAMA3_1_WEIGHTS_DIR}" "${PERSISTENT_VOLUME}/model_weights/${MODEL_NAME}"
    mkdir -p "${PERSISTENT_VOLUME}/tt_metal_cache/cache_${REPACKED_STR}${MODEL_NAME}"
    if [ "${REPACKED}" -eq 1 ]; then
        mkdir -p "${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME}"
        cp "${LLAMA3_1_WEIGHTS_DIR}/tokenizer.model" "${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME}/tokenizer.model"
        cp "${LLAMA3_1_WEIGHTS_DIR}/params.json" "${PERSISTENT_VOLUME}/model_weights/${REPACKED_STR}${MODEL_NAME}/params.json"
    fi
    echo "✅ setup_weights completed!"
}

# Ensure script is being executed, not sourced
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "⛔ Error: This script is being sourced. Please make execute it:"
    echo "chmod +x ./setup.sh && ./setup.sh"
    set +euo pipefail  # Unset 'set -euo pipefail' when sourcing so it doesnt exit or mess up sourcing shell
    return 1;  # 'return' works when sourced; 'exit' would terminate the shell
fi

# Main script logic
if [ $# -lt 1 ]; then
    usage
fi

if [ "$1" == "setup_permissions" ]; then
    setup_permissions
    exit 0
fi

# Set up environment variables for the chosen model
MODEL_TYPE=$1
setup_model_environment "$MODEL_TYPE"
setup_weights
# Call the script again with sudo to execute the sudo-required commands
echo "Switching to sudo portion to set file permissions and complete setup."
sudo "$0" setup_permissions
