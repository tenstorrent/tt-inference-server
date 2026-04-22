#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# Unified entrypoint for multi-host distributed inference.
# Handles both Worker (sshd) and Controller (vLLM) roles.
#
# Usage:
#   Worker:     docker run -e MULTIHOST_ROLE=worker ...
#   Controller: docker run -e MULTIHOST_ROLE=controller ... <command>

set -e

CONTAINER_USER="container_app_user"
USER_HOME="/home/${CONTAINER_USER}"
SSH_DIR="${USER_HOME}/.ssh"

# ========================================
# SSH setup (both roles)
# ========================================
setup_ssh() {
    local role="$1"
    mkdir -p "${SSH_DIR}"

    if [[ "$role" == "worker" ]]; then
        # Worker: copy public key to authorized_keys
        local pubkey_src="${PUBKEY_SRC:-/tmp/authorized_keys.pub}"
        if [[ ! -f "${pubkey_src}" ]]; then
            echo "[multihost_entrypoint] ERROR: Public key not found at ${pubkey_src}"
            exit 1
        fi
        cp "${pubkey_src}" "${SSH_DIR}/authorized_keys"
        chmod 600 "${SSH_DIR}/authorized_keys"
        echo "[multihost_entrypoint] Configured authorized_keys from ${pubkey_src}"

    elif [[ "$role" == "controller" ]]; then
        # Controller: copy full SSH config directory
        local config_src="${SSH_CONFIG_SRC:-/tmp/ssh_config}"
        if [[ ! -d "${config_src}" ]]; then
            echo "[multihost_entrypoint] ERROR: SSH config directory not found at ${config_src}"
            exit 1
        fi
        if [[ -z "$(ls -A "${config_src}" 2>/dev/null)" ]]; then
            echo "[multihost_entrypoint] ERROR: SSH config directory is empty at ${config_src}"
            exit 1
        fi
        cp -r "${config_src}"/* "${SSH_DIR}/"
        # Set file permissions
        [[ -f "${SSH_DIR}/id_ed25519_multihost" ]] && chmod 600 "${SSH_DIR}/id_ed25519_multihost"
        [[ -f "${SSH_DIR}/config" ]] && chmod 644 "${SSH_DIR}/config"
        [[ -f "${SSH_DIR}/known_hosts" ]] && chmod 644 "${SSH_DIR}/known_hosts"
        echo "[multihost_entrypoint] Configured SSH from ${config_src}"
    fi

    # Fix ownership and directory permissions
    chown -R "${CONTAINER_USER}:${CONTAINER_USER}" "${SSH_DIR}"
    chmod 700 "${SSH_DIR}"
}

# ========================================
# Main
# ========================================
ROLE="${MULTIHOST_ROLE:-worker}"
SSH_PORT="${SSH_PORT:-2200}"

echo "[multihost_entrypoint] Role: ${ROLE}"

setup_ssh "${ROLE}"

if [[ "$ROLE" == "worker" ]]; then
    echo "[multihost_entrypoint] Starting sshd on port ${SSH_PORT}..."
    exec /usr/sbin/sshd -D -p "${SSH_PORT}" -e

elif [[ "$ROLE" == "controller" ]]; then
    if [[ $# -eq 0 ]]; then
        echo "[multihost_entrypoint] ERROR: No command provided for controller"
        exit 1
    fi
    echo "[multihost_entrypoint] Executing command as ${CONTAINER_USER}: $*"
    # Activate venv and execute command with gosu for privilege drop
    PYTHON_ENV_DIR="${PYTHON_ENV_DIR:-/opt/venv}"
    exec gosu "${CONTAINER_USER}" /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate && exec \"\$@\"" -- "$@"

else
    echo "[multihost_entrypoint] ERROR: Unknown role: ${ROLE}"
    echo "[multihost_entrypoint] Valid roles: worker, controller"
    exit 1
fi
