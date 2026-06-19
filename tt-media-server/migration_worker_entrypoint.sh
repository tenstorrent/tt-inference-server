#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Entrypoint for the migration_worker image.
#
# The disaggregated migration layer is orchestrated over ssh:
# migration_verify_launcher.sh runs on a "name-server host" and ssh's into
# every prefill/decode host to spawn migration_endpoint, migration_worker,
# prte and prun (the PRRTE DVM). When those "hosts" are containers, each
# container needs an inbound sshd listener and a trusted authorized_keys
# entry so the launcher can drive it.
#
# This entrypoint:
#   1. Generates ssh host keys on first boot if /etc/ssh/ssh_host_*_key is
#      missing (or per-deploy host keys can be mounted at /run/migration/host_keys/).
#   2. Installs authorized_keys from one of (in priority order):
#        - $MIGRATION_AUTHORIZED_KEYS         (raw env value, multi-line OK)
#        - $MIGRATION_AUTHORIZED_KEYS_FILE    (path to a file inside the container)
#        - /run/secrets/migration_authorized_keys   (Docker/Compose secret)
#      Multiple sources are concatenated; duplicates are de-duplicated.
#   3. (Optional) installs the orchestrator's private key from
#      $MIGRATION_SSH_PRIVATE_KEY (env value) or /run/secrets/migration_ssh_private_key
#      so the SAME image can run AS the launcher host (ssh outbound).
#   4. Starts sshd on $MIGRATION_SSHD_PORT (default 22) when
#      MIGRATION_START_SSHD=1 (the default for cluster use).
#   5. Hands off to whatever command was passed (`docker run image <cmd>`),
#      or to bash if none was given. PID 1 is the user command (sshd runs in
#      the background and is reaped via setsid + a SIGTERM trap).
#
# The image disables StrictHostKeyChecking/known_hosts cluster-wide via
# /etc/ssh/ssh_config so the (frequently re-spun) container set is reachable
# without an out-of-band host-key dance. This matches what tt-shield's
# workflow_exabox-deploy.yml does on bare-metal hosts via PRTE_MCA_plm_ssh_args.

set -euo pipefail

log() { echo "[migration_worker] $*" >&2; }

MIGRATION_START_SSHD="${MIGRATION_START_SSHD:-1}"
MIGRATION_SSHD_PORT="${MIGRATION_SSHD_PORT:-22}"
MIGRATION_USER_HOME="${HOME:-/root}"

ensure_dir() {
    local dir="$1" mode="$2"
    mkdir -p "${dir}"
    chmod "${mode}" "${dir}"
}

# --- 1) Host keys --------------------------------------------------------
# Allow operators to mount a stable host-key set via volumes at
# /run/migration/host_keys; otherwise generate ephemeral keys per boot.
if [[ -d /run/migration/host_keys ]] && \
   compgen -G '/run/migration/host_keys/ssh_host_*_key' >/dev/null; then
    log "Installing host keys from /run/migration/host_keys"
    cp /run/migration/host_keys/ssh_host_*_key      /etc/ssh/ 2>/dev/null || true
    cp /run/migration/host_keys/ssh_host_*_key.pub  /etc/ssh/ 2>/dev/null || true
    chmod 600 /etc/ssh/ssh_host_*_key 2>/dev/null || true
    chmod 644 /etc/ssh/ssh_host_*_key.pub 2>/dev/null || true
fi
if ! compgen -G '/etc/ssh/ssh_host_*_key' >/dev/null; then
    log "Generating ephemeral ssh host keys (ssh-keygen -A)"
    ssh-keygen -A
fi

# --- 2) authorized_keys --------------------------------------------------
ensure_dir "${MIGRATION_USER_HOME}/.ssh" 700
AUTH_KEYS="${MIGRATION_USER_HOME}/.ssh/authorized_keys"
: > "${AUTH_KEYS}"

append_keys_from() {
    local src="$1" label="$2"
    [[ -n "${src}" ]] || return 0
    [[ -f "${src}" ]] || { log "skip ${label}: file ${src} not found"; return 0; }
    log "appending authorized_keys from ${label} (${src})"
    cat "${src}" >> "${AUTH_KEYS}"
}

if [[ -n "${MIGRATION_AUTHORIZED_KEYS:-}" ]]; then
    log "appending authorized_keys from \$MIGRATION_AUTHORIZED_KEYS"
    printf '%s\n' "${MIGRATION_AUTHORIZED_KEYS}" >> "${AUTH_KEYS}"
fi
append_keys_from "${MIGRATION_AUTHORIZED_KEYS_FILE:-}"             "MIGRATION_AUTHORIZED_KEYS_FILE"
append_keys_from "/run/secrets/migration_authorized_keys"          "docker secret"

# Strip blank lines + comments; de-duplicate while preserving order.
if [[ -s "${AUTH_KEYS}" ]]; then
    awk 'NF && $1 !~ /^#/ && !seen[$0]++' "${AUTH_KEYS}" > "${AUTH_KEYS}.tmp"
    mv "${AUTH_KEYS}.tmp" "${AUTH_KEYS}"
    chmod 600 "${AUTH_KEYS}"
    log "authorized_keys installed: $(wc -l < "${AUTH_KEYS}") key(s)"
else
    log "WARN: no authorized_keys configured — inbound ssh will reject every connection."
    log "      set MIGRATION_AUTHORIZED_KEYS / MIGRATION_AUTHORIZED_KEYS_FILE, or"
    log "      mount /run/secrets/migration_authorized_keys."
fi

# --- 3) (Optional) outbound private key ----------------------------------
PRIV_KEY="${MIGRATION_USER_HOME}/.ssh/id_ed25519"
if [[ -n "${MIGRATION_SSH_PRIVATE_KEY:-}" ]]; then
    log "installing private key from \$MIGRATION_SSH_PRIVATE_KEY"
    printf '%s\n' "${MIGRATION_SSH_PRIVATE_KEY}" > "${PRIV_KEY}"
    chmod 600 "${PRIV_KEY}"
elif [[ -f /run/secrets/migration_ssh_private_key ]]; then
    log "installing private key from /run/secrets/migration_ssh_private_key"
    cp /run/secrets/migration_ssh_private_key "${PRIV_KEY}"
    chmod 600 "${PRIV_KEY}"
fi

# --- 4) sshd -------------------------------------------------------------
if [[ "${MIGRATION_START_SSHD}" == "1" ]]; then
    ensure_dir /run/sshd 755
    log "starting sshd on port ${MIGRATION_SSHD_PORT}"
    /usr/sbin/sshd -p "${MIGRATION_SSHD_PORT}" -E /var/log/sshd.log
    # Reap sshd cleanly on container shutdown.
    trap 'pkill -TERM sshd 2>/dev/null || true' EXIT INT TERM
fi

# --- 5) Hand off to user command -----------------------------------------
if [[ $# -eq 0 ]]; then
    log "no command supplied; defaulting to /bin/bash"
    set -- /bin/bash
fi

log "exec: $*"
exec "$@"
