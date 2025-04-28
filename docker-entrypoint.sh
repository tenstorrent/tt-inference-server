#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# Docker entry point script:
# ensures CONTAINER_APP_USERNAME has read/write permissions to:
# - CACHE_ROOT 
# 
# This script is run by container root user at startup, CMD is then deescalated 
# to non-root user CONTAINER_APP_USERNAME.
# Note: for additional run-time mounted volumes, mount them as leaf to
# /home/${CONTAINER_APP_USERNAME}/ if read/write permissions are needed.

set -eo pipefail

set_group_permissions() {
    local var_dir="$1"
    local shared_group="$2"
    echo "setting permissions for ${var_dir} ..."
    
    # Skip if directory doesn't exist
    if [ ! -d "$var_dir" ]; then
        return 0
    fi

    # Check current group and permissions
    current_group=$(stat -c "%G" "$var_dir")
    current_perms=$(stat -c "%a" "$var_dir")
    
    # Set group if needed
    if [ "$current_group" != "$shared_group" ]; then
        chown -R :"$shared_group" "$var_dir"
    fi
    
    # Set permissions if needed
    if [ "$current_perms" != "2775" ]; then
        chmod -R 2775 "$var_dir"
    fi
}

echo "using CACHE_ROOT: ${CACHE_ROOT}"

# Get current ownership of volume
VOLUME_OWNER=$(stat -c '%u' "$CACHE_ROOT")
VOLUME_GROUP=$(stat -c '%g' "$CACHE_ROOT")
echo "Mounted CACHE_ROOT volume is owned by UID:GID - $VOLUME_OWNER:$VOLUME_GROUP"

# Create shared group with host's GID if it doesn't exist
if ! getent group "$VOLUME_GROUP" > /dev/null 2>&1; then
    groupadd -g "$VOLUME_GROUP" sharedgroup
fi

# Get the created/existing group name
SHARED_GROUP_NAME=$(getent group "$VOLUME_GROUP" | cut -d: -f1)

# Add container user to the shared group
usermod -a -G "$SHARED_GROUP_NAME" "${CONTAINER_APP_USERNAME}"

# Ensure new files get group write permissions (in current shell)
umask 0002

# only set permisssions for cache_root
set_group_permissions "$CACHE_ROOT" "$SHARED_GROUP_NAME"
# NOTE: running recursive chmod on /home/${CONTAINER_APP_USERNAME} takes long time
echo "Mounted volume permissions setup completed."

# Execute CMD as CONTAINER_APP_USERNAME user
exec gosu "${CONTAINER_APP_USERNAME}" "$@"
