# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Shared permission checking utilities for container bind mounts.

This module provides POSIX permission checking functions used by both
single-host (validate_setup.py) and multi-host (validate_multihost.py)
validation modules.
"""

import grp
import stat
from pathlib import Path


def get_groups_for_uid(uid: int) -> set[int]:
    """Return the set of GIDs that a given UID belongs to on this host.

    Args:
        uid: Numeric UID to lookup

    Returns:
        Set of GIDs the UID belongs to, including primary group and
        supplementary groups. Returns empty set if UID doesn't exist
        on the host.
    """
    gids = set()
    try:
        import pwd

        pw = pwd.getpwuid(uid)
        gids.add(pw.pw_gid)
        username = pw.pw_name
        for group in grp.getgrall():
            if username in group.gr_mem:
                gids.add(group.gr_gid)
    except KeyError:
        # UID doesn't exist on the host; can only rely on "other" bits
        pass
    return gids


def check_path_permissions_for_uid(
    path, uid: int, need_write: bool = False
) -> tuple[bool, str]:
    """Check whether the given UID can access a path based on POSIX permission bits.

    Best-effort pre-flight check. Cannot detect ACLs, SELinux, or other
    security modules, but catches common UID/permission mismatches.

    Args:
        path: Filesystem path to check.
        uid: Numeric UID that will access the path (i.e. --image-user).
        need_write: If True, also check write permission.

    Returns:
        Tuple of (ok: bool, reason: str). reason is empty when ok is True.
    """
    path = Path(path)
    if not path.exists():
        return False, f"path does not exist: {path}"

    st = path.stat()
    mode = st.st_mode
    gids = get_groups_for_uid(uid)

    if uid == st.st_uid:
        has_read = bool(mode & stat.S_IRUSR)
        has_write = bool(mode & stat.S_IWUSR)
        has_exec = bool(mode & stat.S_IXUSR)
        scope = "owner"
    elif st.st_gid in gids:
        has_read = bool(mode & stat.S_IRGRP)
        has_write = bool(mode & stat.S_IWGRP)
        has_exec = bool(mode & stat.S_IXGRP)
        scope = "group"
    else:
        has_read = bool(mode & stat.S_IROTH)
        has_write = bool(mode & stat.S_IWOTH)
        has_exec = bool(mode & stat.S_IXOTH)
        scope = "other"

    if not has_read:
        return False, (
            f"UID {uid} lacks read permission ({scope}) on {path} "
            f"(owner={st.st_uid}, gid={st.st_gid}, mode={oct(mode)})"
        )

    if path.is_dir() and not has_exec:
        return False, (
            f"UID {uid} lacks execute/traverse permission ({scope}) on directory {path} "
            f"(owner={st.st_uid}, gid={st.st_gid}, mode={oct(mode)})"
        )

    if need_write and not has_write:
        return False, (
            f"UID {uid} lacks write permission ({scope}) on {path} "
            f"(owner={st.st_uid}, gid={st.st_gid}, mode={oct(mode)})"
        )

    return True, ""
