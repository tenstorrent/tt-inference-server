# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Read-only host package directories a device spec mounts into its container.

``DeviceModelSpec.docker_host_packages`` maps a container path to a host directory, e.g.
a prebuilt server's binaries and its prequantized weights on shared storage. Such a spec
brings its own weights, so host setup skips the Hugging Face download.
"""

from pathlib import Path, PurePosixPath
from typing import List, Tuple


def resolve_host_package_mounts(model_spec) -> List[Tuple[Path, str]]:
    """(host directory, container path) pairs of the spec's host packages, in spec order.

    Raises ValueError when a container path is not absolute or a host directory is
    missing: the server cannot start without its package, so fail before docker run.
    """
    packages = _host_packages(model_spec)
    mounts = []
    for container_path, host_path in packages.items():
        container = PurePosixPath(container_path)
        if not container.is_absolute() or container == PurePosixPath("/"):
            raise ValueError(
                f"⛔ docker_host_packages container path must be a non-root absolute path: "
                f"{container_path}"
            )
        host = Path(host_path).expanduser()
        if not host.is_dir():
            raise ValueError(
                f"⛔ docker_host_packages: host directory {host} (for {container_path}) does "
                f"not exist on this host; the server needs it. {_storage_report(host)}"
            )
        mounts.append((host.resolve(), str(container)))
    return mounts


# Mount types worth listing when a package is missing: where shared storage could be instead.
_SHARED_FS_TYPES = (
    "nfs",
    "nfs4",
    "wekafs",
    "cifs",
    "lustre",
    "fuse",
    "fuse.sshfs",
    "ceph",
)


def _storage_report(host: Path, mounts_file: Path = Path("/proc/mounts")) -> str:
    """The deepest existing ancestor of a missing package and the host's shared mounts.

    On a CI runner the error is all we see of the host, so it says where shared storage is.
    """
    ancestor = next((p for p in host.parents if p.is_dir()), None)
    shared = []
    try:
        for line in mounts_file.read_text().splitlines():
            fields = line.split()
            if len(fields) < 3:
                continue
            target, fs_type = fields[1], fields[2]
            if fs_type in _SHARED_FS_TYPES or target.startswith("/mnt"):
                shared.append(f"{target} ({fs_type})")
    except OSError:
        pass
    return (
        f"Deepest existing ancestor: {ancestor}. Shared mounts on this host: "
        f"{', '.join(shared) if shared else 'none'}."
    )


def host_packages_provide_weights(model_spec) -> bool:
    """True when the spec mounts host packages, which carry the server's weights."""
    return bool(_host_packages(model_spec))


def _host_packages(model_spec) -> dict:
    device_spec = getattr(model_spec, "device_model_spec", None)
    packages = getattr(device_spec, "docker_host_packages", None)
    return packages if isinstance(packages, dict) else {}
