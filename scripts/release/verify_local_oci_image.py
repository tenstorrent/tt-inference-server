# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Match an imported Docker image to pinned OCI metadata without running it.

Docker image stores can use either the config digest or manifest digest as Id.
This checks both pinned metadata blobs, platform, layers and runtime config;
it neither imports an image nor qualifies model accuracy or hardware startup.
"""

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path


def read_blob(archive, digest):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Expected a full sha256 digest")
    name = "blobs/sha256/" + digest.split(":", 1)[1]
    members = [member for member in archive.getmembers() if member.name == name]
    if len(members) != 1 or not members[0].isfile() or members[0].size > 1024**2:
        raise ValueError("Expected one bounded regular OCI metadata blob")
    with archive.extractfile(members[0]) as stream:
        data = stream.read()
    if "sha256:" + hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("OCI metadata content differs from its pinned digest")
    return json.loads(data), len(data)


def verify(archive_path, manifest_digest, config_digest, inspected):
    with tarfile.open(archive_path, "r:") as archive:
        manifest, manifest_bytes = read_blob(archive, manifest_digest)
        config, config_bytes = read_blob(archive, config_digest)
    if (
        manifest.get("schemaVersion") != 2
        or manifest.get("config", {}).get("digest") != config_digest
        or manifest["config"].get("size") != config_bytes
    ):
        raise ValueError("Manifest does not reference the pinned image config")
    if inspected.get("Id") not in (config_digest, manifest_digest):
        raise ValueError("Docker image Id differs from both pinned identities")
    descriptor = inspected.get("Descriptor")
    if descriptor is not None and (
        descriptor.get("digest") != manifest_digest
        or descriptor.get("size") != manifest_bytes
    ):
        raise ValueError("Docker descriptor differs from the pinned manifest")
    for actual, expected in (("Os", "os"), ("Architecture", "architecture")):
        if not config.get(expected) or inspected.get(actual) != config[expected]:
            raise ValueError("Docker platform differs from pinned OCI config")
    rootfs = config.get("rootfs", {})
    layers = rootfs.get("diff_ids", [])
    if (
        rootfs.get("type") != "layers"
        or not layers
        or len(manifest.get("layers", [])) != len(layers)
        or inspected.get("RootFS") != {"Type": "layers", "Layers": layers}
    ):
        raise ValueError("Docker root filesystem differs from pinned OCI config")
    expected_config = config.get("config", {})
    actual_config = inspected.get("Config", {})
    if not expected_config or any(
        actual_config.get(key) != expected_config.get(key)
        for key in set(actual_config) | set(expected_config)
    ):
        raise ValueError("Docker runtime config differs from pinned OCI config")
    return {
        "state": "local_image_identity_verified_unqualified",
        "image_id": inspected["Id"],
        "manifest_digest": manifest_digest,
        "config_digest": config_digest,
        "layers": len(layers),
        "architecture": config["architecture"],
        "os": config["os"],
        "labels": expected_config.get("Labels", {}),
        "hardware_opened": False,
        "accuracy_qualified": False,
        "archive_layers_rehashed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest-digest", required=True)
    parser.add_argument("--config-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Identity receipt already exists")
    # A digest is immutable; never fall back to a mutable tag.
    inspected = None
    for identity in (args.manifest_digest, args.config_digest):
        result = subprocess.run(
            ["docker", "image", "inspect", identity],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            images = json.loads(result.stdout)
            if len(images) != 1:
                raise ValueError("Expected exactly one imported image")
            inspected = images[0]
            break
    if inspected is None:
        raise ValueError("Neither pinned identity is present in the Docker image store")
    report = verify(args.archive, args.manifest_digest, args.config_digest, inspected)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
