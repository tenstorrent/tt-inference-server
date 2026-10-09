# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Imported OCI identity must work across Docker stores without weakening pins."""

import copy
import hashlib
import io
import json
import tarfile

import pytest

from scripts.release.verify_local_oci_image import verify


def fixture(tmp_path, *, corrupt=False, wrong_config=False, duplicate=False):
    layer = "sha256:" + "1" * 64
    config = {
        "architecture": "amd64",
        "os": "linux",
        "config": {"Env": ["MODEL_SHA=qualified"], "Labels": {"source": "pinned"}},
        "rootfs": {"type": "layers", "diff_ids": [layer]},
    }
    raw_config = json.dumps(config).encode()
    config_digest = "sha256:" + hashlib.sha256(raw_config).hexdigest()
    manifest = {
        "schemaVersion": 2,
        "config": {
            "digest": "sha256:" + "2" * 64 if wrong_config else config_digest,
            "size": len(raw_config),
        },
        "layers": [{"digest": layer}],
    }
    raw_manifest = json.dumps(manifest).encode()
    manifest_digest = "sha256:" + hashlib.sha256(raw_manifest).hexdigest()
    path = tmp_path / "image.tar"
    with tarfile.open(path, "w") as archive:
        blobs = [(config_digest, raw_config), (manifest_digest, raw_manifest)]
        if duplicate:
            blobs.append(blobs[0])
        for digest, data in blobs:
            if corrupt and digest == config_digest:
                data += b" "
            member = tarfile.TarInfo("blobs/sha256/" + digest.split(":")[1])
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    image = {
        "Id": config_digest,
        "Config": copy.deepcopy(config["config"]),
        "Architecture": "amd64",
        "Os": "linux",
        "RootFS": {"Type": "layers", "Layers": [layer]},
    }
    descriptor = {"digest": manifest_digest, "size": len(raw_manifest)}
    return path, manifest_digest, config_digest, image, descriptor


@pytest.mark.parametrize("store", ["legacy", "containerd"])
def test_same_pinned_image_accepts_both_docker_identity_forms(tmp_path, store):
    path, manifest, config, image, descriptor = fixture(tmp_path)
    if store == "containerd":
        image.update(Id=manifest, Descriptor=descriptor)
    report = verify(path, manifest, config, image)
    assert report["image_id"] == image["Id"]
    assert report["manifest_digest"] == manifest
    assert report["config_digest"] == config
    assert not report["accuracy_qualified"]
    assert not report["hardware_opened"]


@pytest.mark.parametrize(
    "fault", ["identity", "descriptor", "layers", "platform", "env", "labels"]
)
def test_wrong_imported_image_is_rejected_even_with_matching_id(tmp_path, fault):
    path, manifest, config, image, descriptor = fixture(tmp_path)
    if fault == "identity":
        image["Id"] = "sha256:" + "0" * 64
    elif fault == "descriptor":
        image["Descriptor"] = {**descriptor, "digest": "sha256:" + "0" * 64}
    elif fault == "layers":
        image["RootFS"]["Layers"] = ["sha256:" + "0" * 64]
    elif fault == "platform":
        image["Architecture"] = "arm64"
    elif fault == "env":
        image["Config"]["Env"] = ["MODEL_SHA=unqualified"]
    else:
        image["Config"]["Labels"]["source"] = "different"
    with pytest.raises(ValueError, match="differs"):
        verify(path, manifest, config, image)


@pytest.mark.parametrize("fault", ["corrupt", "wrong_config", "duplicate"])
def test_archive_metadata_chain_is_checked(tmp_path, fault):
    path, manifest, config, image, _ = fixture(tmp_path, **{fault: True})
    with pytest.raises(ValueError, match="differs|reference|one bounded"):
        verify(path, manifest, config, image)
