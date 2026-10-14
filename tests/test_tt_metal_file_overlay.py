# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import hashlib
from types import SimpleNamespace

import pytest

from workflows.tt_metal_file_overlay import tt_metal_file_overlay_args


def _policy(root):
    source = root / "reference_config" / "precision.json"
    source.parent.mkdir()
    source.write_text("{}\n")
    return source


def _spec(source, target, digest):
    return SimpleNamespace(
        metadata={
            "tt_metal_file_overlays": [
                {"source": source, "target": target, "sha256": digest}
            ]
        }
    )


def test_precision_overlay_is_read_only_and_hash_bound(tmp_path):
    source = _policy(tmp_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = "models/demos/example/config/precision.json"
    args = tt_metal_file_overlay_args(
        _spec("reference_config/precision.json", target, digest),
        SimpleNamespace(dev_mode=True),
        tmp_path,
    )
    assert args == [
        "--mount",
        f"type=bind,src={source},dst=/home/container_app_user/tt-metal/{target},readonly",
    ]


@pytest.mark.parametrize(
    "source,target,digest,dev_mode",
    [
        ("reference_config/precision.json", "models/demos/example/config/precision.json", "0" * 64, True),
        ("reference_config/precision.json", "models/demos/example/config/precision.json", "unused", False),
        ("/etc/passwd", "models/demos/example/config/precision.json", "unused", True),
        ("reference_config/precision.json", "models/demos/../bad/config/precision.json", "unused", True),
        ("reference_config/precision.json", "models/demos/example/tt/model.py", "unused", True),
    ],
)
def test_rejects_unsafe_or_unattested_overlay(
    tmp_path, source, target, digest, dev_mode
):
    _policy(tmp_path)
    with pytest.raises(ValueError):
        tt_metal_file_overlay_args(
            _spec(source, target, digest),
            SimpleNamespace(dev_mode=dev_mode),
            tmp_path,
        )
