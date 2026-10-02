# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import hashlib
import json
from types import SimpleNamespace

import pytest

from workflows.autoport_precision_overlay import (
    pinned_hf_cache_args,
    precision_overlay_args,
    source_overlay_args,
)


def spec(path="reference_config/policy.json", code_path="models/autoports/example"):
    return SimpleNamespace(
        metadata={"autoport_precision_config": path},
        impl=SimpleNamespace(code_path=code_path),
    )


def policy(root):
    target = root / "reference_config/policy.json"
    target.parent.mkdir()
    target.write_text(
        json.dumps({"schema_version": 1, "config_id": "diagnostic_unselected"})
    )
    return target


def source_policy(root):
    model = spec()
    files = {}
    for name in ("precision_policy", "multichip_decoder"):
        relative = f"reference_config/autoport_sources/{name}.source"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source fixture\n")
        files[f"tt/{name}.py"] = {
            "source": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    model.metadata["autoport_source_overlay"] = {
        "tt_metal_revision": "a" * 40,
        "files": files,
    }
    return model


def test_attested_source_overlay_is_readonly_and_keeps_image_native_code(tmp_path):
    model = source_policy(tmp_path)
    args = source_overlay_args(model, SimpleNamespace(dev_mode=True), tmp_path)
    assert len(args) == 4
    assert args[::2] == ["--mount", "--mount"]
    assert all(arg.endswith(",readonly") for arg in args[1::2])
    assert all("/tt/" in arg and ".py,readonly" in arg for arg in args[1::2])
    assert source_overlay_args(spec(), SimpleNamespace(dev_mode=False), tmp_path) == []


@pytest.mark.parametrize("failure", ["nondev", "revision", "target", "hash", "escape"])
def test_source_overlay_rejects_unattested_or_unscoped_files(tmp_path, failure):
    model = source_policy(tmp_path)
    overlay = model.metadata["autoport_source_overlay"]
    entry = overlay["files"]["tt/precision_policy.py"]
    if failure == "revision":
        overlay["tt_metal_revision"] = "main"
    elif failure == "target":
        overlay["files"]["../../native.so"] = entry
    elif failure == "hash":
        entry["sha256"] = "b" * 64
    elif failure == "escape":
        source = tmp_path / entry["source"]
        outside = tmp_path / "outside.source"
        outside.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(outside)
    with pytest.raises(ValueError):
        source_overlay_args(
            model, SimpleNamespace(dev_mode=failure != "nondev"), tmp_path
        )


def test_schema_two_requires_attested_source_overlay(tmp_path):
    target = policy(tmp_path)
    target.write_text('{"schema_version":2,"config_id":"prefill_control"}')
    with pytest.raises(ValueError, match="attested"):
        precision_overlay_args(spec(), SimpleNamespace(dev_mode=True), tmp_path)
    model = source_policy(tmp_path)
    assert precision_overlay_args(model, SimpleNamespace(dev_mode=True), tmp_path)


def test_no_overlay_preserves_existing_command(tmp_path):
    assert (
        precision_overlay_args(
            SimpleNamespace(metadata={}), SimpleNamespace(dev_mode=False), tmp_path
        )
        == []
    )


def test_pinned_cache_is_explicit_and_set_before_python():
    model = spec()
    assert pinned_hf_cache_args(model) == []
    model.metadata["autoport_pinned_hf_cache"] = True
    assert pinned_hf_cache_args(model) == [
        "--env",
        "HF_HOME=/home/container_app_user/cache_root/huggingface",
        "--env",
        "HF_HUB_CACHE=/home/container_app_user/cache_root/huggingface/hub",
    ]
    model.impl.code_path = "models/demos/example"
    with pytest.raises(ValueError, match="autoport implementation"):
        pinned_hf_cache_args(model)


def test_explicit_overlay_is_readonly_and_scoped(tmp_path, caplog):
    source = policy(tmp_path)
    with caplog.at_level("INFO"):
        args = precision_overlay_args(spec(), SimpleNamespace(dev_mode=True), tmp_path)
    assert args == [
        "--mount",
        f"type=bind,src={source},dst=/home/container_app_user/tt-metal/models/autoports/example/doc/datatype_sweep/selected_precision_config.json,readonly",
    ]
    assert "sha256=" in caplog.text


@pytest.mark.parametrize(
    "path,code_path,dev",
    [
        ("reference_config/policy.json", "models/autoports/example", False),
        ("../policy.json", "models/autoports/example", True),
        ("/etc/passwd", "models/autoports/example", True),
        ("reference_config/policy.json", "models/demos/example", True),
        ("reference_config/policy.json", "models/autoports/../example", True),
    ],
)
def test_rejects_unsafe_or_nondev_overlays(tmp_path, path, code_path, dev):
    policy(tmp_path)
    with pytest.raises(ValueError):
        precision_overlay_args(
            spec(path, code_path), SimpleNamespace(dev_mode=dev), tmp_path
        )


def test_rejects_symlink_escape_and_invalid_policy(tmp_path):
    target = policy(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text(target.read_text())
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(ValueError):
        precision_overlay_args(spec(), SimpleNamespace(dev_mode=True), tmp_path)
    target.unlink()
    target.write_text('{"schema_version":true,"config_id":"bad"}')
    with pytest.raises(ValueError):
        precision_overlay_args(spec(), SimpleNamespace(dev_mode=True), tmp_path)
