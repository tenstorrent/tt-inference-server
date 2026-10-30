# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""vllm-tt-metal/src/run_dsv41_tt.py: the dsv41-serve command it builds (no device, no binary)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "vllm-tt-metal" / "src" / "run_dsv41_tt.py"
spec = importlib.util.spec_from_file_location("run_dsv41_tt", SCRIPT)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

USAGE = "dsv41-serve -m CHECKPOINT_DIR [--host ADDR] ... TT_LAB_API_KEY, when set, ..."


def checkpoint(tmp_path, split=True, part_bytes=4):
    weights = tmp_path / "cache" / "weights" / "DeepSeek-V4.1-Flash-1017"
    weights.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json", "token_map.u32"):
        (weights / name).write_text("{}")
    if split:
        name = "model-00047-of-00048.safetensors"
        (weights / f"{name}.part1-of-2").write_bytes(b"x" * part_bytes)
        (weights / f"{name}.part2-of-2").write_bytes(b"y" * 4)
        (weights / f"{name}.manifest").write_text(
            f"part {name}.part1-of-2 0 4 aa\npart {name}.part2-of-2 4 4 bb\n"
        )
    return weights


def env_for(tmp_path, **extra):
    env = {
        "DEVICE": "blackhole_galaxy",
        "MODEL": "DeepSeek-V4.1-Flash-1017",
        "CACHE_ROOT": str(tmp_path / "cache"),
        "API_KEY": "k",
    }
    env.update(extra)
    return env


def test_galaxy_command(tmp_path):
    weights = checkpoint(tmp_path)
    serve, updates = run.serve_command(env_for(tmp_path), USAGE, 32)
    assert serve[:3] == ["dsv41-serve", "-m", str(weights)]
    assert serve[serve.index("--token-map") + 1] == str(weights / "token_map.u32")
    assert serve[serve.index("--context") + 1] == "262144"
    assert serve[serve.index("--max") + 1] == "131072"
    assert serve[serve.index("--slots") + 1] == "4"
    assert serve[serve.index("--groups") + 1] == "4"
    assert serve[serve.index("--host") + 1] == "0.0.0.0"
    assert serve[serve.index("--model") + 1] == "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    assert "k" not in serve  # the key never reaches argv
    assert updates["TT_LAB_API_KEY"] == "k"
    assert updates["TT_LAB_DEVICE_GROUPS"] == "0,24,16,8"


def test_key_order_and_group_override(tmp_path):
    checkpoint(tmp_path)
    env = env_for(tmp_path, VLLM_API_KEY="v", JWT_SECRET="j", TT_LAB_DEVICE_GROUPS="0,8,24,16")
    _, updates = run.serve_command(env, USAGE, 32)
    assert updates["TT_LAB_API_KEY"] == "v"
    assert updates["TT_LAB_DEVICE_GROUPS"] == "0,8,24,16"


def test_refusals(tmp_path):
    checkpoint(tmp_path, part_bytes=3)
    with pytest.raises(run.SetupError, match="incomplete"):
        run.serve_command(env_for(tmp_path), USAGE, 32)
    with pytest.raises(run.SetupError, match="needs 32 chips"):
        run.serve_command(env_for(tmp_path), USAGE, 8)
    with pytest.raises(run.SetupError, match="not supported"):
        run.serve_command(env_for(tmp_path, DEVICE="p150x8"), USAGE, 32)


def test_needs_host_flag(tmp_path):
    checkpoint(tmp_path)
    with pytest.raises(run.SetupError, match="--host"):
        run.serve_command(env_for(tmp_path), "dsv41-serve -m DIR", 32)
