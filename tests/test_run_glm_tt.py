# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import hashlib
import importlib.util
import json
from pathlib import Path

import jwt
import pytest

_SRC = Path(__file__).resolve().parents[1] / "vllm-tt-metal" / "src" / "run_glm_tt.py"
_spec = importlib.util.spec_from_file_location("run_glm_tt", _SRC)
run_glm_tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_glm_tt)

# The hardened glm-tt's usage text: --host, and the key from the environment's API_KEY.
USAGE = (
    "[--groups G] [--host ADDR] [--api-key KEY]\n"
    "Without it, serve takes the key from the environment's API_KEY unless NO_AUTH is set.\n"
)


def _weights(tmp_path):
    ttq = tmp_path / "w.ttq"
    vocab = tmp_path / "v.gguf"
    ttq.write_bytes(b"t")
    vocab.write_bytes(b"v")
    return {"GLM_TT_TTQ": str(ttq), "GLM_TT_VOCAB": str(vocab)}


def test_jwt_matches_tis_clients():
    from utils.vllm_run_utils import get_encoded_api_key

    for secret in ("test-secret-456", "s3cr3t with spaces"):
        assert run_glm_tt.tis_jwt(secret) == get_encoded_api_key(secret)
        decoded = jwt.decode(run_glm_tt.tis_jwt(secret), secret, algorithms=["HS256"])
        assert decoded == {"team_id": "tenstorrent", "token_id": "debug-test"}


@pytest.mark.parametrize(
    "env, expected",
    [
        (
            {"VLLM_API_KEY": "v", "JWT_SECRET": "j", "API_KEY": "a"},
            ("v", "VLLM_API_KEY"),
        ),
        ({"JWT_SECRET": "j", "API_KEY": "a"}, (None, "JWT_SECRET")),
        ({"API_KEY": "a"}, ("a", "API_KEY")),
        ({"NO_AUTH": "true", "API_KEY": "a"}, (None, "NO_AUTH")),
    ],
)
def test_key_order(env, expected):
    key, source = run_glm_tt.resolve_api_key(env)
    assert source == expected[1]
    if source == "JWT_SECRET":
        assert key == run_glm_tt.tis_jwt("j")
    else:
        assert key == expected[0]


def test_no_key_is_a_setup_error():
    with pytest.raises(run_glm_tt.SetupError, match="JWT_SECRET"):
        run_glm_tt.resolve_api_key({})


def test_galaxy_serves_four_groups_over_ethernet(tmp_path):
    env = {"DEVICE": "blackhole_galaxy", "MODEL": "GLM-5.3-Flash", "JWT_SECRET": "j"}
    env.update(_weights(tmp_path))
    serve, updates = run_glm_tt.serve_command(env, USAGE, 32)
    assert serve[:2] == ["serve", "-m"]
    assert serve[serve.index("-m") + 1] == env["GLM_TT_VOCAB"]
    assert serve[serve.index("--ttq") + 1] == env["GLM_TT_TTQ"]
    assert serve[serve.index("--groups") + 1] == "4"
    assert serve[serve.index("-n") + 1] == "163840"
    assert serve[serve.index("--port") + 1] == "8000"
    assert serve[serve.index("--host") + 1] == "0.0.0.0"
    assert serve[serve.index("--prefill-chunk") + 1] == "2048"
    assert serve[serve.index("--served-model-name") + 1] == "zai-org/GLM-5.3-Flash"
    # This glm-tt reads the key from the environment, so it is not on the command line.
    assert "--api-key" not in serve
    assert updates == {
        "TT_LAB_GLM_ROUND": "auto",
        "TT_LAB_ETH_GATHER": "1",
        "API_KEY": run_glm_tt.tis_jwt("j"),
    }


def test_loudbox_is_one_group_with_ethernet_gathers(tmp_path):
    env = {"DEVICE": "P150X8", "API_KEY": "a", "GLM_TT_EXTRA_ARGS": "--drain-seconds 4"}
    env.update(_weights(tmp_path))
    serve, updates = run_glm_tt.serve_command(env, USAGE, 8)
    assert "--groups" not in serve
    assert serve[-2:] == ["--drain-seconds", "4"]
    assert updates == {"TT_LAB_ETH_GATHER": "1", "API_KEY": "a"}


def test_settings_in_the_environment_win(tmp_path):
    env = {
        "DEVICE": "bh_galaxy",
        "NO_AUTH": "1",
        "TT_LAB_GLM_ROUND": "pcie",
        "TT_LAB_ETH_GATHER": "0",
        "SERVICE_PORT": "9000",
    }
    env.update(_weights(tmp_path))
    serve, updates = run_glm_tt.serve_command(env, USAGE, 32)
    assert serve[serve.index("--port") + 1] == "9000"
    assert updates == {
        "TT_LAB_GLM_ROUND": "pcie",
        "TT_LAB_ETH_GATHER": "0",
        "API_KEY": None,
    }


def test_glm_tt_with_only_api_key_flag_gets_it_on_the_command_line(tmp_path):
    env = {"DEVICE": "p150x8", "API_KEY": "a"}
    env.update(_weights(tmp_path))
    serve, _ = run_glm_tt.serve_command(env, "[--host ADDR] [--api-key KEY]", 8)
    assert serve[serve.index("--api-key") + 1] == "a"


@pytest.mark.parametrize(
    "env, usage, chips, error",
    [
        ({}, USAGE, 32, "DEVICE is not set"),
        ({"DEVICE": "n300"}, USAGE, 32, "not supported"),
        ({"DEVICE": "blackhole_galaxy"}, USAGE, 8, "needs 32 chips"),
        (
            {"DEVICE": "p150x8", "GLM_TT_TTQ": "/absent.ttq"},
            USAGE,
            8,
            "absent.ttq not found",
        ),
        ({"DEVICE": "p150x8"}, "[--api-key KEY]", 8, "no --host"),
    ],
)
def test_bad_setup(tmp_path, env, usage, chips, error):
    env = {**_weights(tmp_path), "API_KEY": "a", **env}
    with pytest.raises(run_glm_tt.SetupError, match=error):
        run_glm_tt.serve_command(env, usage, chips)


def test_weights_by_name_in_model_weights_dir(tmp_path):
    (tmp_path / "a.ttq").write_bytes(b"t")
    env = {"MODEL_WEIGHTS_DIR": str(tmp_path)}
    assert run_glm_tt.find_weight("a.ttq", env) == str(tmp_path / "a.ttq")
    assert run_glm_tt.find_weight("b.ttq", env) is None


def test_vllm_image_arguments():
    env = {}
    run_glm_tt.parse_vllm_args(
        ["--model", "zai-org/GLM-5.3-Flash", "--tt-device", "p150x8", "--no-auth"], env
    )
    assert env == {"MODEL": "zai-org/GLM-5.3-Flash", "DEVICE": "p150x8", "NO_AUTH": "1"}
    with pytest.raises(run_glm_tt.SetupError, match="unknown argument"):
        run_glm_tt.parse_vllm_args(["--bogus"], env)


def test_dirty_marker(tmp_path):
    host = tmp_path / "host"
    cache = tmp_path / "cache"
    env = {"CACHE_ROOT": str(cache)}
    assert run_glm_tt.dirty_marker(env, str(host)) == f"{cache}/glm-tt.dirty"
    host.mkdir()
    assert run_glm_tt.dirty_marker(env, str(host)) == f"{host}/glm-tt.dirty"
    with pytest.raises(run_glm_tt.SetupError, match="die with the container"):
        run_glm_tt.dirty_marker({"TT_LAB_DIRTY_MARKER": "/tmp/x"}, str(host))


def _package(tmp_path, glm_tt_status=0):
    """A package whose loader is a shell script that runs the "binary" it is given."""
    home = tmp_path / "glm-tt"
    (home / "bin").mkdir(parents=True)
    (home / "runtime").mkdir()
    loader = home / "runtime" / "ld-linux-x86-64.so.2"
    loader.write_text('#!/bin/sh\nshift 2\nexec "$@"\n')
    record = tmp_path / "serve.json"
    binary = home / "bin" / "glm-tt"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "if len(sys.argv) == 1:\n"
        f"    print({USAGE!r}); sys.exit(1)\n"
        f"json.dump({{'argv': sys.argv[1:], 'env': dict(os.environ)}}, open({str(record)!r}, 'w'))\n"
        f"sys.exit({glm_tt_status})\n"
    )
    for path in (loader, binary):
        path.chmod(0o755)
    sums = "".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(home)}\n"
        for p in (binary, loader)
    )
    (home / "SHA256SUMS").write_text(sums)
    return home, record


@pytest.fixture
def galaxy_env(tmp_path, monkeypatch):
    monkeypatch.setattr(run_glm_tt, "HOST_MARKER_DIR", str(tmp_path / "no-host-dir"))
    real_glob = run_glm_tt.glob.glob

    def fake_glob(pattern):
        if pattern.startswith("/dev/tenstorrent/"):
            return [f"/dev/tenstorrent/{i}" for i in range(32)]
        return real_glob(pattern)

    monkeypatch.setattr(run_glm_tt.glob, "glob", fake_glob)
    env = {
        "DEVICE": "blackhole_galaxy",
        "MODEL": "GLM-5.3-Flash",
        "JWT_SECRET": "j",
        "CACHE_ROOT": str(tmp_path / "cache"),
        "PATH": "/usr/bin:/bin",
    }
    env.update(_weights(tmp_path))
    return env


def test_main_runs_serve_from_the_package(tmp_path, galaxy_env):
    home, record = _package(tmp_path)
    galaxy_env["GLM_TT_HOME"] = str(home)
    assert run_glm_tt.main([], galaxy_env) == 0
    served = json.loads(record.read_text())
    assert served["argv"][0] == "serve"
    assert served["argv"][served["argv"].index("--groups") + 1] == "4"
    assert served["env"]["API_KEY"] == run_glm_tt.tis_jwt("j")
    assert served["env"]["TT_LAB_GLM_ROUND"] == "auto"
    assert served["env"]["TT_LAB_ETH_GATHER"] == "1"
    assert served["env"]["TT_LAB_DIRTY_MARKER"] == f"{tmp_path}/cache/glm-tt.dirty"


def test_main_reports_glm_tt_status(tmp_path, galaxy_env):
    home, _ = _package(tmp_path, glm_tt_status=3)
    galaxy_env["GLM_TT_HOME"] = str(home)
    galaxy_env["TT_LAB_DIRTY_MARKER"] = str(tmp_path / "marker")
    assert run_glm_tt.main([], galaxy_env) == 3


def test_main_refuses_a_tampered_package(tmp_path, galaxy_env):
    home, record = _package(tmp_path)
    (home / "bin" / "glm-tt").write_text("#!/bin/sh\n")
    galaxy_env["GLM_TT_HOME"] = str(home)
    assert run_glm_tt.main([], galaxy_env) == run_glm_tt.SETUP_FAILED
    assert not record.exists()


def test_main_needs_the_package(tmp_path, galaxy_env):
    galaxy_env["GLM_TT_HOME"] = str(tmp_path / "absent")
    assert run_glm_tt.main([], galaxy_env) == run_glm_tt.SETUP_FAILED


def _hf_download(tmp_path, glm_tt_status=0):
    """TIS's download of the weights repo: $CACHE_ROOT/weights/$MODEL, no exec bits."""
    weights = tmp_path / "cache" / "weights" / "GLM-5.3-Flash"
    weights.mkdir(parents=True)
    home, record = _package(tmp_path, glm_tt_status)
    home.rename(weights / "glm-tt")
    for path in (weights / "glm-tt").rglob("*"):
        if path.is_file():
            path.chmod(0o644)
    (weights / "w.ttq.manifest").write_text(
        "glm-tt ttq split 1\nsize 1 sha256 x\npart w.ttq.00001-of-00001 0 1 sha256 x\n"
    )
    (weights / "w.ttq.00001-of-00001").write_bytes(b"t")
    (weights / "v.gguf").write_bytes(b"v")
    return weights, record


def test_split_ttq_is_found_by_its_manifest(tmp_path):
    (tmp_path / "w.ttq.manifest").write_text("glm-tt ttq split 1\n")
    env = {"MODEL_WEIGHTS_DIR": str(tmp_path)}
    assert run_glm_tt.find_weight("w.ttq", env) == f"{tmp_path}/w.ttq.manifest"
    (tmp_path / "w.ttq").write_bytes(b"t")
    assert run_glm_tt.find_weight("w.ttq", env) == f"{tmp_path}/w.ttq"
    assert run_glm_tt.find_weight("absent.ttq", env) is None


def test_main_serves_a_downloaded_weights_repo(tmp_path, galaxy_env, monkeypatch):
    weights, record = _hf_download(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(run_glm_tt.tempfile, "tempdir", str(staging))
    galaxy_env.update(GLM_TT_TTQ="w.ttq", GLM_TT_VOCAB="v.gguf")
    assert run_glm_tt.main([], galaxy_env) == 0
    argv = json.loads(record.read_text())["argv"]
    assert argv[argv.index("--ttq") + 1] == f"{weights}/w.ttq.manifest"
    assert argv[argv.index("-m") + 1] == f"{weights}/v.gguf"
    (staged,) = staging.iterdir()
    assert (staged / "bin" / "glm-tt").stat().st_mode & 0o111
    assert not (weights / "glm-tt" / "bin" / "glm-tt").stat().st_mode & 0o111


def test_main_refuses_a_tampered_download_before_running_it(
    tmp_path, galaxy_env, monkeypatch
):
    weights, record = _hf_download(tmp_path)
    (weights / "glm-tt" / "bin" / "glm-tt").write_text("#!/bin/sh\n")
    monkeypatch.setattr(run_glm_tt.tempfile, "tempdir", str(tmp_path))
    galaxy_env.update(GLM_TT_TTQ="w.ttq", GLM_TT_VOCAB="v.gguf")
    assert run_glm_tt.main([], galaxy_env) == run_glm_tt.SETUP_FAILED
    assert not record.exists()
    assert not list(tmp_path.glob("glm-tt-*"))


def test_unexecutable_package_needs_checksums(tmp_path):
    home, _ = _package(tmp_path)
    (home / "SHA256SUMS").unlink()
    (home / "bin" / "glm-tt").chmod(0o644)
    with pytest.raises(run_glm_tt.SetupError, match="no SHA256SUMS"):
        run_glm_tt.runnable_home(home)
    (home / "bin" / "glm-tt").chmod(0o755)
    assert run_glm_tt.runnable_home(home) == home


def test_main_refuses_an_incomplete_split_ttq_before_glm_tt(
    tmp_path, galaxy_env, monkeypatch
):
    weights, record = _hf_download(tmp_path)
    monkeypatch.setattr(run_glm_tt.tempfile, "tempdir", str(tmp_path))
    galaxy_env.update(GLM_TT_TTQ="w.ttq", GLM_TT_VOCAB="v.gguf")
    (weights / "w.ttq.00001-of-00001").write_bytes(b"")
    assert run_glm_tt.main([], galaxy_env) == run_glm_tt.SETUP_FAILED
    (weights / "w.ttq.00001-of-00001").unlink()
    assert run_glm_tt.main([], galaxy_env) == run_glm_tt.SETUP_FAILED
    assert not record.exists()
    assert not (tmp_path / "cache" / "glm-tt.dirty").exists()


def test_check_split_ttq_names_each_problem(tmp_path):
    manifest = tmp_path / "w.ttq.manifest"
    manifest.write_text(
        "glm-tt ttq split 1\npart a 0 2 sha256 x\npart b 2 2 sha256 x\n"
    )
    (tmp_path / "a").write_bytes(b"a")
    with pytest.raises(run_glm_tt.SetupError, match="a has 1 of 2 bytes; b is missing"):
        run_glm_tt.check_split_ttq(str(manifest))
    (tmp_path / "a").write_bytes(b"aa")
    (tmp_path / "b").write_bytes(b"bb")
    run_glm_tt.check_split_ttq(str(manifest))


def test_main_removes_an_old_marker_on_the_shared_cache(tmp_path, galaxy_env):
    home, record = _package(tmp_path)
    galaxy_env["GLM_TT_HOME"] = str(home)
    marker = tmp_path / "cache" / "glm-tt.dirty"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("group A failed\n")
    assert run_glm_tt.main([], galaxy_env) == 0
    assert record.exists()
    assert not marker.exists()


def test_main_keeps_a_host_marker(tmp_path, galaxy_env, monkeypatch):
    host = tmp_path / "host-state"
    host.mkdir()
    monkeypatch.setattr(run_glm_tt, "HOST_MARKER_DIR", str(host))
    home, _ = _package(tmp_path, glm_tt_status=3)
    galaxy_env["GLM_TT_HOME"] = str(home)
    (host / "glm-tt.dirty").write_text("group A failed\n")
    assert run_glm_tt.main([], galaxy_env) == 3
    assert (host / "glm-tt.dirty").exists()
    explicit = tmp_path / "cache" / "glm-tt.dirty"
    explicit.parent.mkdir(parents=True, exist_ok=True)
    explicit.write_text("x")
    assert (
        run_glm_tt.marker_is_shared(str(explicit), galaxy_env, explicit=True) is False
    )
