# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Unit tests for workflows.compose_config.

Covers version parsing, contract lookup, and sidecar writing. The compose-up /
compose-down wrappers are exercised via integration tests (run.py end-to-end
against real Docker), not here.
"""

import pytest


def test_module_imports():
    """Smoke test: the module imports cleanly."""
    import workflows.compose_config  # noqa: F401


from packaging.version import Version

from workflows.compose_config import parse_image_version


class TestParseImageVersion:
    def test_release_tag_with_build_suffix(self):
        image = "ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.11.0-fae3df"
        assert parse_image_version(image) == Version("0.11.0")

    def test_release_tag_no_suffix(self):
        image = "ghcr.io/foo/bar:0.11.0"
        assert parse_image_version(image) == Version("0.11.0")

    def test_two_part_version(self):
        # Tag like "0.9-abc" should still parse — packaging.Version("0.9") is valid
        assert parse_image_version("foo/bar:0.9-abc") == Version("0.9")

    def test_dev_tag_returns_none(self):
        assert parse_image_version("ghcr.io/foo/bar:dev") is None

    def test_latest_tag_returns_none(self):
        assert parse_image_version("ghcr.io/foo/bar:latest") is None

    def test_no_tag_returns_none(self):
        assert parse_image_version("ghcr.io/foo/bar") is None

    def test_empty_string_returns_none(self):
        assert parse_image_version("") is None

    def test_version_in_image_path_not_tag_returns_none(self):
        # No `:` so no tag, returns None even though "0.11.0" appears in the path
        assert parse_image_version("ghcr.io/foo/0.11.0/bar") is None


from pathlib import Path

from workflows.compose_config import (
    NoMatchingContractError,
    lookup_contract,
)


@pytest.fixture
def contracts_yaml(tmp_path: Path) -> Path:
    """Two engines, each with two eras."""
    p = tmp_path / "contracts.yml"
    p.write_text(
        """\
contracts:
  vllm:
    - min_version: "0.0.0"
      file: docker-compose.vllm-pre-0.11.yml
    - min_version: "0.11.0"
      file: docker-compose.vllm-0.11.yml
  media:
    - min_version: "0.0.0"
      file: docker-compose.media-pre-0.11.yml
    - min_version: "0.11.0"
      file: docker-compose.media-0.11.yml
"""
    )
    return p


@pytest.fixture
def contracts_yaml_only_current(tmp_path: Path) -> Path:
    """Phase-1 state: only the current era is populated."""
    p = tmp_path / "contracts.yml"
    p.write_text(
        """\
contracts:
  vllm:
    - min_version: "0.11.0"
      file: docker-compose.vllm-0.11.yml
  media:
    - min_version: "0.11.0"
      file: docker-compose.media-0.11.yml
"""
    )
    return p


class TestLookupContract:
    def test_exact_match_picks_that_era(self, contracts_yaml):
        path = lookup_contract("vllm", Version("0.11.0"), contracts_path=contracts_yaml)
        assert path.name == "docker-compose.vllm-0.11.yml"

    def test_above_match_picks_that_era(self, contracts_yaml):
        path = lookup_contract("vllm", Version("0.13.5"), contracts_path=contracts_yaml)
        assert path.name == "docker-compose.vllm-0.11.yml"

    def test_below_match_picks_legacy_era(self, contracts_yaml):
        path = lookup_contract("vllm", Version("0.10.4"), contracts_path=contracts_yaml)
        assert path.name == "docker-compose.vllm-pre-0.11.yml"

    def test_pre_zero_picks_lowest_era(self, contracts_yaml):
        # 0.0.0 is the lower bound — anything >= it (which is everything)
        # picks pre-0.11 if there's nothing larger that fits.
        path = lookup_contract("vllm", Version("0.0.1"), contracts_path=contracts_yaml)
        assert path.name == "docker-compose.vllm-pre-0.11.yml"

    def test_media_engine_isolated_from_vllm(self, contracts_yaml):
        path = lookup_contract("media", Version("0.11.0"), contracts_path=contracts_yaml)
        assert path.name == "docker-compose.media-0.11.yml"

    def test_unknown_engine_raises(self, contracts_yaml):
        with pytest.raises(NoMatchingContractError, match="forge"):
            lookup_contract("forge", Version("0.11.0"), contracts_path=contracts_yaml)

    def test_none_version_falls_back_to_newest(self, contracts_yaml):
        path = lookup_contract("vllm", None, contracts_path=contracts_yaml)
        assert path.name == "docker-compose.vllm-0.11.yml"

    def test_below_oldest_era_raises(self, contracts_yaml_only_current):
        # contracts_yaml_only_current has no pre-0.11 entry; a 0.10.4 image
        # has no era that fits (smallest min_version is 0.11.0)
        with pytest.raises(NoMatchingContractError, match="0.10.4"):
            lookup_contract(
                "vllm", Version("0.10.4"), contracts_path=contracts_yaml_only_current
            )

    def test_returned_path_is_relative_to_contracts_dir(self, contracts_yaml):
        # The returned path lives next to contracts.yml, not as a bare basename
        path = lookup_contract("vllm", Version("0.11.0"), contracts_path=contracts_yaml)
        assert path.parent == contracts_yaml.parent


class TestProductionContractsYaml:
    """Sanity tests that exercise the real deploy/contracts.yml against the
    actual InferenceEngine enum values. These catch case-mismatch / missing-engine
    bugs that the isolated-fixture tests above can't see.
    """

    def test_vllm_engine_value_resolves(self):
        from workflows.workflow_types import InferenceEngine

        path = lookup_contract(InferenceEngine.VLLM.value, Version("0.11.0"))
        assert path.name == "docker-compose.vllm-0.11.yml"

    def test_vllm_pre_0_11_resolves(self):
        from workflows.workflow_types import InferenceEngine

        path = lookup_contract(InferenceEngine.VLLM.value, Version("0.10.1"))
        assert path.name == "docker-compose.vllm-pre-0.11.yml"

    def test_media_engine_value_resolves(self):
        from workflows.workflow_types import InferenceEngine

        path = lookup_contract(InferenceEngine.MEDIA.value, Version("0.11.0"))
        assert path.name == "docker-compose.media-0.11.yml"

    def test_forge_engine_value_resolves(self):
        # Forge models route through the media template; production contracts.yml
        # must register the forge engine even though it shares files with media.
        from workflows.workflow_types import InferenceEngine

        path = lookup_contract(InferenceEngine.FORGE.value, Version("0.11.0"))
        assert path.name == "docker-compose.media-0.11.yml"

    def test_every_enum_value_is_registered(self):
        """If a new engine is added to InferenceEngine, the production
        contracts.yml must register it. Otherwise a real run with that engine
        will hit NoMatchingContractError.
        """
        from workflows.workflow_types import InferenceEngine

        for engine in InferenceEngine:
            # Use 0.11.0 which is in-range for every era in the prod file.
            lookup_contract(engine.value, Version("0.11.0"))


from workflows.compose_config import build_compose_command, write_compose_files_sidecar


class TestWriteComposeFilesSidecar:
    def test_writes_dash_f_args(self, tmp_path: Path):
        files = [
            tmp_path / "docker-compose.vllm-0.11.yml",
            tmp_path / "overlays" / "dev-mode.yml",
        ]
        sidecar = tmp_path / ".env.compose.files"
        result = write_compose_files_sidecar(files, path=sidecar)
        assert result == sidecar
        contents = sidecar.read_text().strip()
        # Format: "-f /abs/...vllm-0.11.yml -f /abs/.../dev-mode.yml"
        assert contents.startswith("-f ")
        assert str(files[0]) in contents
        assert str(files[1]) in contents

    def test_single_file(self, tmp_path: Path):
        f = tmp_path / "docker-compose.vllm-0.11.yml"
        sidecar = tmp_path / ".env.compose.files"
        write_compose_files_sidecar([f], path=sidecar)
        assert sidecar.read_text().strip() == f"-f {f}"

    def test_empty_list_writes_empty(self, tmp_path: Path):
        sidecar = tmp_path / ".env.compose.files"
        write_compose_files_sidecar([], path=sidecar)
        assert sidecar.read_text() == "\n"


# ---------------------------------------------------------------------------
# Helpers for TestBuildComposeCommand
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from workflows.workflow_types import InferenceEngine


def _model_spec(image_tag="0.11.0-abc", engine=None):
    """Minimal fake ModelSpec for build_compose_command / resolve_compose_vars."""
    if engine is None:
        engine = InferenceEngine.VLLM.value
    return SimpleNamespace(
        docker_image=f"ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal:{image_tag}",
        inference_engine=engine,
        hf_model_repo="meta-llama/Llama-3.1-8B-Instruct",
        model_name="Llama-3.1-8B-Instruct",
        # device_type.name is read by media/forge branch of resolve_compose_vars
        device_type=SimpleNamespace(name="N150"),
    )


def _runtime_config(dev_mode=False):
    """Minimal fake RuntimeConfig."""
    return SimpleNamespace(
        device="n150",
        service_port=8000,
        bind_host="0.0.0.0",
        dev_mode=dev_mode,
    )


def _setup_config(
    host_model_volume_root=None,
    host_model_weights_mount_dir=None,
    container_model_weights_mount_dir=None,
    container_model_weights_path=None,
    container_tt_metal_cache_dir=None,
    container_model_weights=None,
):
    """Minimal fake SetupConfig.

    When host_model_volume_root is None, resolve_compose_vars falls back to
    generate_docker_volume_name(model_spec), which reads model_spec.impl.impl_id
    and model_spec.model_name.  All tests that pass a setup_config set
    host_model_volume_root to avoid that branch.
    """
    return SimpleNamespace(
        host_model_volume_root=host_model_volume_root,
        host_model_weights_mount_dir=host_model_weights_mount_dir,
        container_model_weights_mount_dir=container_model_weights_mount_dir,
        container_model_weights_path=container_model_weights_path,
        container_tt_metal_cache_dir=container_tt_metal_cache_dir,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

OVERLAYS_DIR = Path(__file__).parent.parent / "deploy" / "overlays"
DEPLOY_DIR = Path(__file__).parent.parent / "deploy"


class TestBuildComposeCommand:
    """Unit tests for build_compose_command overlay-selection logic.

    build_compose_command is a pure function: it reads config objects and
    returns (compose_files: list[Path], compose_vars: dict[str, str]).
    No Docker or subprocess calls are made.
    """

    # ------------------------------------------------------------------
    # 1. Default case — vLLM, no overlays
    # ------------------------------------------------------------------
    def test_default_no_overlays(self):
        ms = _model_spec(image_tag="0.11.0-abc")
        rc = _runtime_config(dev_mode=False)
        files, env = build_compose_command(ms, rc, setup_config=None, json_fpath=None)

        # Only the contract file, no overlays
        assert len(files) == 1
        assert files[0].name == "docker-compose.vllm-0.11.yml"

        # Core vars present
        assert env["DOCKER_IMAGE"] == ms.docker_image
        assert env["HF_MODEL"] == ms.hf_model_repo
        assert env["TT_DEVICE"] == rc.device

        # No overlay-specific vars
        assert "HOST_MODEL_WEIGHTS" not in env
        assert "REPO_ROOT" not in env
        assert "HOST_CACHE_ROOT" not in env
        assert "RUNTIME_MODEL_SPEC_JSON" not in env

    # ------------------------------------------------------------------
    # 2. dev_mode True — adds dev-mode overlay + REPO_ROOT
    # ------------------------------------------------------------------
    def test_dev_mode_adds_overlay_and_repo_root(self):
        ms = _model_spec()
        rc = _runtime_config(dev_mode=True)
        files, env = build_compose_command(ms, rc, setup_config=None, json_fpath=None)

        assert files[0].name == "docker-compose.vllm-0.11.yml"
        assert any(f.name == "dev-mode.yml" for f in files)
        assert "REPO_ROOT" in env
        # No other overlay vars
        assert "HOST_CACHE_ROOT" not in env
        assert "HOST_MODEL_WEIGHTS" not in env
        assert "RUNTIME_MODEL_SPEC_JSON" not in env

    # ------------------------------------------------------------------
    # 3. host_model_volume_root set — adds host-cache overlay + HOST_CACHE_ROOT
    # ------------------------------------------------------------------
    def test_host_cache_overlay(self, tmp_path):
        ms = _model_spec()
        rc = _runtime_config()
        sc = _setup_config(host_model_volume_root=tmp_path / "cache")
        files, env = build_compose_command(ms, rc, setup_config=sc, json_fpath=None)

        assert files[0].name == "docker-compose.vllm-0.11.yml"
        assert any(f.name == "host-cache.yml" for f in files)
        assert env["HOST_CACHE_ROOT"] == str(tmp_path / "cache")
        # No other overlay vars
        assert "HOST_MODEL_WEIGHTS" not in env
        assert "RUNTIME_MODEL_SPEC_JSON" not in env

    # ------------------------------------------------------------------
    # 4. host_model_weights_mount_dir set — adds host-weights overlay
    # ------------------------------------------------------------------
    def test_host_weights_overlay(self, tmp_path):
        weights_dir = tmp_path / "weights"
        container_weights_dir = Path("/models/weights")
        container_weights_path = Path("/models/weights/model")
        container_cache = Path("/cache/tt_metal")

        ms = _model_spec()
        rc = _runtime_config()
        sc = _setup_config(
            host_model_volume_root=tmp_path / "cache",
            host_model_weights_mount_dir=weights_dir,
            container_model_weights_mount_dir=container_weights_dir,
            container_model_weights_path=container_weights_path,
            container_tt_metal_cache_dir=container_cache,
        )
        files, env = build_compose_command(ms, rc, setup_config=sc, json_fpath=None)

        assert any(f.name == "host-weights.yml" for f in files)
        assert env["HOST_MODEL_WEIGHTS"] == str(weights_dir)
        assert env["CONTAINER_MODEL_WEIGHTS"] == str(container_weights_dir)
        # Pre-0.11-era extras also set
        assert env["MODEL_WEIGHTS_PATH"] == str(container_weights_path)
        assert env["TT_CACHE_PATH"] == str(container_cache)
        assert "RUNTIME_MODEL_SPEC_JSON" not in env

    # ------------------------------------------------------------------
    # 5. json_fpath set — adds model-spec overlay + RUNTIME_MODEL_SPEC_JSON
    # ------------------------------------------------------------------
    def test_json_fpath_adds_model_spec_overlay(self, tmp_path):
        json_file = tmp_path / "model_spec.json"
        ms = _model_spec()
        rc = _runtime_config()
        files, env = build_compose_command(
            ms, rc, setup_config=None, json_fpath=json_file
        )

        assert any(f.name == "model-spec.yml" for f in files)
        assert env["RUNTIME_MODEL_SPEC_JSON"] == str(json_file)
        assert env["TT_MODEL_SPEC_HOST_PATH"] == str(json_file)
        assert "HOST_MODEL_WEIGHTS" not in env
        assert "REPO_ROOT" not in env

    # ------------------------------------------------------------------
    # 6. All overlays stacked — order: contract, dev-mode, host-cache,
    #    host-weights, model-spec
    # ------------------------------------------------------------------
    def test_all_overlays_stacking_order(self, tmp_path):
        json_file = tmp_path / "model_spec.json"
        ms = _model_spec()
        rc = _runtime_config(dev_mode=True)
        sc = _setup_config(
            host_model_volume_root=tmp_path / "cache",
            host_model_weights_mount_dir=tmp_path / "weights",
            container_model_weights_mount_dir=Path("/container/weights"),
        )
        files, env = build_compose_command(ms, rc, setup_config=sc, json_fpath=json_file)

        names = [f.name for f in files]
        assert names[0] == "docker-compose.vllm-0.11.yml", "contract must be first"
        assert names.index("dev-mode.yml") < names.index("host-cache.yml"), \
            "dev-mode before host-cache"
        assert names.index("host-cache.yml") < names.index("host-weights.yml"), \
            "host-cache before host-weights"
        assert names.index("host-weights.yml") < names.index("model-spec.yml"), \
            "host-weights before model-spec"

        # All overlay-specific vars present
        assert "REPO_ROOT" in env
        assert "HOST_CACHE_ROOT" in env
        assert "HOST_MODEL_WEIGHTS" in env
        assert "RUNTIME_MODEL_SPEC_JSON" in env
        assert "TT_MODEL_SPEC_HOST_PATH" in env

    # ------------------------------------------------------------------
    # 7. Pre-0.11 image tag — picks vllm-pre-0.11.yml contract
    # ------------------------------------------------------------------
    def test_pre_0_11_image_picks_legacy_contract(self):
        ms = _model_spec(image_tag="0.10.4-abc")
        rc = _runtime_config()
        files, env = build_compose_command(ms, rc, setup_config=None, json_fpath=None)

        assert files[0].name == "docker-compose.vllm-pre-0.11.yml"
        assert len(files) == 1
        assert env["DOCKER_IMAGE"] == ms.docker_image

    # ------------------------------------------------------------------
    # 8. Media engine — picks media-0.11.yml contract
    # ------------------------------------------------------------------
    def test_media_engine_picks_media_contract(self):
        ms = _model_spec(image_tag="0.11.0-abc", engine=InferenceEngine.MEDIA.value)
        rc = _runtime_config()
        files, env = build_compose_command(ms, rc, setup_config=None, json_fpath=None)

        assert files[0].name == "docker-compose.media-0.11.yml"
        # Media branch sets MODEL and DEVICE instead of HF_MODEL / TT_DEVICE
        assert env["MODEL"] == ms.model_name
        assert env["DEVICE"] == ms.device_type.name.lower()
        assert "HF_MODEL" not in env
