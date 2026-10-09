# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Exercise the Galaxy bundle through the real ModelSpec, wrapper and Helm chart."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.release import prepare_qwen38_galaxy as release
from tests.test_run_vllm_api_server import run_vllm_api_server_module  # noqa: F401
from utils.pinned_artifacts import get_pinned_revision


@pytest.fixture(
    params=[
        "precision_accurate_decode.json",
        "precision_accurate_decode_bfp8_head.json",
        "precision_accurate_decode_bfp8_all.json",
    ]
)
def spec(request):
    # The hardware contract is independently checked against an actual G0 receipt
    # by the preparation CLI. This fixture supplies a distinct partition to catch
    # accidental substitution of catalog defaults by the standard TTIS wrapper.
    groups = [",".join(str(4 * rank + chip) for chip in range(4)) for rank in range(8)]
    contract = SimpleNamespace(
        qualified_groups=lambda receipt: groups,
        OPTIMIZATION_ENV={"QWEN_VLLM_KV_POOL_TOKENS": "1050592"},
        precision_fingerprint=lambda policy: "a" * 64,
        MODEL_NAME="Qwen/Qwen3.8-27B",
        additional_config=lambda selected: {
            "_tt_standard_dp_visible_groups": selected,
            "tt": {"fabric_config": "FABRIC_1D"},
        },
    )
    return release.runtime_spec(contract, {"precision": {}}, "b" * 40, request.param)


def test_runtime_bundle_survives_wrapper(
    spec,
    tmp_path,
    monkeypatch,
    run_vllm_api_server_module,  # noqa: F811
):
    module = run_vllm_api_server_module
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setenv("RUNTIME_MODEL_SPEC_JSON_PATH", str(path))
    loaded = module.load_model_spec("Qwen3.8-27B", "blackhole_galaxy")
    assert loaded == spec
    assert loaded["device_model_spec"]["max_concurrency"] == 128
    assert loaded["device_model_spec"]["max_tokens_all_users_override"] == 8404736
    assert get_pinned_revision(loaded, required=True) == release.WEIGHTS_SHA
    with monkeypatch.context() as environment:
        environment.setenv("TT_METAL_OPERATION_TIMEOUT_SECONDS", "5.0")
        environment.setenv(
            "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE", "old-timeout-command"
        )
        # The wrapper mutates os.environ directly, so restore it after this test.
        import os

        environment.setattr(os, "environ", os.environ.copy())
        module.set_runtime_env_vars(loaded)
        module.set_metal_timeout_env_vars()
        assert "TT_METAL_OPERATION_TIMEOUT_SECONDS" not in os.environ
        assert "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE" not in os.environ
        assert os.environ["MESH_DEVICE"] == "(8, 4)"
        assert os.environ["QWEN_EXPECTED_PRECISION_SHA256"] == "a" * 64
        assert (
            os.environ["QWEN_PRECISION_CONFIG"]
            == spec["env_vars"]["QWEN_PRECISION_CONFIG"]
        )
    monkeypatch.setattr(sys, "argv", ["run_vllm_api_server.py"])
    module.set_vllm_sys_argv(
        argparse.Namespace(service_port=None),
        [],
        loaded["device_model_spec"]["vllm_args"],
    )
    # vLLM accepts underscore/hyphen aliases; the wrapper preserves both forms.
    argv = [
        token.replace("_", "-") if token.startswith("--") else token
        for token in sys.argv
    ]
    assert argv[argv.index("--data-parallel-size") + 1] == "8"
    assert argv[argv.index("--max-num-seqs") + 1] == "16"
    assert argv[argv.index("--max-model-len") + 1] == "262144"
    assert argv[argv.index("--model") + 1] == "/mnt/hf-cache"
    assert "--no-enable-prefix-caching" in argv
    assert "--no-enable-chunked-prefill" in argv
    assert "--seed" not in argv
    actual = json.loads(argv[argv.index("--additional-config") + 1])
    assert actual == spec["device_model_spec"]["vllm_args"]["additional_config"]


@pytest.mark.parametrize("api_key", ["", "qwen-health-test-only"])
def test_galaxy_chart_claims_full_device_and_keeps_weights_read_only(
    spec, tmp_path, api_key
):
    values = release.helm_values(spec, Path("/host/checkpoint"))
    values["defaults"] = {"image": {"digest": "sha256:" + "1" * 64}}
    values["auth"] = {"apiKey": api_key}
    path = tmp_path / "values.yaml"
    path.write_text(yaml.safe_dump(values))
    rendered = subprocess.check_output(
        [
            "helm",
            "template",
            "qwen38-test",
            str(release.REPO / "charts/tt-inference-server"),
            "-f",
            str(path),
        ],
        text=True,
    )
    documents = list(yaml.safe_load_all(rendered))
    deployment = next(doc for doc in documents if doc and doc["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["image"] == release.IMAGE_REPOSITORY + "@sha256:" + "1" * 64
    assert all(item["image"] == release.INIT_IMAGE for item in pod["initContainers"])
    assert container["resources"]["limits"]["hugepages-1Gi"] == "32Gi"
    assert container["resources"]["limits"]["memory"] == "384Gi"
    assert (
        next(
            item["value"]
            for item in container["env"]
            if item["name"] == "QWEN_PRECISION_CONFIG"
        )
        == spec["env_vars"]["QWEN_PRECISION_CONFIG"]
    )
    assert (
        next(
            mount for mount in container["volumeMounts"] if mount["name"] == "hf-cache"
        )["readOnly"]
        is True
    )
    assert (
        container["startupProbe"]["failureThreshold"]
        * container["startupProbe"]["periodSeconds"]
        == 7200
    )
    # vLLM guards /v1/models when VLLM_API_KEY is set. Kubernetes probes
    # carry no bearer credentials; /health checks the engine without auth.
    for probe in ("startupProbe", "livenessProbe", "readinessProbe"):
        assert container[probe]["httpGet"] == {"path": "/health", "port": "http"}
    if api_key:
        secret = next(doc for doc in documents if doc and doc["kind"] == "Secret")
        assert secret["stringData"]["VLLM_API_KEY"] == api_key
    claim = next(
        doc for doc in documents if doc and doc["kind"] == "ResourceClaimTemplate"
    )
    request = claim["spec"]["spec"]["devices"]["requests"][0]
    assert request["exactly"]["count"] == 32
    assert "galaxy-blackhole" in json.dumps(request)


def test_unpinned_bundle_cannot_render(spec, tmp_path):
    path = tmp_path / "values.yaml"
    path.write_text(yaml.safe_dump(release.helm_values(spec, Path("/host/checkpoint"))))
    result = subprocess.run(
        [
            "helm",
            "template",
            "qwen38-test",
            str(release.REPO / "charts/tt-inference-server"),
            "-f",
            str(path),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "digest" in result.stderr


@pytest.mark.parametrize("weights", ["relative/checkpoint", "/"])
def test_invalid_weight_mount_rejected_before_source_access(weights):
    with pytest.raises(ValueError, match="absolute checkpoint"):
        release.prepare(SimpleNamespace(weights_host_path=Path(weights)))


@pytest.mark.parametrize(
    "precision",
    ["../precision.json", "/tmp/precision.json", "config/policy.json", "policy", ""],
)
def test_precision_must_be_a_json_basename_before_source_access(precision):
    with pytest.raises(ValueError, match="Precision must name a JSON"):
        release.prepare(
            SimpleNamespace(weights_host_path=Path("/weights"), precision=precision)
        )


@pytest.fixture
def committed_model(tmp_path):
    source = tmp_path / "source"
    model = source / "models/demos/qwen38_27b_qb2"
    (model / "config").mkdir(parents=True)
    (model / "tt").mkdir()
    (model / "tt/model.py").write_text("# qualified runtime\n")
    for name in ("precision.json", "precision_accurate_decode_bfp8_head.json"):
        (model / "config" / name).write_text('{"head": "bfloat8_b"}\n')
    (source / ".gitignore").write_text("*.ignored.json\n")
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Release Test",
            "-c",
            "user.email=release-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--quiet",
            "-m",
            "Fixture",
        ],
        check=True,
    )
    pin = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    return source, model, pin


def receipt_hashes(model, precision):
    return {
        name: hashlib.sha256((model / relative).read_bytes()).hexdigest()
        for name, relative in (
            ("tt/model.py", "tt/model.py"),
            ("config/precision.json", "config/precision.json"),
            ("effective_precision_override", "config/" + precision),
        )
    }


def test_committed_alternate_precision_is_reproducible(committed_model):
    source, model, pin = committed_model
    precision = "precision_accurate_decode_bfp8_head.json"
    release.verify_committed_source(
        source, pin, receipt_hashes(model, precision), precision
    )


@pytest.mark.parametrize("precision", ["new-policy.json", "new-policy.ignored.json"])
def test_untracked_or_ignored_qualified_precision_cannot_claim_a_git_sha(
    committed_model, precision
):
    source, model, pin = committed_model
    (model / "config" / precision).write_text('{"head": "bfloat8_b"}\n')
    # The former cleanliness check succeeds even though the build cannot fetch
    # this locally qualified policy from the advertised commit.
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "models/demos/qwen38_27b_qb2",
        ],
        check=True,
    )
    with pytest.raises(ValueError, match="absent from the recorded model commit"):
        release.verify_committed_source(
            source, pin, receipt_hashes(model, precision), precision
        )


def test_uncommitted_runtime_and_changed_blob_are_rejected(committed_model):
    source, model, pin = committed_model
    precision = "precision_accurate_decode_bfp8_head.json"
    hashes = receipt_hashes(model, precision)
    (model / "tt/new_kernel.py").write_text("# local-only qualified kernel\n")
    hashes["tt/new_kernel.py"] = hashlib.sha256(
        (model / "tt/new_kernel.py").read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="absent from the recorded model commit"):
        release.verify_committed_source(source, pin, hashes, precision)
    hashes.pop("tt/new_kernel.py")
    hashes["tt/model.py"] = hashlib.sha256(b"# different qualified bytes\n").hexdigest()
    with pytest.raises(ValueError, match="differs from the recorded model commit"):
        release.verify_committed_source(source, pin, hashes, precision)
