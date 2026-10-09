# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Prepare the experimental Galaxy image's standard TTIS runtime ModelSpec.

Requires a passing exact-source G0 receipt. This is not an accuracy promotion.
The receipt's physical chip grouping is specific to the qualified host.
"""

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from workflows.model_spec import DeviceModelSpec, ModelSpec, qwen38_27b_qb2_impl
from workflows.workflow_types import DeviceTypes, InferenceEngine, ModelStatusTypes

METAL_SHA = "a08819ddbe23077f8037d3802303939064868ff6"
PLUGIN_SHA = "b7e4292e4193cba20abe9c7c68ce489201b2e36b"
WEIGHTS_SHA = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
BASE_IMAGE = "ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64@sha256:6cec3f1fc25931126d0ed90dff4d7e62128c03db85bcb1313f4cb9bc40c0f8cb"
CONTAINER_MODEL = "/home/container_app_user/tt-metal/models/demos/qwen38_27b_qb2"
IMAGE_REPOSITORY = "ghcr.io/tenstorrent/tt-inference-server/qwen38-galaxy"
INIT_IMAGE = (
    "busybox@sha256:66a6306db78bf2dbf3487f293aa8d6990d8e506fdffab9cc43fe422becf886e4"
)


def runtime_spec(contract, receipt, model_sha, precision_name):
    groups = contract.qualified_groups(receipt)
    environment = dict(contract.OPTIMIZATION_ENV)
    environment.update(
        QWEN_PRECISION_CONFIG=f"{CONTAINER_MODEL}/config/{precision_name}",
        QWEN_EXPECTED_PRECISION_SHA256=contract.precision_fingerprint(
            receipt["precision"]
        ),
        EXTRA_MODELS_DIR="/home/container_app_user/tt-metal/models/demos",
        MESH_DEVICE="(8, 4)",
        HF_HUB_OFFLINE="1",
        VLLM_DEBUG_LOG_API_SERVER_RESPONSE="false",
        VLLM_NO_USAGE_STATS="1",
        TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES="0",
        # The host-qualified launch does not enable the wrapper's 5s watchdog.
        DISABLE_METAL_OP_TIMEOUT="1",
    )
    vllm = {
        "model": "/mnt/hf-cache",
        "served_model_name": contract.MODEL_NAME,
        "revision": WEIGHTS_SHA,
        "tokenizer_revision": WEIGHTS_SHA,
        "hf-overrides": {"architectures": ["TTQwen38ForCausalLM"]},
        "host": "0.0.0.0",
        "port": "8000",
        "data_parallel_size": 8,
        "block_size": "32",
        "max_num_seqs": "16",
        "max_model_len": "262144",
        "max_num_batched_tokens": "262144",
        "max-logprobs": "-1",
        "async-scheduling": True,
        "no-enable-prefix-caching": True,
        "no-enable-chunked-prefill": True,
        "no-enable-log-requests": True,
        "no-enable-log-outputs": True,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
        "enable_auto_tool_choice": True,
        "additional_config": contract.additional_config(groups),
    }
    spec = ModelSpec(
        model_id="qwen38-27b-qb2-galaxy-experimental",
        impl=qwen38_27b_qb2_impl,
        hf_model_repo=contract.MODEL_NAME,
        model_name="Qwen3.8-27B",
        inference_engine=InferenceEngine.VLLM.value,
        device_type=DeviceTypes.BLACKHOLE_GALAXY,
        device_model_spec=DeviceModelSpec(
            device=DeviceTypes.BLACKHOLE_GALAXY,
            max_concurrency=128,
            max_context=262144,
            max_tokens_all_users_override=8 * 1050592,
            default_impl=True,
            env_vars=environment,
            vllm_args=vllm,
        ),
        tt_metal_commit=METAL_SHA,
        vllm_commit=PLUGIN_SHA,
        status=ModelStatusTypes.EXPERIMENTAL,
        has_builtin_warmup=True,
        metadata={
            "pinned_checkpoint": True,
            "model_source_commit": model_sha,
            "qualification": "G0 only; accuracy and container hardware validation pending",
            "reasoning_parser_name": "qwen3",
            "tool_call_parser_name": "qwen3_coder",
        },
    ).get_serialized_dict()
    # These inherited workflow defaults were absent from the measured launch.
    for key in ("seed", "max-log-len"):
        spec["device_model_spec"]["vllm_args"].pop(key, None)
    for key in ("TORCHDYNAMO_DISABLE", "VLLM_RPC_TIMEOUT", "VLLM_CONFIGURE_LOGGING"):
        spec["env_vars"].pop(key, None)
    return spec


def helm_values(spec, weights_host_path):
    return {
        "model": "Qwen3.8-27B",
        "device": "blackhole_galaxy",
        "engine": "vllm",
        "impl": "qwen38_27b_qb2",
        "requireImageDigests": True,
        "initContainerImage": INIT_IMAGE,
        "hfCacheDir": str(weights_host_path),
        "deviceBoardNames": {"blackhole_galaxy": "galaxy-blackhole"},
        "deviceBoardCounts": {"blackhole_galaxy": 32},
        "deviceChipCounts": {"blackhole_galaxy": 32},
        "models": {
            "Qwen3.8-27B": {
                "vllm": {
                    "blackhole_galaxy": {
                        "defaultImpl": "qwen38_27b_qb2",
                        "impls": {
                            "qwen38_27b_qb2": {
                                "image": {"repository": IMAGE_REPOSITORY},
                                "progressDeadlineSeconds": 7200,
                                "resources": {
                                    "requests": {"cpu": "24", "memory": "256Gi"},
                                    "limits": {"memory": "384Gi"},
                                },
                                "env": [
                                    {"name": key, "value": value}
                                    for key, value in sorted(spec["env_vars"].items())
                                ],
                            }
                        },
                    }
                }
            }
        },
    }


def verify_committed_source(source, model_sha, source_hashes, precision):
    """Require qualified runtime bytes to be reproducible from MODEL_SHA alone."""
    model_relative = Path("models/demos/qwen38_27b_qb2")
    for name, expected in source_hashes.items():
        relative = model_relative / (
            "config/" + precision if name == "effective_precision_override" else name
        )
        try:
            content = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(source),
                    "show",
                    f"{model_sha}:{relative.as_posix()}",
                ],
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"Qualified file is absent from the recorded model commit: {relative}"
            ) from error
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(
                f"Qualified file differs from the recorded model commit: {relative}"
            )


def prepare(args):
    if not args.weights_host_path.is_absolute() or args.weights_host_path == Path("/"):
        raise ValueError("Weights host path must be an absolute checkpoint directory")
    if Path(args.precision).name != args.precision or not args.precision.endswith(
        ".json"
    ):
        raise ValueError(
            "Precision must name a JSON file in the model config directory"
        )
    source = args.model_source.resolve()
    model_sha = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if not re.fullmatch("[0-9a-f]{40}", model_sha):
        raise ValueError("Model source needs a full Git commit")
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
    model = source / "models/demos/qwen38_27b_qb2"
    policy = model / "config" / args.precision
    if policy.parent != model / "config" or not policy.is_file():
        raise ValueError(
            "Precision must name a configuration file in the model config directory"
        )
    os.environ["QWEN_PRECISION_CONFIG"] = str(policy)
    sys.path.insert(0, str(source))
    contract = importlib.import_module(
        "models.demos.qwen38_27b_qb2.demo.galaxy_serving"
    )
    receipt_bytes = args.qualification.read_bytes()
    receipt = json.loads(receipt_bytes)
    contract.verify_qualified_source(receipt, model)
    verify_committed_source(source, model_sha, receipt["source_sha256"], args.precision)
    spec = runtime_spec(contract, receipt, model_sha, args.precision)
    output = args.output.resolve()
    output.mkdir()
    manifest = {
        "state": "prepared_unqualified",
        "base_image": BASE_IMAGE,
        "metal_sha": METAL_SHA,
        "model_sha": model_sha,
        "plugin_sha": PLUGIN_SHA,
        "checkpoint_sha": WEIGHTS_SHA,
        "g0_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "model_source_sha256": receipt["source_sha256"],
        "precision_file": args.precision,
        "precision_sha256": contract.precision_fingerprint(receipt["precision"]),
        "versions": {
            "vllm": "0.26.0",
            "torch": "2.11.0",
            "transformers": "5.12.1",
            "numpy": "1.26.4",
        },
    }
    for name, data in (("manifest.json", manifest), ("runtime-model-spec.json", spec)):
        (output / name).write_text(json.dumps(data, indent=2) + "\n")
    (output / "g0.json").write_bytes(receipt_bytes)
    (output / "values.yaml").write_text(
        yaml.safe_dump(helm_values(spec, args.weights_host_path), sort_keys=False)
    )
    verifier = REPO / "scripts/release/verify_qwen38_image.py"
    (output / "verify_image.py").write_bytes(verifier.read_bytes())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--precision", default="precision_accurate_decode.json")
    parser.add_argument("--weights-host-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
