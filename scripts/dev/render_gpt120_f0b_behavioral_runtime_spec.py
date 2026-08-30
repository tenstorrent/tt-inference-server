#!/usr/bin/env python3
"""Render a non-certifying GPT-OSS f0b runtime spec for a TTIS shadow run.

The release planner remains the only authority that can render an enrollable
catalogue row.  This helper deliberately accepts only a mutable, non-official
package root and preserves the normal ``impl=quetzal`` TTIS/vLLM contract so a
local run can reach (and record) the real package-trust rejection.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODEL_ID = "openai/gpt-oss-120b"
CHECKPOINT = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
QUETZAL_COMMIT = "071e23cd264d4b0df67a0d3df4642378663002c4"
TT_METAL_PATCHSET = (
    "22fb0bd2523b8a5c63fa20c3c8a1586dc9ead5150449d0eb02231fa8173a7edd"
)
AUXILIARY_NAME = "openai_gpt-oss-120b-streamed-cache"
CONTAINER_PACKAGE_PARENT = Path("/home/container_app_user/quetzal/packages")
CONTAINER_AUXILIARY_PARENT = Path("/home/container_app_user/quetzal/auxiliary")
OFFICIAL_PACKAGE_PREFIX = Path("/mnt/models/quetzal/immutable")
DESCRIPTOR_CONTAINER_PATH = (
    "/opt/quetzal/mesh_graph_descriptors/"
    "p150_x4_2ch_mesh_graph_descriptor.textproto"
)
EXPECTED_FILES = {
    "QUETZAL_PREFILL_GENERATED_PY": (
        "compiled/openai_gpt-oss-120b-s1024/full/prefill/generated.py"
    ),
    "QUETZAL_PREFILL_METADATA_JSON": (
        "compiled/openai_gpt-oss-120b-s1024/full/prefill/metadata.json"
    ),
    "QUETZAL_DECODE_GENERATED_PY": (
        "compiled/openai_gpt-oss-120b-s1024/full/decode/generated.py"
    ),
    "QUETZAL_DECODE_METADATA_JSON": (
        "compiled/openai_gpt-oss-120b-s1024/full/decode/metadata.json"
    ),
    "QUETZAL_WEIGHTS": (
        "compiled_weights/openai_gpt-oss-120b-s1024/full/weights.pt"
    ),
}


class ContractError(ValueError):
    """The requested behavioral spec could be mistaken for an official row."""


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_local_root(root: Path) -> Path:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ContractError(f"package root must be an existing absolute real directory: {root}")
    root = root.resolve()
    if _inside(root, OFFICIAL_PACKAGE_PREFIX):
        raise ContractError("behavioral helper refuses the official immutable namespace")
    if not re.fullmatch(r"[A-Za-z0-9._@+-]+", root.name):
        raise ContractError(f"package root basename is not path-safe: {root.name!r}")
    for relative in EXPECTED_FILES.values():
        member = root / relative
        if member.is_symlink() or not member.is_file():
            raise ContractError(f"exact f0b member is missing or symlinked: {member}")
    return root


def _validate_auxiliary_root(root: Path) -> Path:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ContractError(
            f"auxiliary root must be an existing absolute real directory: {root}"
        )
    root = root.resolve()
    if _inside(root, OFFICIAL_PACKAGE_PREFIX):
        raise ContractError("behavioral helper refuses the official immutable namespace")
    return root


def render(package_root: Path, auxiliary_root: Path, image: str) -> dict:
    """Return a normal serialized ModelSpec with an explicit local-only marker."""
    # Load the source dev YAML explicitly: callers such as pytest may already
    # have imported workflows.model_spec with the production catalogue.
    from workflows.model_spec import get_model_spec_map, load_templates_from_yaml

    package_root = _validate_local_root(package_root)
    auxiliary_root = _validate_auxiliary_root(auxiliary_root)
    if not image or any(character.isspace() for character in image):
        raise ContractError("behavioral image must be non-empty and unambiguous")
    catalog_path = REPO_ROOT / "workflows/model_specs/dev/llm.yaml"
    catalog = get_model_spec_map(load_templates_from_yaml(catalog_path))
    matches = [
        candidate
        for candidate in catalog.values()
        if candidate.model_name == "Qwen3.6-27B"
        and candidate.impl.impl_id == "quetzal"
        and candidate.device_type.to_string() == "P300X2"
    ]
    if len(matches) != 1:
        raise ContractError(
            f"expected one Quetzal P300X2 base in the dev catalogue, got {len(matches)}"
        )
    base = matches[0]
    spec = copy.deepcopy(base.get_serialized_dict())
    package_id = package_root.name
    container_root = CONTAINER_PACKAGE_PARENT / package_id
    container_auxiliary = (
        CONTAINER_AUXILIARY_PARENT / AUXILIARY_NAME / auxiliary_root.name
    )

    env = copy.deepcopy(spec["env_vars"])
    env.update(
        {
            "QUETZAL_MODEL": MODEL_ID,
            "QUETZAL_HF_REVISION": CHECKPOINT,
            "QUETZAL_REQUIRED_SOURCE_REVISION": QUETZAL_COMMIT,
            "QUETZAL_PACKAGE_ID": package_id,
            # The mutable candidate has no trusted root proof.  A syntactically
            # valid impossible digest lets the real TTIS validator report that
            # boundary without inventing an immutable publication identity.
            "QUETZAL_BUNDLE_MANIFEST_SHA256": "0" * 64,
            "QUETZAL_REQUIRED_TT_METAL_PATCHSET_SHA256": TT_METAL_PATCHSET,
            "QUETZAL_PACKAGE_ROOT": str(container_root),
            "QZ_MODELS_ROOT": str(container_root),
            "QZ_QUALIFICATION_MANIFEST": str(
                container_root / "qualification_manifest.yaml"
            ),
            "QUETZAL_AUXILIARY_ROOTS_JSON": json.dumps(
                {AUXILIARY_NAME: str(container_auxiliary)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "QUETZAL_REQUIRED_AUXILIARY_NAMES": AUXILIARY_NAME,
            "QUETZAL_REQUIRED_PREFILL_BUCKETS": "128,1024",
            "VLLM_PLUGINS": "quetzal_model_registry,tt",
            "TT_VLLM_BUILTIN_MODELS": "0",
            "TTQ_ROW_ALL_REDUCE_TOPOLOGY": "Ring",
            "TTQ_TUNED_ROW_ALL_REDUCE": "1",
            "TTQ_TUNED_ROW_ALL_REDUCE_LINKS": "2",
            "TT_MESH_GRAPH_DESC_PATH": DESCRIPTOR_CONTAINER_PATH,
        }
    )
    for name, relative in EXPECTED_FILES.items():
        env[name] = str(container_root / relative)

    spec.update(
        {
            "model_id": "id_quetzal_gpt-oss-120b_p300x2_local-shadow",
            "model_name": "gpt-oss-120b",
            "hf_model_repo": MODEL_ID,
            "hf_weights_repo": MODEL_ID,
            "param_count": 120,
            "has_builtin_warmup": False,
            # External runtime specs win outright in run.py; CLI overrides are
            # intentionally not applied. Carry the behavioral image in the
            # spec itself so Docker launch cannot degrade to `docker pull None`.
            "docker_image": image,
            "env_vars": env,
            "metadata": {
                "reasoning_parser_name": "openai_gptoss",
                "tool_call_parser_name": "openai",
                "local_shadow": True,
                "certification_eligible": False,
                "claim_boundary": (
                    "mutable f0b package trust discriminator; not a catalogue row, "
                    "publication, Models-CI run, or certification"
                ),
            },
        }
    )
    device = spec["device_model_spec"]
    device["max_concurrency"] = 1
    device["max_context"] = 8192
    device["default_impl"] = False
    device["env_vars"] = copy.deepcopy(env)
    device["vllm_args"].update(
        {
            "model": MODEL_ID,
            "max_model_len": 8192,
            "max_num_seqs": 1,
            "revision": CHECKPOINT,
            "tokenizer_revision": CHECKPOINT,
        }
    )
    return {
        "schema": "ttq.gpt120-f0b-behavioral-runtime-spec/v1",
        "official_models_ci": False,
        "certification_eligible": False,
        "package_trust_expected": "fail_closed",
        "mutable_package_root": str(package_root),
        "mutable_auxiliary_root": str(auxiliary_root),
        "behavioral_image": image,
        "runtime_model_spec": spec,
        # ModelSpec.from_json recognizes the normal combined TTIS envelope
        # only when both keys are present. run.py still builds its live runtime
        # config from the CLI, so this intentionally remains empty.
        "runtime_config": {},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--auxiliary-root", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--acknowledge-mutable-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.acknowledge_mutable_local_only:
        parser.error("--acknowledge-mutable-local-only is required")
    try:
        document = render(args.package_root, args.auxiliary_root, args.image)
    except ContractError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
