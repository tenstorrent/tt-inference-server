#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Promote exact release-scoped dev leaves into a leaf-granular prod catalog."""

import argparse
import json
import sys
import tempfile
from copy import deepcopy
from io import StringIO
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import YAMLError as RuamelYAMLError
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
from yaml import YAMLError as PyYAMLError

# Add repo root to path for direct script execution.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.release.model_spec_resolver import (  # noqa: E402
    LeafIdentity,
    collect_release_combos,
    load_dev_model_spec_sources,
    resolve_release_combos,
)
from workflows.model_spec import (  # noqa: E402
    MODEL_SPEC_CATALOG_FILES,
    get_model_spec_map,
    load_templates_from_yaml,
    model_spec_leaf_identity,
)
from workflows.workflow_types import DeviceTypes, InferenceEngine  # noqa: E402


REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CI_CONFIG = REPO_ROOT / ".github" / "workflows" / "models-ci-config.json"
DEFAULT_DEV_DIR = REPO_ROOT / "workflows" / "model_specs" / "dev"
DEFAULT_PROD_DIR = REPO_ROOT / "workflows" / "model_specs" / "prod"

_PIN_FIELDS = ("version", "tt_metal_commit", "vllm_commit")


def _round_trip_yaml() -> YAML:
    yaml = YAML(typ="rt")
    yaml.allow_duplicate_keys = False
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def _load_document(path: Path) -> CommentedMap:
    document = _round_trip_yaml().load(path.read_text())
    if not isinstance(document, CommentedMap) or not isinstance(
        document.get("templates"), CommentedSeq
    ):
        raise ValueError(f"Catalog {path} must contain a templates sequence")
    return document


def split_into_blocks(text: str):
    """Split catalog text into top-level template blocks and untouched filler."""
    lines = text.splitlines(keepends=True)
    segments = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("- ") or line.rstrip("\n") == "-":
            block = [line]
            index += 1
            while index < len(lines):
                if lines[index].startswith((" ", "\t")):
                    block.append(lines[index])
                    index += 1
                elif lines[index].strip() == "":
                    lookahead = index
                    while lookahead < len(lines) and lines[lookahead].strip() == "":
                        lookahead += 1
                    if lookahead < len(lines) and lines[lookahead].startswith(
                        (" ", "\t")
                    ):
                        block.extend(lines[index:lookahead])
                        index = lookahead
                    else:
                        break
                else:
                    break
            segments.append(("block", block))
        else:
            segments.append(("filler", [line]))
            index += 1
    return segments


def _parse_block(lines) -> CommentedMap:
    parsed = _round_trip_yaml().load("".join(lines))
    if (
        not isinstance(parsed, CommentedSeq)
        or len(parsed) != 1
        or not isinstance(parsed[0], CommentedMap)
    ):
        raise ValueError(f"Expected one YAML template block, got {''.join(lines)!r}")
    return parsed[0]


def _render_block(template: CommentedMap) -> list[str]:
    """Render one template using the repository's top-level list indentation."""
    stream = StringIO()
    _round_trip_yaml().dump(template, stream)
    lines = stream.getvalue().splitlines(keepends=True)
    if not lines:
        raise ValueError("Cannot render an empty template")
    rendered = [f"- {lines[0]}"] + [f"  {line}" for line in lines[1:]]
    return ["\n" if line.strip() == "" else line for line in rendered]


def _template_identity(
    template: CommentedMap,
    weight: str,
    device_spec: CommentedMap,
) -> LeafIdentity:
    try:
        device = DeviceTypes.from_string(str(device_spec["device"])).to_string()
        engine = InferenceEngine.from_string(str(template["inference_engine"])).value
        impl_id = str(template["impl"])
    except (AttributeError, KeyError, ValueError) as exc:
        raise ValueError(
            f"Invalid catalog leaf for weight {weight!r}: {template!r}"
        ) from exc
    return (str(weight), device, engine, impl_id)


def _filter_metadata(template: CommentedMap, weight: str) -> None:
    metadata = template.get("metadata")
    if not metadata:
        template.pop("metadata", None)
        return
    selected = metadata.get(weight)
    if selected is None:
        template.pop("metadata", None)
        return
    filtered = CommentedMap()
    filtered[weight] = deepcopy(selected)
    template["metadata"] = filtered


def _make_leaf(
    template: CommentedMap,
    weight_index: int,
    device_index: int,
) -> tuple[LeafIdentity, CommentedMap]:
    weights = template.get("weights")
    devices = template.get("device_model_specs")
    if not isinstance(weights, list) or not isinstance(devices, list):
        raise ValueError("Template must contain weights and device_model_specs lists")
    try:
        weight = str(weights[weight_index])
        device_spec = devices[device_index]
    except IndexError as exc:
        raise ValueError(
            f"Leaf indexes out of range: weight={weight_index}, device={device_index}"
        ) from exc
    if not isinstance(device_spec, CommentedMap):
        raise ValueError(f"Device model spec must be a mapping: {device_spec!r}")

    leaf = deepcopy(template)
    leaf["weights"] = CommentedSeq([deepcopy(weights[weight_index])])
    leaf["device_model_specs"] = CommentedSeq([deepcopy(device_spec)])
    _filter_metadata(leaf, weight)
    return _template_identity(leaf, weight, leaf["device_model_specs"][0]), leaf


def _flat_template_identity(template: CommentedMap) -> LeafIdentity:
    weights = template.get("weights")
    devices = template.get("device_model_specs")
    if not isinstance(weights, list) or len(weights) != 1:
        raise ValueError("Prod template must contain exactly one weight")
    if not isinstance(devices, list) or len(devices) != 1:
        raise ValueError("Prod template must contain exactly one device")
    return _template_identity(template, str(weights[0]), devices[0])


def _has_unsafe_yaml_structure(value) -> bool:
    anchor = getattr(value, "anchor", None)
    if anchor is not None and anchor.value is not None:
        return True
    if isinstance(value, CommentedMap):
        if getattr(value, "merge", None):
            return True
        return any(
            _has_unsafe_yaml_structure(key) or _has_unsafe_yaml_structure(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_has_unsafe_yaml_structure(item) for item in value)
    return False


def _inject_pins(
    leaf: CommentedMap,
    *,
    version: str,
    tt_metal_commit: str,
    vllm_commit: str | None,
) -> None:
    for field in _PIN_FIELDS:
        leaf.pop(field, None)

    keys = list(leaf)
    insert_at = (
        keys.index("device_model_specs") if "device_model_specs" in keys else len(keys)
    )
    pins = [
        ("version", version),
        ("tt_metal_commit", tt_metal_commit),
    ]
    if vllm_commit is not None:
        pins.append(("vllm_commit", vllm_commit))
    for offset, (key, value) in enumerate(pins):
        leaf.insert(
            insert_at + offset,
            key,
            DoubleQuotedScalarString(value),
        )


def _semantic_value(value):
    if isinstance(value, dict):
        return {str(key): _semantic_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_semantic_value(item) for item in value]
    return value


def _append_block(segments, lines) -> None:
    last_block_index = next(
        (
            index
            for index in range(len(segments) - 1, -1, -1)
            if segments[index][0] == "block"
        ),
        None,
    )
    segment = ("block", lines)
    if last_block_index is None:
        for index, (kind, filler_lines) in enumerate(segments):
            if kind == "filler" and any(
                line.strip() == "templates: []" for line in filler_lines
            ):
                segments[index] = (
                    "filler",
                    [
                        "templates:\n" if line.strip() == "templates: []" else line
                        for line in filler_lines
                    ],
                )
        segments.append(segment)
    else:
        if segments[last_block_index][1] and not segments[last_block_index][1][
            -1
        ].endswith("\n"):
            segments[last_block_index][1][-1] += "\n"
        segments.insert(last_block_index + 1, segment)


def _render_segments(segments) -> str:
    return "".join(line for _, lines in segments for line in lines)


def _validate_candidate_catalog(candidate_text: dict[str, str]):
    with tempfile.TemporaryDirectory() as temp_dir:
        prod_dir = Path(temp_dir) / "prod"
        prod_dir.mkdir()
        for filename in MODEL_SPEC_CATALOG_FILES:
            (prod_dir / filename).write_text(candidate_text[filename])
            _load_document(prod_dir / filename)
        templates = [
            template
            for filename in MODEL_SPEC_CATALOG_FILES
            for template in load_templates_from_yaml(prod_dir / filename, env="prod")
        ]
        return get_model_spec_map(templates)


def _identity_json(identity: LeafIdentity) -> list[str]:
    return list(identity)


def report_to_jsonable(report: dict) -> dict:
    """Convert the structured promotion report into deterministic JSON data."""
    return {
        "ok": True,
        "dry_run": report["dry_run"],
        "configured_combos": [
            {
                "model_name": combo.model_name,
                "engine": combo.engine.value,
                "device": combo.device.to_string(),
            }
            for combo in report["configured_combos"]
        ],
        "requested_identities": [
            _identity_json(identity) for identity in report["requested_identities"]
        ],
        "resolved": [
            {
                "identity": _identity_json(item.identity),
                "source_path": str(item.source_path),
                "template_index": item.template_index,
                "weight_index": item.weight_index,
                "device_index": item.device_index,
            }
            for item in report["resolved"]
        ],
        "actions": [
            {"identity": _identity_json(identity), "action": action}
            for identity, action in report["actions"].items()
        ],
        "added_identities": [
            _identity_json(identity) for identity in report["added_identities"]
        ],
        "updated_identities": [
            _identity_json(identity) for identity in report["updated_identities"]
        ],
        "unchanged_identities": [
            _identity_json(identity) for identity in report["unchanged_identities"]
        ],
        "retained_identities": [
            _identity_json(identity) for identity in report["retained_identities"]
        ],
        "changed_files": report["changed_files"],
        "leaf_count_before": report["leaf_count_before"],
        "leaf_count_after": report["leaf_count_after"],
    }


def _require_non_empty(name: str, value: str | None) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{name} must be a non-empty value")
    return value


def promote(
    ci_config_path,
    dev_dir,
    prod_dir,
    *,
    tt_metal_commit,
    version,
    vllm_commit=None,
    dry_run=False,
) -> dict:
    """Plan, validate, and optionally write exact leaf promotion."""
    version = _require_non_empty("version", version)
    tt_metal_commit = _require_non_empty("tt_metal_commit", tt_metal_commit)

    ci_config = json.loads(Path(ci_config_path).read_text())
    combos = collect_release_combos(ci_config)
    sources = load_dev_model_spec_sources(Path(dev_dir))
    resolved = resolve_release_combos(combos, sources)

    # resolve_release_combos() rejects two selectors that collide on one
    # identity, so this index keeps every resolved entry.
    resolved_by_identity = {item.identity: item for item in resolved}

    needs_vllm = any(
        item.combo.engine == InferenceEngine.VLLM
        for item in resolved_by_identity.values()
    )
    if needs_vllm:
        vllm_commit = _require_non_empty("vllm_commit", vllm_commit)

    raw_dev_documents = {}
    promoted_leaves = {}
    target_filenames = {}
    for identity, item in resolved_by_identity.items():
        document = raw_dev_documents.setdefault(
            item.source_path,
            _load_document(item.source_path),
        )
        try:
            raw_template = document["templates"][item.template_index]
        except IndexError as exc:
            raise ValueError(
                f"Source template index changed for release identity {identity!r}"
            ) from exc
        if _has_unsafe_yaml_structure(raw_template):
            raise ValueError(
                f"Source template for {identity!r} uses YAML anchors or merge keys"
            )
        actual_identity, leaf = _make_leaf(
            raw_template,
            item.weight_index,
            item.device_index,
        )
        if actual_identity != identity:
            raise ValueError(
                f"Source provenance mismatch: expected {identity!r}, "
                f"found {actual_identity!r}"
            )
        _inject_pins(
            leaf,
            version=version,
            tt_metal_commit=tt_metal_commit,
            vllm_commit=vllm_commit
            if item.combo.engine == InferenceEngine.VLLM
            else None,
        )
        promoted_leaves[identity] = leaf
        target_filenames[identity] = item.source_path.name

    prod_dir = Path(prod_dir)
    before_templates = [
        template
        for filename in MODEL_SPEC_CATALOG_FILES
        for template in load_templates_from_yaml(prod_dir / filename, env="prod")
    ]
    before_payloads = {
        model_spec_leaf_identity(spec): spec.get_serialized_dict()
        for spec in get_model_spec_map(before_templates).values()
    }
    original_text = {}
    candidate_segments = {}
    prod_locations = {}

    for filename in MODEL_SPEC_CATALOG_FILES:
        path = prod_dir / filename
        text = path.read_text()
        original_text[filename] = text
        segments = split_into_blocks(text)
        candidate_segments[filename] = [(kind, list(lines)) for kind, lines in segments]

        for segment_index, (kind, lines) in enumerate(segments):
            if kind != "block":
                continue
            raw_template = _parse_block(lines)
            identity = _flat_template_identity(raw_template)
            prod_locations[identity] = (filename, segment_index, raw_template)

    for identity, (filename, _, _) in prod_locations.items():
        if identity in target_filenames and filename != target_filenames[identity]:
            raise ValueError(
                f"Release identity {identity!r} belongs to {filename!r}, "
                f"but dev source is {target_filenames[identity]!r}"
            )

    for identity, leaf in promoted_leaves.items():
        location = prod_locations.get(identity)
        if location is not None:
            filename, segment_index, current = location
            if _semantic_value(current) != _semantic_value(leaf):
                candidate_segments[filename][segment_index] = (
                    "block",
                    _render_block(leaf),
                )
            continue
        filename = target_filenames[identity]
        if filename not in candidate_segments:
            raise ValueError(
                f"No prod catalog file {filename!r} for release identity {identity!r}"
            )
        _append_block(candidate_segments[filename], _render_block(leaf))

    candidate_text = {
        filename: _render_segments(segments)
        for filename, segments in candidate_segments.items()
    }

    requested = tuple(promoted_leaves)
    requested_set = set(requested)
    candidate_model_specs = _validate_candidate_catalog(candidate_text)
    after_payloads = {
        model_spec_leaf_identity(spec): spec.get_serialized_dict()
        for spec in candidate_model_specs.values()
    }
    after_set = set(after_payloads)
    if not requested_set <= after_set:
        raise ValueError(
            f"Candidate prod is missing requested identities "
            f"{sorted(requested_set - after_set)!r}"
        )

    # A release may only move the identities it asked for. Everything else in
    # prod must survive byte-identical, so a block-splitting or rendering defect
    # that damages a neighbouring leaf fails the promotion instead of shipping.
    for identity in sorted(after_set - requested_set):
        if identity not in before_payloads:
            raise ValueError(f"Promotion introduced unrequested identity {identity!r}")
        if before_payloads[identity] != after_payloads[identity]:
            raise ValueError(f"Promotion changed retained identity {identity!r}")
    dropped = sorted(set(before_payloads) - after_set)
    if dropped:
        raise ValueError(f"Promotion dropped existing identities {dropped!r}")

    actions = {}
    for identity in requested:
        if identity not in before_payloads:
            actions[identity] = "added"
        elif before_payloads[identity] != after_payloads[identity]:
            actions[identity] = "updated"
        else:
            actions[identity] = "unchanged"

    changed_files = [
        filename
        for filename in MODEL_SPEC_CATALOG_FILES
        if candidate_text[filename] != original_text[filename]
    ]
    if not dry_run:
        staged_paths = {}
        try:
            for filename in changed_files:
                staged_path = prod_dir / f".{filename}.promotion.tmp"
                staged_path.write_text(candidate_text[filename])
                staged_paths[filename] = staged_path
            for filename, staged_path in staged_paths.items():
                staged_path.replace(prod_dir / filename)
        finally:
            for staged_path in staged_paths.values():
                if staged_path.exists():
                    staged_path.unlink()

    return {
        "dry_run": dry_run,
        "configured_combos": combos,
        "resolved": tuple(resolved_by_identity.values()),
        "requested_identities": requested,
        "actions": actions,
        "added_identities": tuple(
            identity for identity, action in actions.items() if action == "added"
        ),
        "updated_identities": tuple(
            identity for identity, action in actions.items() if action == "updated"
        ),
        "unchanged_identities": tuple(
            identity for identity, action in actions.items() if action == "unchanged"
        ),
        "retained_identities": tuple(sorted(after_set - requested_set)),
        "changed_files": changed_files,
        "leaf_count_before": len(before_payloads),
        "leaf_count_after": len(after_payloads),
    }


def _print_human_report(report: dict) -> None:
    mode = "DRY RUN" if report["dry_run"] else "APPLIED"
    print(f"{mode}: {len(report['configured_combos'])} configured release combos")
    for item in report["resolved"]:
        print(
            f"RESOLVED  {item.combo.model_name} [{item.combo.engine.value}] "
            f"on {item.combo.device.to_string()} -> {item.identity} "
            f"({item.source_path}:{item.template_index}/"
            f"{item.weight_index}/{item.device_index})"
        )
    for identity, action in report["actions"].items():
        print(f"{action.upper():9} {identity}")
    print(f"RETAINED  {len(report['retained_identities'])} existing identities")
    print(f"Prod leaves: {report['leaf_count_before']} -> {report['leaf_count_after']}")
    print(
        f"{len(report['changed_files'])} prod file(s) changed: "
        f"{report['changed_files']}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote exact dev leaves into a leaf-granular prod catalog."
    )
    parser.add_argument("--ci-config", type=Path, default=DEFAULT_CI_CONFIG)
    parser.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_DIR)
    parser.add_argument("--prod-dir", type=Path, default=DEFAULT_PROD_DIR)
    parser.add_argument("--tt-metal-commit", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--vllm-commit", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    try:
        report = promote(
            args.ci_config,
            args.dev_dir,
            args.prod_dir,
            tt_metal_commit=args.tt_metal_commit,
            version=args.version,
            vllm_commit=args.vllm_commit,
            dry_run=args.dry_run,
        )
    except (OSError, PyYAMLError, RuamelYAMLError, ValueError) as exc:
        if args.json_output:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                    indent=2,
                )
            )
            return 2
        parser.error(str(exc))

    if args.json_output:
        print(json.dumps(report_to_jsonable(report), indent=2))
    else:
        _print_human_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
