#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""
Promote dev model specs to prod for every model-device combination marked
``release`` in models-ci-config.json.

For each (model, inference_engine, device) under ``ci.release`` in the CI config,
the matching template in workflows/model_specs/dev/ is copied into the same-named
file in workflows/model_specs/prod/, upserting by
(impl, inference_engine, weights, devices) identity.

The catalogue YAML files are hand-authored with inconsistent block-sequence
indentation (top-level list items at column 0, nested lists indented to column 4),
which no single ruamel.yaml indent setting can reproduce. Round-tripping a whole
file therefore reformats every untouched template. To preserve formatting exactly,
this tool splices template TEXT blocks: each catalogue file is segmented into
top-level ``- ...`` template blocks and the filler between them; only the specific
blocks that change are replaced/appended, so untouched templates stay byte-identical.
"""

import argparse
import json
import re
import sys
from collections import namedtuple
from pathlib import Path

import yaml

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from workflows.workflow_types import DeviceTypes, InferenceEngine  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CI_CONFIG = REPO_ROOT / ".github" / "workflows" / "models-ci-config.json"
DEFAULT_DEV_DIR = REPO_ROOT / "workflows" / "model_specs" / "dev"
DEFAULT_PROD_DIR = REPO_ROOT / "workflows" / "model_specs" / "prod"

ReleaseCombo = namedtuple("ReleaseCombo", ["model_name", "engine", "device"])

# A matched dev template: its upsert identity, parsed dict, and exact source lines.
MatchedBlock = namedtuple("MatchedBlock", ["identity", "template", "lines"])

# A top-level template item starts with "- " (or a bare "-") at column 0.
_TEMPLATE_ITEM_RE = re.compile(r"^-(\s|$)")


def model_name_from_weight(weight: str) -> str:
    """Extract the model name (basename) from a HuggingFace repo path."""
    return Path(weight).name


def iter_implementations(model_entry: dict):
    """Yield each implementation dict for a CI-config model entry.

    Handles both the flat shape ({inference_engine, ci}) and the
    implementations:[...] array shape.
    """
    if "implementations" in model_entry:
        yield from model_entry["implementations"]
    else:
        yield model_entry


def collect_release_combos(ci_config: dict) -> set:
    """Return the set of ReleaseCombo(model_name, engine, device) marked release."""
    combos = set()
    for model_name, entry in ci_config.get("models", {}).items():
        for impl in iter_implementations(entry):
            release = impl.get("ci", {}).get("release")
            if not release:
                continue
            engine = InferenceEngine.from_string(impl["inference_engine"])
            for device in release.get("devices", []):
                combos.add(
                    ReleaseCombo(model_name, engine, DeviceTypes.from_string(device))
                )
    return combos


def template_engine(template: dict) -> InferenceEngine:
    return InferenceEngine.from_string(template["inference_engine"])


def template_devices(template: dict) -> set:
    return {
        DeviceTypes.from_string(d["device"])
        for d in template.get("device_model_specs", [])
    }


def template_model_names(template: dict) -> set:
    return {model_name_from_weight(w) for w in template.get("weights", [])}


def template_matches(template: dict, combo: ReleaseCombo) -> bool:
    """True if the template provides the given release combo."""
    return (
        combo.model_name in template_model_names(template)
        and combo.engine == template_engine(template)
        and combo.device in template_devices(template)
    )


def template_identity(template: dict):
    """Upsert identity for a template block: (impl, engine, weights, devices).

    The device set is part of the identity because the catalogue intentionally
    holds multiple blocks per (impl, engine, weights) — one per device group
    (e.g. the same model validated on different hardware at different commits).
    Without devices, distinct blocks would collide and overwrite each other.
    """
    return (
        template["impl"],
        template_engine(template),
        frozenset(template.get("weights", [])),
        frozenset(template_devices(template)),
    )


def split_into_blocks(text: str):
    """Segment catalogue text into ("block", lines) and ("filler", lines) parts.

    A "block" is a top-level ``- ...`` template item: its first line plus all
    following indented body lines. "filler" is everything else (the ``templates:``
    header, column-0 comment banners, blank lines). Concatenating every segment's
    lines reproduces ``text`` exactly.
    """
    lines = text.splitlines(keepends=True)
    segments = []
    i = 0
    while i < len(lines):
        if _TEMPLATE_ITEM_RE.match(lines[i].rstrip("\n")):
            block = [lines[i]]
            i += 1
            # Body lines are indented. A blank line is interior to the block only
            # if an indented line follows it (after any run of blanks); otherwise
            # it separates this block from the next template and stays filler.
            while i < len(lines):
                if lines[i].startswith((" ", "\t")):
                    block.append(lines[i])
                    i += 1
                elif lines[i].strip() == "":
                    j = i
                    while j < len(lines) and lines[j].strip() == "":
                        j += 1
                    if j < len(lines) and lines[j].startswith((" ", "\t")):
                        block.extend(lines[i:j])
                        i = j
                    else:
                        break
                else:
                    break
            segments.append(("block", block))
        else:
            segments.append(("filler", [lines[i]]))
            i += 1
    return segments


def parse_block(block_lines) -> dict:
    """Parse a template block's text into a plain dict."""
    parsed = yaml.safe_load("".join(block_lines))
    assert isinstance(parsed, list) and len(parsed) == 1, (
        f"expected a single-item YAML list, got: {''.join(block_lines)!r}"
    )
    return parsed[0]


def find_matches(dev_dir: Path, combos: set):
    """Scan dev catalogue files for templates matching any release combo.

    Returns (matches_by_file, unmatched):
      - matches_by_file: dict filename -> list of MatchedBlock, in file order,
        de-duplicated by identity.
      - unmatched: set of combos that matched no dev template.
    """
    matched_combos = set()
    matches_by_file = {}
    for dev_file in sorted(Path(dev_dir).glob("*.yaml")):
        picked = []
        picked_ids = set()
        for kind, lines in split_into_blocks(dev_file.read_text()):
            if kind != "block":
                continue
            template = parse_block(lines)
            hits = [c for c in combos if template_matches(template, c)]
            if not hits:
                continue
            matched_combos.update(hits)
            identity = template_identity(template)
            if identity not in picked_ids:
                picked.append(MatchedBlock(identity, template, lines))
                picked_ids.add(identity)
        if picked:
            matches_by_file[dev_file.name] = picked
    unmatched = combos - matched_combos
    return matches_by_file, unmatched


def upsert_block(segments, identity, lines) -> str:
    """Replace the block segment with matching identity, else append a new one.

    Operates in place on the ``segments`` list from split_into_blocks. Returns
    "updated" if an existing same-identity block was replaced, else "appended".
    """
    for idx, (kind, seg_lines) in enumerate(segments):
        if kind == "block" and template_identity(parse_block(seg_lines)) == identity:
            segments[idx] = ("block", list(lines))
            return "updated"

    last_block_idx = None
    for idx, (kind, _) in enumerate(segments):
        if kind == "block":
            last_block_idx = idx
    new_segment = ("block", _ensure_trailing_newline(list(lines)))
    if last_block_idx is None:
        segments.append(new_segment)
    else:
        # Ensure the block we insert after ends in a newline so the two don't
        # fuse onto one line.
        _ensure_trailing_newline(segments[last_block_idx][1])
        segments.insert(last_block_idx + 1, new_segment)
    return "appended"


def _ensure_trailing_newline(lines):
    """Guarantee the block's last line ends with a newline (safe to append after)."""
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def _render(segments) -> str:
    return "".join(line for _, seg_lines in segments for line in seg_lines)


def promote(ci_config_path, dev_dir, prod_dir, dry_run=False) -> dict:
    """Promote release-marked dev templates into prod.

    Returns a report dict:
      - combos: set of all release combos
      - matches_by_file: dict filename -> list of MatchedBlock
      - unmatched: set of combos with no dev template
      - actions: dict filename -> list of (identity, "appended"|"updated")
      - changed_files: list of prod filenames whose content would change
    """
    ci_config = json.loads(Path(ci_config_path).read_text())
    combos = collect_release_combos(ci_config)
    matches_by_file, unmatched = find_matches(Path(dev_dir), combos)

    actions = {}
    changed_files = []
    for filename, matched in matches_by_file.items():
        prod_file = Path(prod_dir) / filename
        original = prod_file.read_text() if prod_file.exists() else ""
        segments = split_into_blocks(original)

        file_actions = []
        for block in matched:
            action = upsert_block(segments, block.identity, block.lines)
            file_actions.append((block.identity, action))
        actions[filename] = file_actions

        new_text = _render(segments)
        if new_text != original:
            changed_files.append(filename)
            if not dry_run:
                prod_file.parent.mkdir(parents=True, exist_ok=True)
                prod_file.write_text(new_text)

    return {
        "combos": combos,
        "matches_by_file": matches_by_file,
        "unmatched": unmatched,
        "actions": actions,
        "changed_files": changed_files,
    }


def _combo_str(combo: ReleaseCombo) -> str:
    return f"{combo.model_name} [{combo.engine.name}] on {combo.device.name}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote release-marked dev model specs into the prod catalogue."
    )
    parser.add_argument("--ci-config", type=Path, default=DEFAULT_CI_CONFIG)
    parser.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_DIR)
    parser.add_argument("--prod-dir", type=Path, default=DEFAULT_PROD_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report intended changes without writing any files.",
    )
    args = parser.parse_args(argv)

    report = promote(args.ci_config, args.dev_dir, args.prod_dir, dry_run=args.dry_run)

    prefix = "[dry-run] " if args.dry_run else ""
    for filename, file_actions in sorted(report["actions"].items()):
        for identity, action in file_actions:
            impl, engine, weights, devices = identity
            print(
                f"{prefix}{action.upper():8} {filename}: "
                f"{impl} [{engine.name}] {sorted(weights)} "
                f"on {sorted(d.name for d in devices)}"
            )
    changed = report["changed_files"]
    print(f"{prefix}{len(changed)} prod file(s) changed: {sorted(changed)}")

    for combo in sorted(
        report["unmatched"],
        key=lambda c: (c.model_name, c.engine.name, c.device.name),
    ):
        print(f"WARNING: no dev template found for {_combo_str(combo)}")

    return 1 if report["unmatched"] else 0


if __name__ == "__main__":
    sys.exit(main())
