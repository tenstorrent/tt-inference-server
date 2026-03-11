#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Prepare the next development state on `main` after a release.

This helper:
- reads `release_logs/v{released_version}/pre_release_models_diff.json`
- increments `VERSION`
- applies released template updates back onto `main` when `main` still matches
  the released starting values
- always stamps matching templates with `release_version=released_version`
- regenerates `default_model_spec.json` and model support docs
- writes `release_logs/post_release_pr.md`
"""

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, cast

try:
    from release_diff import (
        AppliedRecordSummary,
        PostReleaseSummary,
        ReleaseDiffRecord,
        build_snapshot_index_by_template_key,
        format_template_identity_label,
        validate_record_template_match,
    )
    from release_paths import get_versioned_release_logs_dir
    from update_model_spec import (
        build_template_snapshots,
        extract_template_block_spans,
        regenerate_model_support_docs_and_update_readme,
        reload_and_export_model_specs_json,
        update_template_fields,
        update_template_optional_string_field,
    )
except ImportError:
    from scripts.release.release_diff import (
        AppliedRecordSummary,
        PostReleaseSummary,
        ReleaseDiffRecord,
        build_snapshot_index_by_template_key,
        format_template_identity_label,
        validate_record_template_match,
    )
    from scripts.release.release_paths import get_versioned_release_logs_dir
    from scripts.release.update_model_spec import (
        build_template_snapshots,
        extract_template_block_spans,
        regenerate_model_support_docs_and_update_readme,
        reload_and_export_model_specs_json,
        update_template_fields,
        update_template_optional_string_field,
    )


FIELD_SPECS = (
    ("tt_metal_commit", "tt_metal_commit_before", "tt_metal_commit_after"),
    ("vllm_commit", "vllm_commit_before", "vllm_commit_after"),
    ("status", "status_before", "status_after"),
)


@dataclass(frozen=True)
class PreparedPostReleaseUpdates:
    released_version: str
    next_version: str
    diff_json_path: Path
    diff_records: List[ReleaseDiffRecord]
    current_content: str
    updated_content: str
    summary: PostReleaseSummary
    pr_markdown: str


def write_text_atomically(path: Path, content: str) -> None:
    """Write text via a temporary file and replace the target atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_path_string = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent), text=True
    )
    temp_path = Path(temp_path_string)

    try:
        with os.fdopen(file_descriptor, "w") as temp_file:
            temp_file.write(content)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def read_version(version_file: Path) -> str:
    """Read the current semantic version from the VERSION file."""
    version = version_file.read_text().strip()
    if not version:
        raise ValueError(f"VERSION file is empty: {version_file}")
    return version


def write_version(version_file: Path, version: str) -> None:
    """Write the next semantic version to the VERSION file."""
    write_text_atomically(version_file, f"{version}\n")


def increment_version(version: str, increment: str) -> str:
    """Increment a semantic version by major, minor, or patch."""
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise ValueError(f"Invalid semantic version: {version}")

    major, minor, patch = (int(part) for part in match.groups())
    if increment == "major":
        return f"{major + 1}.0.0"
    if increment == "minor":
        return f"{major}.{minor + 1}.0"
    if increment == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Unsupported increment: {increment}")


def load_release_diff_records(diff_json_path: Path) -> List[ReleaseDiffRecord]:
    """Load the pre-release diff JSON that drives post-release updates."""
    if not diff_json_path.exists():
        raise FileNotFoundError(f"Pre-release diff JSON not found: {diff_json_path}")
    with diff_json_path.open("r") as file:
        records = cast(List[ReleaseDiffRecord], json.load(file))

    for record in records:
        if "template_key" not in record or "inference_engine" not in record:
            raise ValueError(
                "Release diff JSON is missing `template_key` or `inference_engine`. "
                "Regenerate it with update_model_spec.py before running post_release.py."
            )

    return records


def format_record_label(record: ReleaseDiffRecord) -> str:
    """Format a compact label for logs and PR notes."""
    return format_template_identity_label(
        record["impl_id"], record["weights"], record["devices"]
    )


def format_status_value(status_name: Optional[str]) -> Optional[str]:
    """Convert a status enum member name into source text form."""
    if not status_name:
        return None
    return f"ModelStatusTypes.{status_name}"


def extract_release_version_value(template_text: str) -> Optional[str]:
    """Extract the current release_version value from template source text."""
    match = re.search(r'release_version="([^"]*)"', template_text)
    if match:
        return match.group(1)
    return None


def apply_record_to_template(
    record: ReleaseDiffRecord, snapshot, released_version: str
) -> Tuple[str, AppliedRecordSummary]:
    """Apply one release-diff record onto a matching template snapshot."""
    template_text = snapshot["template_text"]
    applied_fields = []
    discarded_fields = []
    release_version_updated = False

    tt_metal_commit_to_apply = None
    vllm_commit_to_apply = None
    status_to_apply = None
    apply_vllm_optional_update = False
    apply_release_version = False

    for field_name, before_key, after_key in FIELD_SPECS:
        before_value = record.get(before_key)
        after_value = record.get(after_key)
        current_value = snapshot.get(field_name)

        if before_value == after_value:
            continue

        if current_value != before_value:
            discarded_fields.append(
                {
                    "field": field_name,
                    "expected": before_value,
                    "current": current_value,
                    "released": after_value,
                }
            )
            continue

        applied_fields.append(field_name)
        if field_name == "tt_metal_commit":
            tt_metal_commit_to_apply = after_value
            apply_release_version = True
        elif field_name == "vllm_commit":
            vllm_commit_to_apply = after_value
            apply_vllm_optional_update = True
        else:
            status_to_apply = format_status_value(after_value)

    updated_template = update_template_fields(
        template_text,
        tt_metal_commit_to_apply,
        vllm_commit_to_apply,
        status_to_apply,
        release_version=released_version if apply_release_version else None,
    )

    if apply_vllm_optional_update:
        updated_template = update_template_optional_string_field(
            updated_template, "vllm_commit", vllm_commit_to_apply
        )

    release_version_updated = extract_release_version_value(
        template_text
    ) != extract_release_version_value(updated_template)

    result = {
        "label": format_record_label(record),
        "applied_fields": applied_fields,
        "discarded_fields": discarded_fields,
        "release_version_updated": release_version_updated,
        "changed": updated_template != template_text,
    }
    return updated_template, result


def build_updated_model_spec_content(
    model_spec_path: Path,
    current_content: str,
    diff_records: Sequence[ReleaseDiffRecord],
    released_version: str,
) -> Tuple[str, PostReleaseSummary]:
    """Build updated model_spec.py content and a detailed PR/report summary."""
    snapshots = build_template_snapshots(
        model_spec_path, current_content, "model_spec_post_release_apply"
    )
    spans = extract_template_block_spans(current_content)
    if len(spans) != len(snapshots):
        raise ValueError("Template span count does not match template snapshots")

    build_snapshot_index_by_template_key(
        snapshots, "applying post-release diff records on main"
    )
    snapshots_by_template_key = {
        snapshot["template_key"]: (index, snapshot)
        for index, snapshot in enumerate(snapshots)
    }

    updated_blocks = {}
    processed_template_keys = set()
    applied_records = []
    skipped_records = []

    for record in diff_records:
        label = format_record_label(record)
        snapshot_match = snapshots_by_template_key.get(record["template_key"])
        if snapshot_match is None:
            skipped_records.append(
                {"label": label, "reason": "No matching template found on main."}
            )
            continue

        template_index, snapshot = snapshot_match
        validate_record_template_match(record, snapshot)
        if record["template_key"] in processed_template_keys:
            skipped_records.append(
                {
                    "label": label,
                    "reason": "Duplicate release diff record matched the same template.",
                }
            )
            continue

        updated_template, result = apply_record_to_template(
            record, snapshot, released_version
        )
        processed_template_keys.add(record["template_key"])
        updated_blocks[template_index] = updated_template
        applied_records.append(result)

    updated_content = current_content
    if updated_blocks:
        parts = []
        last_pos = 0
        for index, (start_pos, end_pos, template_text) in enumerate(spans):
            parts.append(current_content[last_pos:start_pos])
            parts.append(updated_blocks.get(index, template_text))
            last_pos = end_pos
        parts.append(current_content[last_pos:])
        updated_content = "".join(parts)

    summary: PostReleaseSummary = {
        "matched_records": len(applied_records),
        "updated_templates": sum(1 for record in applied_records if record["changed"]),
        "applied_records": applied_records,
        "skipped_records": skipped_records,
    }
    return updated_content, summary


def build_post_release_pr_markdown(
    released_version: str, next_version: str, summary: PostReleaseSummary
) -> str:
    """Render the markdown body used for the post-release PR."""
    lines = [
        "# Post-release PR",
        "",
        "## Summary",
        "",
        f"- Bumped `VERSION` from `{released_version}` to `{next_version}`.",
        (
            f"- Processed `{summary['matched_records']}` matching release-diff records and "
            f"changed `{summary['updated_templates']}` templates in `workflows/model_spec.py`."
        ),
        "- Regenerated `default_model_spec.json`, `docs/model_support/`, and `README.md`.",
        "",
        "## Applied template updates",
        "",
    ]

    applied_records = summary["applied_records"]
    if applied_records:
        for record in applied_records:
            details = []
            if record["applied_fields"]:
                details.append(
                    "applied fields: "
                    + ", ".join(f"`{field}`" for field in record["applied_fields"])
                )
            if record["discarded_fields"]:
                details.append(
                    "discarded fields: "
                    + ", ".join(
                        f"`{field['field']}`" for field in record["discarded_fields"]
                    )
                )
            if record["release_version_updated"]:
                details.append("updated `release_version`")
            if not details:
                details.append("no text change required")
            lines.append(f"- `{record['label']}`: " + "; ".join(details) + ".")
    else:
        lines.append("- No matching release-diff records were applied.")

    lines.extend(["", "## Discarded field updates", ""])

    discarded_details = []
    for record in applied_records:
        for field in record["discarded_fields"]:
            discarded_details.append((record["label"], field))

    if discarded_details:
        for label, field in discarded_details:
            lines.append(
                f"- `{label}`: discarded `{field['field']}` because `main` has "
                f"`{field['current']}` instead of released start `{field['expected']}`."
            )
    else:
        lines.append("- No field updates were discarded.")

    lines.extend(["", "## Skipped records", ""])
    skipped_records = summary["skipped_records"]
    if skipped_records:
        for record in skipped_records:
            lines.append(f"- `{record['label']}`: {record['reason']}")
    else:
        lines.append("- No records were skipped.")

    lines.extend(
        [
            "",
            "## Regenerated artifacts",
            "",
            "- `workflows/model_spec.py`",
            "- `default_model_spec.json`",
            "- `docs/model_support/`",
            "- `README.md`",
            "- `release_logs/post_release_pr.md`",
            "- `VERSION`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def prepare_post_release_updates(
    version_file: Path,
    model_spec_path: Path,
    increment: str,
    diff_json_path: Optional[Path] = None,
) -> PreparedPostReleaseUpdates:
    """Load all required inputs and build the post-release outputs in memory."""
    released_version = read_version(version_file)
    next_version = increment_version(released_version, increment)
    resolved_diff_json_path = (
        diff_json_path
        if diff_json_path is not None
        else get_versioned_release_logs_dir(released_version)
        / "pre_release_models_diff.json"
    )

    diff_records = load_release_diff_records(resolved_diff_json_path)
    current_content = model_spec_path.read_text()
    updated_content, summary = build_updated_model_spec_content(
        model_spec_path,
        current_content,
        diff_records,
        released_version,
    )
    pr_markdown = build_post_release_pr_markdown(
        released_version,
        next_version,
        summary,
    )
    return PreparedPostReleaseUpdates(
        released_version=released_version,
        next_version=next_version,
        diff_json_path=resolved_diff_json_path,
        diff_records=diff_records,
        current_content=current_content,
        updated_content=updated_content,
        summary=summary,
        pr_markdown=pr_markdown,
    )


def apply_post_release_updates(
    prepared_updates: PreparedPostReleaseUpdates,
    version_file: Path,
    model_spec_path: Path,
    default_model_spec_path: Path,
    pr_output_path: Path,
) -> None:
    """Write prepared post-release updates with VERSION committed last."""
    if prepared_updates.updated_content != prepared_updates.current_content:
        write_text_atomically(model_spec_path, prepared_updates.updated_content)

    reload_and_export_model_specs_json(model_spec_path, default_model_spec_path)
    regenerate_model_support_docs_and_update_readme(model_spec_path)
    write_text_atomically(pr_output_path, prepared_updates.pr_markdown)
    write_version(version_file, prepared_updates.next_version)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare next development state after a release"
    )
    parser.add_argument(
        "--increment",
        required=True,
        choices=("major", "minor", "patch"),
        help="Semantic version component to increment for the next development cycle.",
    )
    parser.add_argument(
        "--version-file",
        default="VERSION",
        help="Path to VERSION file (default: VERSION).",
    )
    parser.add_argument(
        "--model-spec-path",
        default="workflows/model_spec.py",
        help="Path to model_spec.py (default: workflows/model_spec.py).",
    )
    parser.add_argument(
        "--diff-json",
        help=(
            "Path to pre_release_models_diff.json. Defaults to "
            "release_logs/v{released_version}/pre_release_models_diff.json."
        ),
    )
    parser.add_argument(
        "--default-model-spec-path",
        default="default_model_spec.json",
        help="Path to regenerated default_model_spec.json (default: default_model_spec.json).",
    )
    parser.add_argument(
        "--pr-output",
        default="release_logs/post_release_pr.md",
        help="Path to the generated post-release PR markdown.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for post-release preparation."""
    args = parse_args()

    version_file = Path(args.version_file)
    model_spec_path = Path(args.model_spec_path)
    default_model_spec_path = Path(args.default_model_spec_path)
    pr_output_path = Path(args.pr_output)

    prepared_updates = prepare_post_release_updates(
        version_file=version_file,
        model_spec_path=model_spec_path,
        increment=args.increment,
        diff_json_path=Path(args.diff_json) if args.diff_json else None,
    )

    print(
        "Prepared post-release updates from "
        f"{prepared_updates.released_version} to {prepared_updates.next_version}: "
        f"{prepared_updates.summary['updated_templates']} template changes, "
        f"{len(prepared_updates.summary['skipped_records'])} skipped records."
    )

    if args.dry_run:
        print("\n[DRY RUN] No files were written.")
        print(f"[DRY RUN] Would update VERSION file: {version_file}")
        print(f"[DRY RUN] Would update model spec: {model_spec_path}")
        print(
            f"[DRY RUN] Would regenerate default model spec: {default_model_spec_path}"
        )
        print("[DRY RUN] Would regenerate docs/model_support/ and update README.md")
        print(f"[DRY RUN] Would write PR markdown: {pr_output_path}")
        return 0

    apply_post_release_updates(
        prepared_updates,
        version_file,
        model_spec_path,
        default_model_spec_path,
        pr_output_path,
    )
    print(f"Wrote post-release PR draft to {pr_output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
