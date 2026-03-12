#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Generate GitHub release notes from structured release artifacts."""

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from release_diff import ReleaseDiffRecord
    from release_paths import get_versioned_release_logs_dir
    from release_performance import (
        RELEASE_PERFORMANCE_SCHEMA_VERSION,
        get_release_performance_path,
        iter_release_performance_entries,
        normalize_device_name,
        normalize_inference_engine_name,
    )
except ImportError:
    from scripts.release.release_diff import ReleaseDiffRecord
    from scripts.release.release_paths import get_versioned_release_logs_dir
    from scripts.release.release_performance import (
        RELEASE_PERFORMANCE_SCHEMA_VERSION,
        get_release_performance_path,
        iter_release_performance_entries,
        normalize_device_name,
        normalize_inference_engine_name,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EMPTY_RELEASE_PERFORMANCE_DATA = {
    "schema_version": RELEASE_PERFORMANCE_SCHEMA_VERSION,
    "models": {},
}
_AMBIGUOUS_MATCH = object()

SECTION_TITLES = {
    "summary_of_changes": "Summary of Changes",
    "recommended_system_software_versions": "Recommended system software versions",
    "known_issues": "Known Issues",
    "model_hardware_support_diff": "Model and Hardware Support Diff",
    "scale_out": "Scale Out",
    "deprecations_breaking_changes": "Deprecations and breaking changes",
    "release_artifacts_summary": "Release Artifacts Summary",
    "contributors": "Contributors",
    "assets": "Assets",
}

PerformanceEntry = Dict[str, Any]
PerformanceFullKey = Tuple[str, str, str, str]
PerformancePartialKey = Tuple[str, str, str]


def read_version(version_file: Path) -> str:
    """Read VERSION file and return a trimmed semantic version string."""
    version = version_file.read_text().strip()
    if not version:
        raise ValueError(f"VERSION file is empty: {version_file}")
    return version


def _default_value_copy(default: Any) -> Any:
    return deepcopy(default)


def load_optional_json(path: Optional[Path], default: Any) -> Any:
    """Return decoded JSON if present, otherwise a copy of the default value."""
    if not path or not path.exists():
        return _default_value_copy(default)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_git_file_text(path: Path, ref: str = "HEAD") -> Optional[str]:
    """Read a tracked file directly from git, returning None when unavailable."""
    resolved_path = path.resolve()
    try:
        relative_path = resolved_path.relative_to(PROJECT_ROOT).as_posix()
        repo_root = PROJECT_ROOT
    except ValueError:
        relative_path = resolved_path.name
        repo_root = resolved_path.parent

    result = subprocess.run(
        ["git", "show", f"{ref}:{relative_path}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def load_git_release_performance_data(
    path: Optional[Path] = None, ref: str = "HEAD"
) -> Dict[str, Any]:
    """Load the git-base release-performance baseline when it exists."""
    data_path = path or get_release_performance_path()
    git_text = _load_git_file_text(data_path, ref=ref)
    if not git_text:
        return _default_value_copy(EMPTY_RELEASE_PERFORMANCE_DATA)
    data = json.loads(git_text)
    if "models" not in data or not isinstance(data["models"], dict):
        return _default_value_copy(EMPTY_RELEASE_PERFORMANCE_DATA)
    return data


def render_section(title: str, body: str) -> str:
    """Render a markdown section, leaving the body blank when no content exists."""
    if body.strip():
        return f"## {title}\n\n{body.strip()}"
    return f"## {title}\n"


def normalize_status_value(status_value: Optional[str]) -> Optional[str]:
    """Normalize `ModelStatusTypes.*` strings to their enum member names."""
    if not status_value:
        return None
    if status_value.startswith("ModelStatusTypes."):
        return status_value.split(".", 1)[1]
    return status_value


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _format_commit_change(before: Optional[str], after: Optional[str]) -> str:
    if before and after:
        if before != after:
            return f"`{before}` -> `{after}`"
        return f"`{after}`"
    if after:
        return f"New: `{after}`"
    if before:
        return f"Removed: `{before}`"
    return "N/A"


def _format_status_change(
    before: Optional[str], after: Optional[str], unchanged_label: str = "No change"
) -> str:
    normalized_before = normalize_status_value(before)
    normalized_after = normalize_status_value(after)
    if normalized_before and normalized_after:
        if normalized_before != normalized_after:
            return f"{normalized_before} -> {normalized_after}"
        return f"{normalized_after} (no change)"
    if normalized_after:
        return f"New: {normalized_after}"
    if normalized_before:
        return f"Removed: {normalized_before}"
    return unchanged_label


def _format_ci_link(record: ReleaseDiffRecord) -> str:
    ci_job_url = record.get("ci_job_url")
    ci_run_number = record.get("ci_run_number")
    if ci_job_url and ci_run_number:
        return f"[Run {ci_run_number}]({ci_job_url})"
    if ci_job_url:
        return f"[View Job]({ci_job_url})"
    return "N/A"


def load_release_diff_records(path: Optional[Path]) -> List[ReleaseDiffRecord]:
    """Load structured release diff records from disk."""
    records = load_optional_json(path, default=[])
    if not isinstance(records, list):
        return []
    return records


def _render_release_artifact_list(
    title: str,
    items: Sequence[str],
    empty_message: str,
    description: str = "",
) -> str:
    lines = [f"### {title}", ""]
    if description:
        lines.extend([description, ""])
    if items:
        for item in items:
            lines.append(f"- {item.replace('ghcr.io/', 'https://ghcr.io/')}")
        lines.extend(["", f"**Total:** {len(items)}"])
    else:
        lines.append(empty_message)
    return "\n".join(lines)


def _render_release_artifact_map(
    title: str,
    mapping: Dict[str, str],
    empty_message: str,
    source_label: str,
    description: str = "",
) -> str:
    lines = [f"### {title}", ""]
    if description:
        lines.extend([description, ""])
    if mapping:
        for destination, source in sorted(mapping.items()):
            lines.append(f"- {destination.replace('ghcr.io/', 'https://ghcr.io/')}")
            lines.append(
                f"  - {source_label}: {source.replace('ghcr.io/', 'https://ghcr.io/')}"
            )
            lines.append("")
        lines.append(f"**Total:** {len(mapping)}")
    else:
        lines.append(empty_message)
    return "\n".join(lines).rstrip()


def render_release_artifacts_summary(summary_data: Dict[str, Any]) -> str:
    """Render the release artifact summary from the structured JSON output."""
    if not isinstance(summary_data, dict):
        return ""

    copied_images = summary_data.get("copied_images", {})
    existing_with_ci_ref = summary_data.get("existing_with_ci_ref", {})
    existing_without_ci_ref = summary_data.get("existing_without_ci_ref", [])
    images_to_build = summary_data.get("images_to_build", [])

    sections = [
        _render_release_artifact_map(
            "Images Promoted from Models CI",
            copied_images if isinstance(copied_images, dict) else {},
            "No images were copied from Models CI.",
            "from",
        ),
        _render_release_artifact_map(
            "Existing Images with Models CI reference",
            existing_with_ci_ref if isinstance(existing_with_ci_ref, dict) else {},
            "No existing images with Models CI reference.",
            "CI source",
            description=(
                "Images that already exist on remote and have a valid Models CI image available."
            ),
        ),
        _render_release_artifact_list(
            "Existing Images without Models CI reference",
            sorted(existing_without_ci_ref)
            if isinstance(existing_without_ci_ref, list)
            else [],
            "No existing images without Models CI reference.",
            description=(
                "Images that already exist on remote but have no valid Models CI reference "
                "(manually built/pushed)."
            ),
        ),
        _render_release_artifact_list(
            "Docker Images Requiring New Builds",
            sorted(images_to_build) if isinstance(images_to_build, list) else [],
            "No images need to be built.",
            description=(
                "**Note:** Model Specs added outside of Models CI will need to have Docker "
                "images built manually and will show up here if not already existing. This "
                "will happen by design when the VERSION file is incremented."
            ),
        ),
    ]
    return "\n\n".join(section for section in sections if section.strip())


def _normalize_model_name(model_name: Any) -> str:
    return str(model_name or "").strip().lower()


def _normalize_engine_key(inference_engine: Any) -> str:
    return normalize_inference_engine_name(inference_engine).strip().lower()


def _performance_full_key(entry: PerformanceEntry) -> PerformanceFullKey:
    return (
        _normalize_model_name(entry.get("model")),
        normalize_device_name(entry.get("device")),
        str(entry.get("impl_id") or "").strip(),
        _normalize_engine_key(entry.get("inference_engine")),
    )


def _performance_partial_key(entry: PerformanceEntry) -> PerformancePartialKey:
    return (
        str(entry.get("impl_id") or "").strip(),
        normalize_device_name(entry.get("device")),
        _normalize_engine_key(entry.get("inference_engine")),
    )


def _build_performance_entry_indices(
    release_performance_data: Dict[str, Any],
) -> Tuple[
    Dict[PerformanceFullKey, PerformanceEntry],
    Dict[PerformancePartialKey, PerformanceEntry],
]:
    full_index: Dict[PerformanceFullKey, PerformanceEntry] = {}
    partial_candidates: Dict[PerformancePartialKey, Any] = {}

    for entry in iter_release_performance_entries(release_performance_data):
        full_key = _performance_full_key(entry)
        partial_key = _performance_partial_key(entry)
        full_index[full_key] = entry

        existing_entry = partial_candidates.get(partial_key)
        if existing_entry is None:
            partial_candidates[partial_key] = entry
        else:
            partial_candidates[partial_key] = _AMBIGUOUS_MATCH

    partial_index = {
        key: value
        for key, value in partial_candidates.items()
        if value is not _AMBIGUOUS_MATCH
    }
    return full_index, partial_index


def _lookup_performance_entry(
    full_index: Dict[PerformanceFullKey, PerformanceEntry],
    partial_index: Dict[PerformancePartialKey, PerformanceEntry],
    record: ReleaseDiffRecord,
    device: str,
) -> Optional[PerformanceEntry]:
    full_key = (
        _normalize_model_name(record.get("model_arch")),
        normalize_device_name(device),
        str(record.get("impl_id") or "").strip(),
        _normalize_engine_key(record.get("inference_engine")),
    )
    entry = full_index.get(full_key)
    if entry is not None:
        return entry

    partial_key = (
        str(record.get("impl_id") or "").strip(),
        normalize_device_name(device),
        _normalize_engine_key(record.get("inference_engine")),
    )
    return partial_index.get(partial_key)


def _benchmark_row_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("task_type", "text"),
        row.get("isl", row.get("input_sequence_length", 0)),
        row.get("osl", row.get("output_sequence_length", 0)),
        row.get("image_height", 0),
        row.get("image_width", 0),
        row.get("images_per_prompt", 0),
        row.get("max_concurrency", row.get("max_con", 0)),
    )


def _index_benchmark_rows(rows: Any) -> Dict[Tuple[Any, ...], str]:
    indexed_rows = {}
    if not isinstance(rows, list):
        return indexed_rows
    for row in rows:
        if not isinstance(row, dict):
            continue
        indexed_rows[_benchmark_row_key(row)] = json.dumps(row, sort_keys=True)
    return indexed_rows


def _summarize_parameter_support_results(
    entry: Optional[PerformanceEntry],
) -> Dict[str, str]:
    if entry is None:
        return {}
    tests = entry.get("report_data", {}).get("parameter_support_tests", {})
    results = tests.get("results", {})
    if not isinstance(results, dict):
        return {}

    summary = {}
    for test_case, test_results in results.items():
        if not isinstance(test_results, list):
            continue
        total_count = len(test_results)
        passed_count = sum(
            1 for test_result in test_results if test_result.get("status") == "passed"
        )
        failed_count = sum(
            1 for test_result in test_results if test_result.get("status") == "failed"
        )
        summary[test_case] = (
            f"{passed_count}/{total_count} passed, {failed_count} failed"
        )
    return summary


def _format_change_counts(label: str, added: int, removed: int, updated: int) -> str:
    parts = []
    if added:
        parts.append(f"+{added}")
    if removed:
        parts.append(f"-{removed}")
    if updated:
        parts.append(f"~{updated}")
    if not parts:
        return ""
    return f"{label} {' '.join(parts)}"


def _summarize_benchmark_changes(
    before_entry: Optional[PerformanceEntry], after_entry: Optional[PerformanceEntry]
) -> str:
    before_rows = _index_benchmark_rows(
        before_entry.get("benchmarks_summary") if before_entry else []
    )
    after_rows = _index_benchmark_rows(
        after_entry.get("benchmarks_summary") if after_entry else []
    )
    if not before_rows and not after_rows:
        return ""
    if not before_rows:
        return f"Benchmarks added ({len(after_rows)} row(s))"
    if not after_rows:
        return f"Benchmarks removed ({len(before_rows)} row(s))"

    before_keys = set(before_rows.keys())
    after_keys = set(after_rows.keys())
    updated_count = sum(
        1 for key in before_keys & after_keys if before_rows[key] != after_rows[key]
    )
    return _format_change_counts(
        "Benchmarks",
        len(after_keys - before_keys),
        len(before_keys - after_keys),
        updated_count,
    )


def _summarize_test_result_changes(
    before_entry: Optional[PerformanceEntry], after_entry: Optional[PerformanceEntry]
) -> str:
    before_summary = _summarize_parameter_support_results(before_entry)
    after_summary = _summarize_parameter_support_results(after_entry)
    if not before_summary and not after_summary:
        return ""
    if not before_summary:
        return f"LLM API tests added ({len(after_summary)} case(s))"
    if not after_summary:
        return f"LLM API tests removed ({len(before_summary)} case(s))"

    before_keys = set(before_summary.keys())
    after_keys = set(after_summary.keys())
    updated_count = sum(
        1
        for key in before_keys & after_keys
        if before_summary[key] != after_summary[key]
    )
    return _format_change_counts(
        "LLM API tests",
        len(after_keys - before_keys),
        len(before_keys - after_keys),
        updated_count,
    )


def _summarize_performance_diff(
    before_entry: Optional[PerformanceEntry], after_entry: Optional[PerformanceEntry]
) -> str:
    if before_entry is None and after_entry is None:
        return "No release data"
    if before_entry is None:
        return "New release data"
    if after_entry is None:
        return "Removed release data"

    changes = []
    if before_entry.get("perf_status") != after_entry.get("perf_status"):
        changes.append(
            "Perf status: "
            + _format_status_change(
                before_entry.get("perf_status"),
                after_entry.get("perf_status"),
                unchanged_label="No change",
            )
        )
    if before_entry.get("accuracy_status") != after_entry.get("accuracy_status"):
        changes.append(
            "Accuracy status: "
            + _format_status_change(
                str(before_entry.get("accuracy_status"))
                if before_entry.get("accuracy_status") is not None
                else None,
                str(after_entry.get("accuracy_status"))
                if after_entry.get("accuracy_status") is not None
                else None,
                unchanged_label="No change",
            )
        )

    benchmark_change = _summarize_benchmark_changes(before_entry, after_entry)
    if benchmark_change:
        changes.append(benchmark_change)

    test_change = _summarize_test_result_changes(before_entry, after_entry)
    if test_change:
        changes.append(test_change)

    if not changes:
        return "No performance changes"
    return "; ".join(changes)


def _render_performance_diff_cell(
    record: ReleaseDiffRecord,
    current_indices: Tuple[
        Dict[PerformanceFullKey, PerformanceEntry],
        Dict[PerformancePartialKey, PerformanceEntry],
    ],
    base_indices: Tuple[
        Dict[PerformanceFullKey, PerformanceEntry],
        Dict[PerformancePartialKey, PerformanceEntry],
    ],
) -> str:
    device_summaries = []
    for device in record.get("devices", []):
        current_entry = _lookup_performance_entry(
            current_indices[0], current_indices[1], record, device
        )
        base_entry = _lookup_performance_entry(
            base_indices[0], base_indices[1], record, device
        )
        summary = _summarize_performance_diff(base_entry, current_entry)
        device_summaries.append(f"{device}: {summary}")

    if not device_summaries:
        return "No release data"
    return "<br>".join(device_summaries)


def render_model_diff_markdown(
    release_diff_records: Sequence[ReleaseDiffRecord],
    release_performance_data: Dict[str, Any],
    base_release_performance_data: Dict[str, Any],
) -> str:
    """Render the model diff section from raw diff and performance data."""
    lines = ["This document shows model specification updates.", ""]
    if not release_diff_records:
        lines.append("No updates were made.")
        return "\n".join(lines)

    current_indices = _build_performance_entry_indices(release_performance_data)
    base_indices = _build_performance_entry_indices(base_release_performance_data)

    lines.extend(
        [
            "| Impl | Model Arch | Weights | Devices | TT-Metal Commit Change | Status Change | CI Job Link | Performance Diff |",
            "|------|------------|---------|---------|------------------------|---------------|-------------|------------------|",
        ]
    )

    for record in release_diff_records:
        weights = record.get("weights") or []
        devices = record.get("devices") or []
        performance_diff = _render_performance_diff_cell(
            record, current_indices, base_indices
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table_cell(f"`{record.get('impl_id', '')}`"),
                    _escape_table_cell(f"`{record.get('model_arch', '')}`"),
                    _escape_table_cell(
                        "<br>".join(f"`{weight}`" for weight in weights) or "N/A"
                    ),
                    _escape_table_cell("<br>".join(devices) or "N/A"),
                    _escape_table_cell(
                        _format_commit_change(
                            record.get("tt_metal_commit_before"),
                            record.get("tt_metal_commit_after"),
                        )
                    ),
                    _escape_table_cell(
                        _format_status_change(
                            record.get("status_before"),
                            record.get("status_after"),
                        )
                    ),
                    _escape_table_cell(_format_ci_link(record)),
                    _escape_table_cell(performance_diff),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def build_release_notes(
    version: str,
    model_diff_records: Sequence[ReleaseDiffRecord],
    artifacts_summary_data: Dict[str, Any],
    release_performance_data: Dict[str, Any],
    base_release_performance_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the full release notes markdown document from raw release data."""
    ordered_sections = [
        ("summary_of_changes", ""),
        ("recommended_system_software_versions", ""),
        ("known_issues", ""),
        (
            "model_hardware_support_diff",
            render_model_diff_markdown(
                model_diff_records,
                release_performance_data,
                base_release_performance_data or EMPTY_RELEASE_PERFORMANCE_DATA,
            ),
        ),
        ("scale_out", ""),
        ("deprecations_breaking_changes", ""),
        (
            "release_artifacts_summary",
            render_release_artifacts_summary(artifacts_summary_data),
        ),
        ("contributors", ""),
        ("assets", ""),
    ]

    parts = [f"# tt-inference-server v{version}"]
    for section_key, section_body in ordered_sections:
        parts.append(render_section(SECTION_TITLES[section_key], section_body))
    return "\n\n".join(parts).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate GitHub release notes from versioned release outputs"
    )
    parser.add_argument(
        "--version",
        help="Release version without leading v. Defaults to the VERSION file.",
    )
    parser.add_argument(
        "--version-file",
        default="VERSION",
        help="Path to VERSION file (default: VERSION).",
    )
    parser.add_argument(
        "--artifacts-summary-json",
        help="Path to release artifact summary JSON.",
    )
    parser.add_argument(
        "--model-diff-json",
        help="Path to pre-release model and hardware diff JSON.",
    )
    parser.add_argument(
        "--release-performance-json",
        help="Path to the current release performance JSON baseline.",
    )
    parser.add_argument(
        "--base-release-performance-json",
        help="Optional path to the base release performance JSON for diffing.",
    )
    parser.add_argument(
        "--output",
        help="Output markdown path. Defaults to release_logs/v{VERSION}/release_notes_v{VERSION}.md.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    version = args.version or read_version(Path(args.version_file))
    release_dir = get_versioned_release_logs_dir(version)

    artifacts_summary_path = (
        Path(args.artifacts_summary_json)
        if args.artifacts_summary_json
        else release_dir / "release_artifacts_summary.json"
    )
    model_diff_path = (
        Path(args.model_diff_json)
        if args.model_diff_json
        else release_dir / "pre_release_models_diff.json"
    )
    release_performance_path = (
        Path(args.release_performance_json)
        if args.release_performance_json
        else get_release_performance_path()
    )
    output_path = (
        Path(args.output)
        if args.output
        else release_dir / f"release_notes_v{version}.md"
    )

    model_diff_records = load_release_diff_records(model_diff_path)
    artifacts_summary_data = load_optional_json(artifacts_summary_path, default={})
    release_performance_data = load_optional_json(
        release_performance_path,
        default=EMPTY_RELEASE_PERFORMANCE_DATA,
    )
    if args.base_release_performance_json:
        base_release_performance_data = load_optional_json(
            Path(args.base_release_performance_json),
            default=EMPTY_RELEASE_PERFORMANCE_DATA,
        )
    else:
        base_release_performance_data = load_git_release_performance_data(
            release_performance_path
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    notes = build_release_notes(
        version=version,
        model_diff_records=model_diff_records,
        artifacts_summary_data=artifacts_summary_data,
        release_performance_data=release_performance_data,
        base_release_performance_data=base_release_performance_data,
    )
    output_path.write_text(notes, encoding="utf-8")

    print(f"Wrote draft release notes to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
