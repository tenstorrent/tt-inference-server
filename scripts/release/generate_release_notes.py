#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Generate GitHub release notes from versioned release artifacts.

The generated markdown follows the release-note structure requested by the
release workflow and combines:
- the pre-release model and hardware diff markdown
- the release artifacts summary markdown
"""

import argparse
from pathlib import Path
from typing import Optional

try:
    from release_paths import get_versioned_release_logs_dir
except ImportError:
    from scripts.release.release_paths import get_versioned_release_logs_dir


SECTION_TITLES = {
    "summary_of_changes": "Summary of Changes",
    "recommended_system_software_versions": "Recommended system software versions",
    "known_issues": "Known Issues",
    "model_hardware_support_diff": "Model and Hardware Support Diff",
    "performance": "Performance",
    "scale_out": "Scale Out",
    "deprecations_breaking_changes": "Deprecations and breaking changes",
    "release_artifacts_summary": "Release Artifacts Summary",
    "contributors": "Contributors",
    "assets": "Assets",
}


def read_version(version_file: Path) -> str:
    """Read VERSION file and return a trimmed semantic version string."""
    version = version_file.read_text().strip()
    if not version:
        raise ValueError(f"VERSION file is empty: {version_file}")
    return version


def normalize_markdown_block(text: str) -> str:
    """Trim leading and trailing blank lines while preserving body content."""
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def strip_leading_heading(text: str) -> str:
    """Remove a leading markdown heading so embedded sections do not double-title."""
    lines = normalize_markdown_block(text).splitlines()
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines)


def load_optional_text(path: Optional[Path]) -> str:
    """Return trimmed file contents if present, otherwise an empty string."""
    if not path or not path.exists():
        return ""
    return normalize_markdown_block(path.read_text())


def render_section(title: str, body: str) -> str:
    """Render a markdown section, leaving the body blank when no content exists."""
    if body.strip():
        return f"## {title}\n\n{body.strip()}"
    return f"## {title}\n"


def build_release_notes(
    version: str,
    model_diff_markdown: str,
    artifacts_summary_markdown: str,
) -> str:
    """Build the full release notes markdown document."""
    ordered_sections = [
        ("summary_of_changes", ""),
        ("recommended_system_software_versions", ""),
        ("known_issues", ""),
        (
            "model_hardware_support_diff",
            strip_leading_heading(model_diff_markdown) if model_diff_markdown else "",
        ),
        ("performance", ""),
        ("scale_out", ""),
        ("deprecations_breaking_changes", ""),
        (
            "release_artifacts_summary",
            strip_leading_heading(artifacts_summary_markdown)
            if artifacts_summary_markdown
            else "",
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
        "--artifacts-summary-markdown",
        help="Path to release artifact summary markdown.",
    )
    parser.add_argument(
        "--model-diff-markdown",
        help="Path to pre-release model and hardware diff markdown.",
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
        Path(args.artifacts_summary_markdown)
        if args.artifacts_summary_markdown
        else release_dir / "release_artifacts_summary.md"
    )
    model_diff_path = (
        Path(args.model_diff_markdown)
        if args.model_diff_markdown
        else release_dir / "pre_release_models_diff.md"
    )
    output_path = (
        Path(args.output)
        if args.output
        else release_dir / f"release_notes_v{version}.md"
    )

    model_diff_markdown = load_optional_text(model_diff_path)
    artifacts_summary_markdown = load_optional_text(artifacts_summary_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    notes = build_release_notes(
        version=version,
        model_diff_markdown=model_diff_markdown,
        artifacts_summary_markdown=artifacts_summary_markdown,
    )
    output_path.write_text(notes)

    print(f"Wrote draft release notes to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
