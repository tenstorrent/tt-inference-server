from argparse import Namespace
from scripts.release.generate_release_notes import build_release_notes
from scripts.release.generate_release_notes import main
from scripts.release.release_paths import (
    get_versioned_release_logs_dir,
    resolve_release_output_dir,
)
from unittest.mock import patch


def test_release_paths_use_versioned_release_logs():
    assert get_versioned_release_logs_dir("0.10.0").as_posix() == "release_logs/v0.10.0"
    assert (
        resolve_release_output_dir(version="0.10.0")
        .as_posix()
        .endswith("/release_logs/v0.10.0")
    )


def test_build_release_notes_uses_requested_sections_and_sources():
    notes = build_release_notes(
        version="0.10.0",
        model_diff_markdown="# Model Spec Release Updates\n\n- Updated supported models.",
        artifacts_summary_markdown="# Release Artifacts Summary\n\n- Promoted 3 images.",
    )

    assert notes.startswith("# tt-inference-server v0.10.0\n")
    assert "## Summary of Changes\n" in notes
    assert "## Recommended system software versions\n" in notes
    assert "## Known Issues\n" in notes
    assert "## Model and Hardware Support Diff\n\n- Updated supported models." in notes
    assert "## Performance\n" in notes
    assert "## Scale Out\n" in notes
    assert "## Deprecations and breaking changes\n" in notes
    assert "## Release Artifacts Summary\n\n- Promoted 3 images." in notes
    assert "## Contributors\n" in notes
    assert "## Assets\n" in notes
    assert "\n# Release Artifacts Summary\n\n- Promoted 3 images." not in notes
    assert "\n# Model Spec Release Updates\n\n- Updated supported models." not in notes


def test_main_reads_pre_release_model_diff_markdown_from_release_dir(tmp_path):
    version_file = tmp_path / "VERSION"
    version_file.write_text("0.10.0\n")

    release_dir = tmp_path / "release_logs" / "v0.10.0"
    release_dir.mkdir(parents=True)
    (release_dir / "pre_release_models_diff.md").write_text(
        "# Model Spec Release Updates\n\n- Updated supported models.\n"
    )
    (release_dir / "release_artifacts_summary.md").write_text(
        "# Release Artifacts Summary\n\n- Promoted 3 images.\n"
    )
    output_path = release_dir / "release_notes_v0.10.0.md"

    args = Namespace(
        version=None,
        version_file=str(version_file),
        artifacts_summary_markdown=None,
        model_diff_markdown=None,
        output=str(output_path),
    )

    with patch(
        "scripts.release.generate_release_notes.parse_args",
        return_value=args,
    ), patch(
        "scripts.release.generate_release_notes.get_versioned_release_logs_dir",
        return_value=release_dir,
    ):
        assert main() == 0

    notes = output_path.read_text()
    assert "## Model and Hardware Support Diff\n\n- Updated supported models." in notes
