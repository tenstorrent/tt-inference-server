#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Backfill historical release report_data payloads from GitHub Releases.

This script:
- lists GitHub releases for a repository
- downloads each release artifact ZIP attachment
- extracts nested model artifact ZIPs when present
- finds report_data_*.json for models still present in the current
  release_model_spec.json
- copies matching report_data JSON files into a dedicated local backfill directory
- merges older release data into the checked-in release performance baseline
  only when no same-version or newer entry already exists
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.request import Request, urlopen

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from generate_release_artifacts import build_merged_spec_from_report_data_json
    from models_ci_reader import (
        check_auth,
        http_get,
        sanitize_filename,
    )
    from release_performance import (
        ReleasePerformanceWriteMode,
        load_release_performance_data,
        normalize_release_version,
        parse_release_version,
        update_release_performance_outputs,
        write_release_performance_data,
    )
    from release_paths import DEFAULT_RELEASE_LOG_ROOT
except ImportError:
    from scripts.release.generate_release_artifacts import (
        build_merged_spec_from_report_data_json,
    )
    from scripts.release.models_ci_reader import (
        check_auth,
        http_get,
        sanitize_filename,
    )
    from scripts.release.release_performance import (
        ReleasePerformanceWriteMode,
        load_release_performance_data,
        normalize_release_version,
        parse_release_version,
        update_release_performance_outputs,
        write_release_performance_data,
    )
    from scripts.release.release_paths import DEFAULT_RELEASE_LOG_ROOT

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
DEFAULT_OUTPUT_DIR = DEFAULT_RELEASE_LOG_ROOT / "backfill_release_performance"
DEFAULT_OWNER = "tenstorrent"
DEFAULT_REPO = "tt-inference-server"
BACKFILL_SUBDIR = "backfill"
MIN_RELEASE_VERSION = (0, 5, 0)
RELEASE_ARTIFACT_NAME_MARKERS = ("release_artifacts", "release_artefacts")


@dataclass
class BackfillStats:
    releases_visited: int = 0
    assets_downloaded: int = 0
    matching_models_found: int = 0
    reports_copied: int = 0
    missing_reports: int = 0
    releases_without_assets: int = 0
    download_failures: int = 0
    reports_patched: int = 0
    report_processing_errors: int = 0
    baseline_entries_updated: int = 0
    baseline_entries_skipped: int = 0

    def merge(self, other: "BackfillStats") -> None:
        self.releases_visited += other.releases_visited
        self.assets_downloaded += other.assets_downloaded
        self.matching_models_found += other.matching_models_found
        self.reports_copied += other.reports_copied
        self.missing_reports += other.missing_reports
        self.releases_without_assets += other.releases_without_assets
        self.download_failures += other.download_failures
        self.reports_patched += other.reports_patched
        self.report_processing_errors += other.report_processing_errors
        self.baseline_entries_updated += other.baseline_entries_updated
        self.baseline_entries_skipped += other.baseline_entries_skipped


@dataclass(frozen=True)
class ExtractedArtifact:
    """One archive root with an optional model identity and extracted reports."""

    artifact_dir: Path
    model_id: Optional[str]
    report_paths: Tuple[Path, ...]


def configure_logging() -> None:
    """Configure CLI logging."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill release report_data JSON files from GitHub releases."
    )
    parser.add_argument(
        "--owner", default=DEFAULT_OWNER, help="GitHub repository owner."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository name.")
    parser.add_argument(
        "--release-model-spec-path",
        default="release_model_spec.json",
        help="Path to the current checked-in release_model_spec.json file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where downloaded release bundles and copied reports are written.",
    )
    parser.add_argument(
        "--max-releases",
        type=int,
        default=None,
        help="Optional maximum number of releases to process after filtering.",
    )
    parser.add_argument(
        "--include-prereleases",
        action="store_true",
        help="Include prereleases in addition to published releases.",
    )
    return parser.parse_args(argv)


def _flatten_release_model_specs(
    release_model_spec_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Load the exported release model spec JSON into a model_id keyed map."""
    with release_model_spec_path.open("r", encoding="utf-8") as file:
        export_data = json.load(file)

    flattened: Dict[str, Dict[str, Any]] = {}
    nested_specs = export_data.get("model_specs") or {}
    for _, device_map in nested_specs.items():
        for _, engine_map in device_map.items():
            for _, impl_map in engine_map.items():
                for _, spec_dict in impl_map.items():
                    model_id = spec_dict.get("model_id")
                    if not model_id:
                        raise ValueError(
                            "release_model_spec.json entry is missing model_id: "
                            f"{spec_dict}"
                        )
                    flattened[str(model_id)] = spec_dict
    return flattened


def list_releases(owner: str, repo: str, token: str, per_page: int = 100) -> List[dict]:
    """Return all releases from the GitHub Releases API."""
    releases: List[dict] = []
    page = 1
    while True:
        url = (
            f"{GITHUB_API}/repos/{owner}/{repo}/releases"
            f"?per_page={per_page}&page={page}"
        )
        page_data = http_get(
            url,
            token,
            accept="application/vnd.github+json",
            return_json=True,
        )
        if not isinstance(page_data, list):
            raise ValueError(f"Expected list response from GitHub releases API: {url}")
        if not page_data:
            break
        releases.extend(page_data)
        if len(page_data) < per_page:
            break
        page += 1
    return releases


def filter_releases(
    releases: Sequence[dict],
    include_prereleases: bool = False,
    max_releases: Optional[int] = None,
) -> List[dict]:
    """Filter releases to published release entries only."""
    filtered = []
    for release in releases:
        if release.get("draft"):
            continue
        if not include_prereleases and release.get("prerelease"):
            continue

        parsed_version = parse_release_version(release.get("tag_name"))
        if parsed_version is None or parsed_version < MIN_RELEASE_VERSION:
            continue
        filtered.append(release)

    if max_releases is not None:
        return filtered[:max_releases]
    return filtered


def _is_release_artifacts_zip_name(asset_name: str) -> bool:
    normalized_name = asset_name.lower()
    return normalized_name.endswith(".zip") and any(
        marker in normalized_name for marker in RELEASE_ARTIFACT_NAME_MARKERS
    )


def select_release_asset(release: dict) -> Optional[dict]:
    """Pick the release artifacts zip asset for one GitHub release."""
    tag_name = str(release.get("tag_name") or "")
    assets = list(release.get("assets") or [])
    exact_names = {
        f"{tag_name}-release_artifacts.zip",
        f"{tag_name}-release_artefacts.zip",
    }

    for asset in assets:
        if asset.get("name") in exact_names:
            return asset

    fallback_assets = [
        asset
        for asset in assets
        if _is_release_artifacts_zip_name(str(asset.get("name") or ""))
    ]
    if fallback_assets:
        fallback_assets.sort(key=lambda asset: str(asset.get("name") or ""))
        return fallback_assets[0]
    return None


def download_release_asset_zip(asset: dict, token: str) -> bytes:
    """Download a release asset ZIP via the GitHub releases API."""
    asset_url = asset.get("url")
    browser_download_url = asset.get("browser_download_url")
    if asset_url:
        try:
            return http_get(
                str(asset_url),
                token,
                accept="application/octet-stream",
                timeout=300,
                strip_auth_on_redirect=True,
            )
        except Exception:
            if not browser_download_url:
                raise
            logger.warning(
                "Falling back to browser_download_url for release asset %s",
                asset.get("name"),
            )
    if browser_download_url:
        return download_release_asset_from_browser_url(str(browser_download_url))
    raise ValueError(f"Release asset is missing download URL fields: {asset}")


def download_release_asset_from_browser_url(browser_download_url: str) -> bytes:
    """Download a public GitHub release asset without API auth headers."""
    request = Request(
        browser_download_url,
        headers={"User-Agent": "models-ci-reader/1.0"},
        method="GET",
    )
    with urlopen(request, timeout=300) as response:
        return response.read()


def prepare_release_workspace(
    output_root: Path, release_tag: str, asset_name: str
) -> Dict[str, Path]:
    """Prepare stable on-disk paths for one release tag."""
    tag_dir = output_root / BACKFILL_SUBDIR / release_tag
    download_dir = tag_dir / "_download"
    extract_dir = tag_dir / "_extracted"
    reports_dir = tag_dir / "reports"
    download_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / sanitize_filename(asset_name)
    return {
        "tag_dir": tag_dir,
        "download_dir": download_dir,
        "extract_dir": extract_dir,
        "reports_dir": reports_dir,
        "zip_path": zip_path,
    }


def _normalize_zip_member_parts(member_name: str) -> Optional[Tuple[str, ...]]:
    member_path = PurePosixPath(member_name)
    normalized_parts = []
    for part in member_path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            return None
        normalized_parts.append(sanitize_filename(part))
    return tuple(normalized_parts)


def _is_report_data_member(member_path: PurePosixPath) -> bool:
    return member_path.suffix.lower() == ".json" and member_path.name.startswith(
        "report_data"
    )


def _is_model_spec_member(member_path: PurePosixPath) -> bool:
    if member_path.suffix.lower() != ".json":
        return False
    path_text = member_path.as_posix()
    return (
        "/runtime_model_specs/" in f"/{path_text}" or "/run_specs/" in f"/{path_text}"
    )


def _extract_model_id_from_json_bytes(json_bytes: bytes) -> Optional[str]:
    try:
        payload = json.loads(json_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("model_id"):
        return str(metadata["model_id"])
    model_id = payload.get("model_id")
    if model_id:
        return str(model_id)
    return None


def _infer_model_id_from_report_name(report_name: str) -> Optional[str]:
    if not report_name.startswith("report_data_") or not report_name.endswith(".json"):
        return None
    stem = report_name[len("report_data_") : -len(".json")]
    timestamp_marker = "_20"
    timestamp_index = stem.rfind(timestamp_marker)
    if timestamp_index > 0:
        return stem[:timestamp_index]
    suffix_parts = stem.rsplit("_", 1)
    if len(suffix_parts) == 2 and suffix_parts[1].isdigit():
        return suffix_parts[0]
    return stem or None


def _extract_report_data_archives(
    zip_bytes: bytes,
    extract_dir: Path,
) -> List[ExtractedArtifact]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        extracted_artifacts: List[ExtractedArtifact] = []
        report_paths: List[Path] = []
        model_id: Optional[str] = None

        for member_info in archive.infolist():
            if member_info.is_dir():
                continue

            member_path = PurePosixPath(member_info.filename)
            normalized_parts = _normalize_zip_member_parts(member_info.filename)
            if normalized_parts is None:
                logger.warning(
                    "Skipping archive member with unsafe path: %s", member_info.filename
                )
                continue

            if member_path.suffix.lower() == ".zip":
                nested_extract_dir = extract_dir.joinpath(
                    *normalized_parts[:-1],
                    f"{sanitize_filename(member_path.stem)}__unzipped",
                )
                try:
                    extracted_artifacts.extend(
                        _extract_report_data_archives(
                            archive.read(member_info), nested_extract_dir
                        )
                    )
                except zipfile.BadZipFile:
                    shutil.rmtree(nested_extract_dir, ignore_errors=True)
                    logger.warning(
                        "Skipping nested archive that is not a valid zip file: %s",
                        member_path,
                    )
                continue

            member_bytes = archive.read(member_info)
            if _is_model_spec_member(member_path) and model_id is None:
                model_id = _extract_model_id_from_json_bytes(member_bytes)
                continue

            if not _is_report_data_member(member_path):
                continue

            output_path = extract_dir.joinpath(*normalized_parts)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(member_bytes)
            report_paths.append(output_path)

            if model_id is None:
                model_id = _extract_model_id_from_json_bytes(
                    member_bytes
                ) or _infer_model_id_from_report_name(member_path.name)

        if report_paths or model_id:
            extracted_artifacts.insert(
                0,
                ExtractedArtifact(
                    artifact_dir=extract_dir,
                    model_id=model_id,
                    report_paths=tuple(report_paths),
                ),
            )
        return extracted_artifacts


def build_matching_model_artifact_index(
    extracted_artifacts: Iterable[ExtractedArtifact],
    current_model_ids: Set[str],
) -> Dict[str, List[ExtractedArtifact]]:
    """Index extracted artifact reports by model_id, restricted to current specs."""
    model_artifact_index: Dict[str, List[ExtractedArtifact]] = {}
    for artifact in extracted_artifacts:
        if not artifact.model_id or artifact.model_id not in current_model_ids:
            continue
        model_artifact_index.setdefault(artifact.model_id, []).append(artifact)
    return model_artifact_index


def select_report_data_file(artifacts: Sequence[ExtractedArtifact]) -> Optional[Path]:
    """Pick the newest extracted report_data JSON among matching artifacts."""
    report_paths = [
        report_path for artifact in artifacts for report_path in artifact.report_paths
    ]
    if not report_paths:
        return None
    return max(
        report_paths,
        key=lambda report_path: (
            report_path.stat().st_mtime_ns,
            report_path.name,
        ),
    )


def copy_report_data_file(report_path: Path, reports_dir: Path, model_id: str) -> Path:
    """Copy one report_data JSON file into the stable backfill output layout."""
    model_dir = reports_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / report_path.name
    shutil.copy2(report_path, destination)
    return destination


def patch_missing_release_version(report_path: Path, release_tag: str) -> bool:
    """Backfill metadata.release_version in a staged report when it is absent."""
    release_version = normalize_release_version(release_tag)
    if release_version is None:
        raise ValueError(f"Could not parse release version from tag {release_tag}")

    with report_path.open("r", encoding="utf-8") as file:
        report_data = json.load(file)
    if not isinstance(report_data, dict):
        raise ValueError(f"Report data JSON must contain an object: {report_path}")

    metadata = report_data.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Report data JSON is missing metadata object: {report_path}")
    if metadata.get("release_version"):
        return False

    metadata["release_version"] = release_version
    report_path.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")
    return True


def merge_report_into_release_performance(
    report_path: Path,
    model_id: str,
    release_performance_data: Dict[str, Any],
) -> bool:
    """Merge one staged report into the checked-in baseline when it is newer."""
    merged_spec = build_merged_spec_from_report_data_json(report_path, is_dev=False)
    record = merged_spec.get(model_id)
    if record is None or not record.ci_data:
        raise ValueError(
            "Standalone report did not produce merged CI data for "
            f"expected model_id {model_id}: {report_path}"
        )

    update_result = update_release_performance_outputs(
        [record],
        mode=ReleasePerformanceWriteMode.MERGE_NEWER_ONLY,
        existing_baseline_data=release_performance_data,
    )
    if not update_result.artifacts.records_with_entries:
        return False
    release_version = update_result.artifacts.records_with_entries[
        0
    ].baseline_entry.get("release_version")
    if normalize_release_version(release_version) is None:
        raise ValueError(
            "Release performance entry is missing a parseable release_version: "
            f"{report_path}"
        )

    release_performance_data.clear()
    release_performance_data.update(update_result.final_baseline_data)
    return update_result.updated_count > 0


def collect_reports_for_release(
    release: dict,
    output_root: Path,
    current_model_ids: Set[str],
    token: str,
    release_performance_data: Dict[str, Any],
) -> BackfillStats:
    """Download one release bundle and collect matching report_data JSON files."""
    stats = BackfillStats(releases_visited=1)
    release_tag = str(release.get("tag_name") or "unknown-release")
    asset = select_release_asset(release)
    if asset is None:
        logger.warning(f"No release artifacts ZIP found for {release_tag}")
        stats.releases_without_assets = 1
        return stats

    asset_name = str(asset.get("name") or f"{release_tag}-release_artifacts.zip")
    workspace_paths = prepare_release_workspace(output_root, release_tag, asset_name)

    try:
        zip_bytes = download_release_asset_zip(asset, token)
    except Exception as exc:
        logger.error(f"Failed to download release asset for {release_tag}: {exc}")
        stats.download_failures = 1
        return stats

    stats.assets_downloaded = 1
    workspace_paths["zip_path"].write_bytes(zip_bytes)
    extracted_artifacts = _extract_report_data_archives(
        zip_bytes,
        workspace_paths["extract_dir"] / sanitize_filename(Path(asset_name).stem),
    )

    model_artifact_index = build_matching_model_artifact_index(
        extracted_artifacts, current_model_ids
    )
    stats.matching_models_found = len(model_artifact_index)

    for model_id, artifacts in sorted(model_artifact_index.items()):
        report_path = select_report_data_file(artifacts)
        if report_path is None:
            logger.error(
                "Missing report_data JSON for current model %s in release %s "
                "(artifact roots: %s)",
                model_id,
                release_tag,
                ", ".join(str(artifact.artifact_dir) for artifact in artifacts),
            )
            stats.missing_reports += 1
            continue

        destination = copy_report_data_file(
            report_path, workspace_paths["reports_dir"], model_id
        )
        logger.info(
            "Copied %s report data from %s to %s",
            model_id,
            release_tag,
            destination,
        )
        stats.reports_copied += 1

        try:
            if patch_missing_release_version(destination, release_tag):
                stats.reports_patched += 1

            if merge_report_into_release_performance(
                destination,
                model_id,
                release_performance_data,
            ):
                logger.info(
                    "Updated release performance baseline for %s from %s",
                    model_id,
                    release_tag,
                )
                stats.baseline_entries_updated += 1
            else:
                logger.info(
                    "Skipped release performance baseline update for %s from %s "
                    "because same or newer data already exists",
                    model_id,
                    release_tag,
                )
                stats.baseline_entries_skipped += 1
        except Exception as exc:
            logger.error(
                "Failed to process staged report data for current model %s in release "
                "%s at %s: %s",
                model_id,
                release_tag,
                destination,
                exc,
            )
            stats.report_processing_errors += 1

    return stats


def backfill_release_reports(
    owner: str,
    repo: str,
    release_model_spec_path: Path,
    output_root: Path,
    max_releases: Optional[int] = None,
    include_prereleases: bool = False,
) -> BackfillStats:
    """Backfill report_data JSON files from GitHub release artifacts."""
    token = check_auth(owner=owner, repo=repo)
    current_model_specs = _flatten_release_model_specs(release_model_spec_path)
    current_model_ids = set(current_model_specs.keys())
    release_performance_data = load_release_performance_data()
    logger.info(
        "Loaded %s current release model specs from %s",
        len(current_model_ids),
        release_model_spec_path,
    )

    releases = filter_releases(
        list_releases(owner, repo, token),
        include_prereleases=include_prereleases,
        max_releases=max_releases,
    )
    logger.info("Processing %s GitHub releases", len(releases))

    output_root.mkdir(parents=True, exist_ok=True)
    stats = BackfillStats()
    for release in releases:
        release_stats = collect_reports_for_release(
            release=release,
            output_root=output_root,
            current_model_ids=current_model_ids,
            token=token,
            release_performance_data=release_performance_data,
        )
        stats.merge(release_stats)

    if stats.baseline_entries_updated:
        write_release_performance_data(release_performance_data)
        logger.info(
            "Written %s merged release performance updates",
            stats.baseline_entries_updated,
        )

    return stats


def log_summary(stats: BackfillStats, output_root: Path) -> None:
    """Log an end-of-run summary."""
    logger.info("Backfill complete")
    logger.info("  releases visited: %s", stats.releases_visited)
    logger.info("  assets downloaded: %s", stats.assets_downloaded)
    logger.info("  matching current models found: %s", stats.matching_models_found)
    logger.info("  reports copied: %s", stats.reports_copied)
    logger.info("  missing reports: %s", stats.missing_reports)
    logger.info("  releases without matching asset: %s", stats.releases_without_assets)
    logger.info("  download failures: %s", stats.download_failures)
    logger.info("  patched release_version fields: %s", stats.reports_patched)
    logger.info("  report processing errors: %s", stats.report_processing_errors)
    logger.info("  baseline entries updated: %s", stats.baseline_entries_updated)
    logger.info("  baseline entries skipped: %s", stats.baseline_entries_skipped)
    logger.info("  output directory: %s", output_root)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    configure_logging()
    args = parse_args(argv)
    stats = backfill_release_reports(
        owner=args.owner,
        repo=args.repo,
        release_model_spec_path=Path(args.release_model_spec_path).resolve(),
        output_root=Path(args.output_dir).resolve(),
        max_releases=args.max_releases,
        include_prereleases=args.include_prereleases,
    )
    log_summary(stats, Path(args.output_dir).resolve())
    return (
        1
        if stats.missing_reports
        or stats.download_failures
        or stats.report_processing_errors
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
