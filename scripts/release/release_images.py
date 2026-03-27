#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Promote or build Step-6 release Docker images and write the release summary."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.build_docker_images import filter_model_configs_by_docker_images
from workflows.model_spec import MODEL_SPECS, get_model_id, model_weights_to_model_name

try:
    from generate_release_artifacts import (
        IMAGE_STATUS_COPY_FROM_CI,
        IMAGE_STATUS_EXISTS_WITH_CI,
        IMAGE_STATUS_EXISTS_WITHOUT_CI,
        IMAGE_STATUS_NEEDS_BUILD,
        build_generated_artifact_paths,
        check_crane_installed,
        check_docker_installed,
        check_image_exists,
        configure_logging,
        copy_docker_image,
        get_versioned_release_logs_dir,
        load_ci_data_from_artifacts_path,
        load_ci_data_from_run_id,
        merge_specs_with_ci_data,
        plan_image_action,
    )
    from release_paths import resolve_release_output_dir
except ImportError:
    from scripts.release.generate_release_artifacts import (
        IMAGE_STATUS_COPY_FROM_CI,
        IMAGE_STATUS_EXISTS_WITH_CI,
        IMAGE_STATUS_EXISTS_WITHOUT_CI,
        IMAGE_STATUS_NEEDS_BUILD,
        build_generated_artifact_paths,
        check_crane_installed,
        check_docker_installed,
        configure_logging,
        copy_docker_image,
        check_image_exists,
        get_versioned_release_logs_dir,
        load_ci_data_from_artifacts_path,
        load_ci_data_from_run_id,
        merge_specs_with_ci_data,
        plan_image_action,
    )
    from scripts.release.release_paths import resolve_release_output_dir

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for Step 6 release-image promotion."""
    default_output_dir = get_versioned_release_logs_dir()
    parser = argparse.ArgumentParser(
        description="Promote or build Step-6 release Docker images."
    )
    parser.add_argument(
        "ci_artifacts_path",
        nargs="?",
        help=(
            "Path to downloaded raw CI artifacts. If omitted, defaults to the versioned "
            "release output directory when it contains downloaded CI run logs."
        ),
    )
    parser.add_argument(
        "--models-ci-run-id",
        type=int,
        default=None,
        help=(
            "GitHub Actions workflow run ID; automatically downloads raw workflow "
            "artifacts and processes them directly."
        ),
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help=(
            "Output directory for downloaded CI reader artifacts when using "
            f"--models-ci-run-id (default: {default_output_dir})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Directory for release logs and summary outputs (default: {default_output_dir})",
    )
    parser.add_argument(
        "--release-model-spec-path",
        default="release_model_spec.json",
        help="Path to release_model_spec.json (default: release_model_spec.json)",
    )
    parser.add_argument(
        "--readme-path",
        default="README.md",
        help="Path to README.md used for generated-artifact links (default: README.md)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate release image state and write the summary without mutating images.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Do not build missing release images locally before promotion.",
    )
    parser.add_argument(
        "--accept-images",
        action="store_true",
        help="Skip the Enter-to-continue confirmation prompts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview build and promotion actions without executing them.",
    )
    args = parser.parse_args(argv)
    if args.ci_artifacts_path and args.models_ci_run_id is not None:
        parser.error(
            "Provide at most one of ci_artifacts_path or --models-ci-run-id for Step 6."
        )
    return args


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
                            f"release_model_spec.json entry is missing model_id: {spec_dict}"
                        )
                    flattened[str(model_id)] = spec_dict
    return flattened


def _load_release_diff_records(output_dir: Path) -> List[Dict[str, Any]]:
    diff_path = output_dir / "pre_release_models_diff.json"
    with diff_path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"Expected list in {diff_path}")
    return records


def _build_release_scope_model_ids(
    release_diff_records: Sequence[Dict[str, Any]],
) -> Set[str]:
    """Expand release diff template identities to concrete model_ids."""
    model_ids: Set[str] = set()
    for record in release_diff_records:
        impl_name = str(record.get("impl_id") or record.get("impl") or "").strip()
        weights = record.get("weights") or []
        devices = record.get("devices") or []
        if not impl_name or not weights or not devices:
            raise ValueError(f"Invalid release diff record: {record}")
        for weight in weights:
            model_name = model_weights_to_model_name(str(weight))
            for device in devices:
                model_ids.add(get_model_id(impl_name, model_name, str(device).lower()))
    return model_ids


def _load_optional_acceptance_warnings(output_dir: Path) -> List[Dict[str, Any]]:
    path = output_dir / "release_acceptance_warnings.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else []


def _collect_generated_artifacts(
    output_dir: Path, release_model_spec_path: Path, readme_path: Path
) -> List[str]:
    expected_paths = build_generated_artifact_paths(
        output_dir=output_dir,
        release_model_spec_path=release_model_spec_path,
        readme_path=readme_path,
    )
    existing_paths: List[str] = []
    for path_str in expected_paths:
        if Path(path_str).exists():
            existing_paths.append(path_str)
    return existing_paths


def _default_ci_artifacts_path(output_dir: Path) -> Optional[Path]:
    if (output_dir / "ci_run_logs").exists() or (
        output_dir / "run_ci_metadata.json"
    ).exists():
        return output_dir
    return None


def _load_optional_ci_data(
    args: argparse.Namespace, output_dir: Path
) -> Dict[str, Dict[str, Any]]:
    if args.models_ci_run_id is not None:
        return load_ci_data_from_run_id(
            args.models_ci_run_id,
            resolve_release_output_dir(args.out_root),
            workflow_file="release.yml",
        )

    if args.ci_artifacts_path:
        return load_ci_data_from_artifacts_path(Path(args.ci_artifacts_path))

    default_ci_path = _default_ci_artifacts_path(output_dir)
    if not default_ci_path:
        logger.info("No downloaded CI artifacts found under %s", output_dir)
        return {}

    try:
        return load_ci_data_from_artifacts_path(default_ci_path)
    except Exception as exc:
        logger.warning("Unable to load CI artifacts from %s: %s", default_ci_path, exc)
        return {}


def _prompt_to_continue(message: str, accept_images: bool) -> None:
    if accept_images:
        logger.info("%s --accept-images set; continuing without prompt.", message)
        return
    input(f"{message} Press Enter to continue or Ctrl-C to abort.")


def _build_missing_images(
    output_dir: Path,
    docker_images: Sequence[str],
    *,
    dry_run: bool,
    accept_images: bool,
) -> List[str]:
    if not docker_images:
        return []

    _prompt_to_continue(
        f"About to build {len(docker_images)} missing release image(s).",
        accept_images,
    )
    filtered_model_configs = filter_model_configs_by_docker_images(
        MODEL_SPECS,
        docker_images,
    )

    from scripts.build_docker_images import build_docker_images

    build_docker_images(
        filtered_model_configs,
        release=True,
        push=True,
        dry_run=dry_run,
    )

    if dry_run:
        return []

    built_images: List[str] = []
    for image in docker_images:
        if check_image_exists(image):
            built_images.append(image)
    return built_images


def write_output(
    *,
    output_dir: Path,
    built_images: Sequence[str],
    images_to_build: Sequence[str],
    copied_images: Dict[str, str],
    existing_with_ci_ref: Dict[str, str],
    existing_without_ci_ref: Sequence[str],
    generated_artifacts: Sequence[str],
    acceptance_warnings: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Write the final Step-6 release summary JSON and markdown files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_data = {
        "built_images": sorted(built_images),
        "images_to_build": sorted(images_to_build),
        "copied_images": copied_images,
        "existing_with_ci_ref": existing_with_ci_ref,
        "existing_without_ci_ref": sorted(existing_without_ci_ref),
        "generated_artifacts": sorted(generated_artifacts),
        "acceptance_warnings": list(acceptance_warnings),
        "summary": {
            "total_built": len(built_images),
            "total_to_build": len(images_to_build),
            "total_copied": len(copied_images),
            "total_existing_with_ci": len(existing_with_ci_ref),
            "total_existing_without_ci": len(existing_without_ci_ref),
            "total_generated_artifacts": len(generated_artifacts),
            "total_acceptance_warnings": len(acceptance_warnings),
        },
    }

    json_path = output_dir / "release_artifacts_summary.json"
    json_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    markdown_lines = ["# Release Artifacts Summary", ""]
    markdown_lines.extend(["## Release Acceptance Warnings", ""])
    if acceptance_warnings:
        markdown_lines.append(
            "These warnings are computed locally from release report data and do not block artifact generation."
        )
        markdown_lines.append("")
        for warning in acceptance_warnings:
            heading = (
                warning.get("heading") or warning.get("model_id") or "Unknown model"
            )
            markdown_lines.append(f"### {heading}")
            markdown_lines.append("")
            if warning.get("ci_job_url"):
                markdown_lines.append(f"CI job: {warning['ci_job_url']}")
                markdown_lines.append("")
            summary_markdown = str(warning.get("summary_markdown") or "").strip()
            if summary_markdown:
                markdown_lines.append(summary_markdown)
                markdown_lines.append("")
    else:
        markdown_lines.append("No release acceptance warnings were generated.")
        markdown_lines.append("")

    markdown_lines.extend(["## Generated Release Artifacts", ""])
    if generated_artifacts:
        for artifact_path in sorted(generated_artifacts):
            markdown_lines.append(f"- `{artifact_path}`")
        markdown_lines.extend(["", f"**Total:** {len(generated_artifacts)}", ""])
    else:
        markdown_lines.extend(["No generated release artifacts were recorded.", ""])

    markdown_lines.extend(["## Docker Images Built Locally", ""])
    if built_images:
        for image in sorted(built_images):
            markdown_lines.append(f"- {image.replace('ghcr.io/', 'https://ghcr.io/')}")
        markdown_lines.extend(["", f"**Total:** {len(built_images)}", ""])
    else:
        markdown_lines.extend(["No images were built locally.", ""])

    markdown_lines.extend(["## Images Promoted from Models CI", ""])
    if copied_images:
        for destination, source in sorted(copied_images.items()):
            markdown_lines.append(
                f"- {destination.replace('ghcr.io/', 'https://ghcr.io/')}"
            )
            markdown_lines.append(
                f"  - from: {source.replace('ghcr.io/', 'https://ghcr.io/')}"
            )
            markdown_lines.append("")
        markdown_lines.append(f"**Total:** {len(copied_images)}")
        markdown_lines.append("")
    else:
        markdown_lines.extend(["No images were copied from Models CI.", ""])

    markdown_lines.extend(["## Existing Images with Models CI reference", ""])
    if existing_with_ci_ref:
        for destination, source in sorted(existing_with_ci_ref.items()):
            markdown_lines.append(
                f"- {destination.replace('ghcr.io/', 'https://ghcr.io/')}"
            )
            markdown_lines.append(
                f"  - CI source: {source.replace('ghcr.io/', 'https://ghcr.io/')}"
            )
            markdown_lines.append("")
        markdown_lines.append(f"**Total:** {len(existing_with_ci_ref)}")
        markdown_lines.append("")
    else:
        markdown_lines.extend(["No existing images with Models CI reference.", ""])

    markdown_lines.extend(["## Existing Images without Models CI reference", ""])
    if existing_without_ci_ref:
        for image in sorted(existing_without_ci_ref):
            markdown_lines.append(f"- {image.replace('ghcr.io/', 'https://ghcr.io/')}")
        markdown_lines.extend(["", f"**Total:** {len(existing_without_ci_ref)}", ""])
    else:
        markdown_lines.extend(["No existing images without Models CI reference.", ""])

    markdown_lines.extend(["## Docker Images Requiring New Builds", ""])
    if images_to_build:
        for image in sorted(images_to_build):
            markdown_lines.append(f"- {image.replace('ghcr.io/', 'https://ghcr.io/')}")
        markdown_lines.extend(["", f"**Total:** {len(images_to_build)}"])
    else:
        markdown_lines.append("No images need to be built.")

    markdown_path = output_dir / "release_artifacts_summary.md"
    markdown_path.write_text(
        "\n".join(markdown_lines).rstrip() + "\n", encoding="utf-8"
    )
    logger.info("Written release summary to %s and %s", json_path, markdown_path)
    return output_data


def run_from_args(args: argparse.Namespace) -> int:
    """Run the Step-6 release image flow from parsed arguments."""
    output_dir = resolve_release_output_dir(args.output_dir)
    release_model_spec_path = Path(args.release_model_spec_path)
    readme_path = Path(args.readme_path)
    acceptance_warnings = _load_optional_acceptance_warnings(output_dir)
    generated_artifacts = _collect_generated_artifacts(
        output_dir,
        release_model_spec_path,
        readme_path,
    )

    if not check_docker_installed():
        return 1

    flattened_release_specs = _flatten_release_model_specs(release_model_spec_path)
    all_target_images = sorted(
        {
            str(spec_dict["docker_image"])
            for spec_dict in flattened_release_specs.values()
            if spec_dict.get("docker_image")
        }
    )
    release_diff_records = _load_release_diff_records(output_dir)
    release_scope_model_ids = _build_release_scope_model_ids(release_diff_records)
    missing_scope_model_ids = sorted(
        model_id
        for model_id in release_scope_model_ids
        if model_id not in flattened_release_specs
    )
    if missing_scope_model_ids:
        raise ValueError(
            "Release diff references model_ids missing from release_model_spec.json: "
            + ", ".join(missing_scope_model_ids)
        )

    release_scope_images = {
        str(flattened_release_specs[model_id]["docker_image"])
        for model_id in release_scope_model_ids
        if flattened_release_specs[model_id].get("docker_image")
    }

    ci_data = _load_optional_ci_data(args, output_dir)
    merged_spec = merge_specs_with_ci_data(ci_data, is_dev=False)
    records_by_image: Dict[str, List[Any]] = {}
    for record in merged_spec.values():
        records_by_image.setdefault(record.target_docker_image, []).append(record)

    image_exists_cache: Dict[str, bool] = {}
    existing_with_ci_ref: Dict[str, str] = {}
    existing_without_ci_ref: List[str] = []
    promotable_images: Dict[str, str] = {}
    buildable_images: List[str] = []
    unexpected_missing_images: List[str] = []

    for target_image in all_target_images:
        records = records_by_image.get(target_image, [])
        image_plan = plan_image_action(
            target_image,
            records,
            image_exists_cache,
            allow_ci_promotion=bool(ci_data),
        )
        if (
            image_plan.status == IMAGE_STATUS_EXISTS_WITH_CI
            and image_plan.ci_source_image
        ):
            existing_with_ci_ref[target_image] = image_plan.ci_source_image
        elif image_plan.status == IMAGE_STATUS_EXISTS_WITHOUT_CI:
            existing_without_ci_ref.append(target_image)
        elif (
            image_plan.status == IMAGE_STATUS_COPY_FROM_CI
            and image_plan.ci_source_image
        ):
            promotable_images[target_image] = image_plan.ci_source_image
        elif image_plan.status == IMAGE_STATUS_NEEDS_BUILD:
            if target_image in release_scope_images:
                buildable_images.append(target_image)
            else:
                unexpected_missing_images.append(target_image)

    if unexpected_missing_images:
        for image in unexpected_missing_images:
            logger.error(
                "Missing release image outside current release scope: %s",
                image,
            )
        return 1

    built_images: List[str] = []
    unresolved_images_to_build = list(buildable_images)

    if args.validate_only:
        logger.info("Validate-only mode: skipping image build and promotion.")
    else:
        if buildable_images and args.no_build:
            logger.warning(
                "--no-build set; %d image(s) still require local builds.",
                len(buildable_images),
            )
        elif buildable_images:
            built_images = _build_missing_images(
                output_dir,
                buildable_images,
                dry_run=args.dry_run,
                accept_images=args.accept_images,
            )
            unresolved_images_to_build = [
                image for image in buildable_images if image not in set(built_images)
            ]

        copied_images: Dict[str, str] = {}
        if promotable_images:
            if not args.dry_run and not check_crane_installed():
                return 1
            _prompt_to_continue(
                f"About to promote {len(promotable_images)} image(s) from Models CI.",
                args.accept_images,
            )
            for destination, source in sorted(promotable_images.items()):
                if copy_docker_image(source, destination, dry_run=args.dry_run):
                    copied_images[destination] = source
                else:
                    logger.error("Failed to promote %s from %s", destination, source)
                    return 1
        else:
            copied_images = {}

        summary = write_output(
            output_dir=output_dir,
            built_images=built_images,
            images_to_build=unresolved_images_to_build,
            copied_images=copied_images,
            existing_with_ci_ref=existing_with_ci_ref,
            existing_without_ci_ref=existing_without_ci_ref,
            generated_artifacts=generated_artifacts,
            acceptance_warnings=acceptance_warnings,
        )

        if unresolved_images_to_build:
            logger.error(
                "Release still has %d unresolved image(s) requiring local builds.",
                len(unresolved_images_to_build),
            )
            return 1

        logger.info(
            "Release image step complete: built=%d promoted=%d existing=%d",
            summary["summary"]["total_built"],
            summary["summary"]["total_copied"],
            summary["summary"]["total_existing_with_ci"]
            + summary["summary"]["total_existing_without_ci"],
        )
        return 0

    write_output(
        output_dir=output_dir,
        built_images=[],
        images_to_build=buildable_images,
        copied_images={},
        existing_with_ci_ref=existing_with_ci_ref,
        existing_without_ci_ref=existing_without_ci_ref,
        generated_artifacts=generated_artifacts,
        acceptance_warnings=acceptance_warnings,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
