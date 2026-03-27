#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, DefaultDict, Dict, List, Optional, Sequence, Tuple

from workflows.model_spec import ModelSpecTemplate, spec_templates

CheckImageExists = Callable[[str, Optional[Dict[str, bool]]], bool]


@dataclass(frozen=True)
class TemplateImageReference:
    """Context for one template/image relationship."""

    template_label: str
    model_ids: Tuple[str, ...]


@dataclass(frozen=True)
class MissingTemplateImage:
    """A missing image and all templates/specs that reference it."""

    image: str
    references: Tuple[TemplateImageReference, ...]


@dataclass(frozen=True)
class TemplateImageValidationResult:
    """Structured result for template image validation."""

    missing_images: Tuple[MissingTemplateImage, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.missing_images


def _get_mode_specific_image(image: str, is_dev: bool) -> str:
    if is_dev:
        return image.replace("-release-", "-dev-")
    return image


def _build_template_label(template: ModelSpecTemplate) -> str:
    weights = ", ".join(template.weights)
    devices = ", ".join(
        device_model_spec.device.name
        for device_model_spec in template.device_model_specs
    )
    return (
        f"{template.impl.impl_name} | engine={template.inference_engine} | "
        f"weights={weights} | devices={devices}"
    )


def _collect_image_references(
    templates: Sequence[ModelSpecTemplate], is_dev: bool
) -> Dict[str, Tuple[TemplateImageReference, ...]]:
    image_references: DefaultDict[str, List[TemplateImageReference]] = defaultdict(list)
    for template in templates:
        model_ids_by_image: DefaultDict[str, List[str]] = defaultdict(list)
        for spec in template.expand_to_specs():
            image = _get_mode_specific_image(spec.docker_image, is_dev)
            model_ids_by_image[image].append(spec.model_id)

        template_label = _build_template_label(template)
        for image, model_ids in sorted(model_ids_by_image.items()):
            image_references[image].append(
                TemplateImageReference(
                    template_label=template_label,
                    model_ids=tuple(sorted(model_ids)),
                )
            )

    return {
        image: tuple(
            sorted(
                references,
                key=lambda reference: (reference.template_label, reference.model_ids),
            )
        )
        for image, references in sorted(image_references.items())
    }


def validate_model_spec_template_images(
    is_dev: bool,
    check_image_exists: CheckImageExists,
    *,
    templates: Optional[Sequence[ModelSpecTemplate]] = None,
    cache: Optional[Dict[str, bool]] = None,
) -> TemplateImageValidationResult:
    """Validate that every effective template image exists for the selected mode."""
    resolved_templates = templates or spec_templates
    image_references = _collect_image_references(resolved_templates, is_dev=is_dev)

    missing_images: List[MissingTemplateImage] = []
    for image, references in image_references.items():
        if check_image_exists(image, cache=cache):
            continue
        missing_images.append(
            MissingTemplateImage(
                image=image,
                references=references,
            )
        )

    return TemplateImageValidationResult(missing_images=tuple(missing_images))


def format_missing_template_image_validation_error(
    validation_result: TemplateImageValidationResult,
    *,
    is_dev: bool,
    readme_path: str = "scripts/release/README.md",
) -> str:
    """Format a user-facing error with manual build remediation steps."""
    image_target = "dev" if is_dev else "release"
    lines = [
        f"Missing {image_target} Docker images for ModelSpecTemplates.",
        "",
        "Follow the manual image build instructions in "
        f"{readme_path} step 5B for manually added models:",
        "python3 scripts/build_docker_images.py --push --release",
        "",
        "Missing images:",
    ]

    for missing_image in validation_result.missing_images:
        lines.append(f"- {missing_image.image}")
        for reference in missing_image.references:
            lines.append(f"  template: {reference.template_label}")
            lines.append(f"  model_ids: {', '.join(reference.model_ids)}")

    return "\n".join(lines)
