# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Resolve release-scope CI entries to exact dev catalog leaves."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from workflows.model_spec import (
    MODEL_SPEC_CATALOG_FILES,
    ModelSpec,
    ModelSpecTemplate,
    load_templates_from_yaml,
    model_spec_leaf_identity,
    resolve_model_spec,
    validate_model_specs,
)
from workflows.workflow_types import DeviceTypes, InferenceEngine


LeafIdentity = Tuple[str, str, str, str]

_RELEASE_DEVICE_ALIASES = {
    "BH_GALAXY": "BLACKHOLE_GALAXY",
}


@dataclass(frozen=True)
class ReleaseCombo:
    model_name: str
    engine: InferenceEngine
    device: DeviceTypes


@dataclass(frozen=True)
class ModelSpecSource:
    spec: ModelSpec
    template: ModelSpecTemplate
    path: Path
    template_index: int
    weight_index: int
    device_index: int


@dataclass(frozen=True)
class ResolvedReleaseCombo:
    combo: ReleaseCombo
    model_spec: ModelSpec
    identity: LeafIdentity
    source_path: Path
    template_index: int
    weight_index: int
    device_index: int
    source_template: ModelSpecTemplate


def iter_implementations(model_entry: dict) -> Iterable[dict]:
    """Yield flat and multi-engine CI entries through one interface."""
    if not isinstance(model_entry, dict):
        raise ValueError(f"Model CI entry must be an object, got {model_entry!r}")
    implementations = model_entry.get("implementations")
    if implementations is None:
        yield model_entry
    else:
        if not isinstance(implementations, list) or not implementations:
            raise ValueError(
                f"Model implementations must be a non-empty list, got "
                f"{implementations!r}"
            )
        for implementation in implementations:
            if not isinstance(implementation, dict):
                raise ValueError(
                    f"Model implementation must be an object, got {implementation!r}"
                )
            yield implementation


def _release_device(raw_device: str) -> DeviceTypes:
    if not isinstance(raw_device, str):
        raise ValueError(f"Release device must be a string, got {raw_device!r}")
    normalized = _RELEASE_DEVICE_ALIASES.get(raw_device.upper(), raw_device)
    try:
        return DeviceTypes.from_string(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid release device {raw_device!r}") from exc


def collect_release_combos(ci_config: dict) -> List[ReleaseCombo]:
    """Return the ordered, de-duplicated release scope from CI configuration."""
    if not isinstance(ci_config, dict):
        raise ValueError("CI configuration must be an object")
    models = ci_config.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("CI configuration 'models' must be an object")

    combos: List[ReleaseCombo] = []
    seen = set()
    for model_name, model_entry in models.items():
        for implementation in iter_implementations(model_entry):
            ci = implementation.get("ci", {})
            if not isinstance(ci, dict):
                raise ValueError(
                    f"CI schedules must be an object for model {model_name!r}"
                )
            release = ci.get("release")
            if release is None:
                continue
            if not isinstance(release, dict):
                raise ValueError(
                    f"Release schedule must be an object for model {model_name!r}"
                )

            raw_devices = release.get("devices")
            if raw_devices == "ALL":
                raise ValueError(
                    f"Release devices cannot be 'ALL' for model {model_name!r}"
                )
            if not isinstance(raw_devices, list) or not raw_devices:
                raise ValueError(
                    f"Release devices must be a non-empty list for model {model_name!r}"
                )

            raw_engine = implementation.get("inference_engine")
            try:
                engine = InferenceEngine.from_string(raw_engine)
            except (KeyError, AttributeError) as exc:
                raise ValueError(
                    f"Invalid release inference engine {raw_engine!r} "
                    f"for model {model_name!r}"
                ) from exc

            for raw_device in raw_devices:
                combo = ReleaseCombo(
                    model_name=model_name,
                    engine=engine,
                    device=_release_device(raw_device),
                )
                if combo not in seen:
                    combos.append(combo)
                    seen.add(combo)
    return combos


def load_dev_model_spec_sources(dev_dir: Path) -> List[ModelSpecSource]:
    """Load and validate expanded dev leaves while retaining YAML provenance."""
    sources: List[ModelSpecSource] = []
    for filename in MODEL_SPEC_CATALOG_FILES:
        path = Path(dev_dir) / filename
        for template_index, template in enumerate(load_templates_from_yaml(path)):
            for spec in template.expand_to_specs():
                weight_index = template.weights.index(spec.hf_model_repo)
                device_index = next(
                    index
                    for index, device_spec in enumerate(template.device_model_specs)
                    if device_spec.device == spec.device_type
                )
                sources.append(
                    ModelSpecSource(
                        spec=spec,
                        template=template,
                        path=path,
                        template_index=template_index,
                        weight_index=weight_index,
                        device_index=device_index,
                    )
                )

    validate_model_specs([source.spec for source in sources])
    return sources


def resolve_release_combo(
    combo: ReleaseCombo,
    sources: Iterable[ModelSpecSource],
) -> ResolvedReleaseCombo:
    """Resolve one release tuple to its unique default dev leaf and provenance."""
    source_list = list(sources)
    model_spec = resolve_model_spec(
        (source.spec for source in source_list),
        model=combo.model_name,
        device=combo.device,
        engine=combo.engine,
        catalog_name="dev release catalog",
    )
    if not model_spec.device_model_spec.default_impl:
        raise ValueError(
            f"Release combo {combo!r} does not resolve to an explicit default "
            f"implementation; selected {model_spec_leaf_identity(model_spec)!r}"
        )

    identity = model_spec_leaf_identity(model_spec)
    matching_sources = [
        source
        for source in source_list
        if model_spec_leaf_identity(source.spec) == identity
    ]
    if len(matching_sources) != 1:
        raise ValueError(
            f"Expected one source template for release identity {identity!r}, "
            f"found {len(matching_sources)}"
        )
    source = matching_sources[0]
    return ResolvedReleaseCombo(
        combo=combo,
        model_spec=model_spec,
        identity=identity,
        source_path=source.path,
        template_index=source.template_index,
        weight_index=source.weight_index,
        device_index=source.device_index,
        source_template=source.template,
    )


def resolve_release_combos(
    combos: Iterable[ReleaseCombo],
    sources: Iterable[ModelSpecSource],
) -> List[ResolvedReleaseCombo]:
    """Resolve every release tuple in input order, failing on the first error."""
    source_list = list(sources)
    return [resolve_release_combo(combo, source_list) for combo in combos]
