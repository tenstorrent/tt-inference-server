# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast

from workflows.helm_generator import ENGINE_PRECEDENCE, MAPPERS
from workflows.helm_generator.base_mapper import HelmValuesMapper, model_name_from_spec
from workflows.helm_generator.device import device_key, is_multihost
from workflows.helm_generator.errors import GenerateHelmValuesError
from workflows.helm_generator.merge import (
    MergeResult,
    format_change_summary,
    merge_spec,
    set_default_engine,
)
from workflows.helm_generator.schema import HelmModelSpec
from workflows.helm_generator.yaml_io import dump_values, dumps_values, load_values
from workflows.model_spec import MODEL_SPECS, ModelSpec

logger = logging.getLogger("helm_generator")

DEFAULT_VALUES_PATH = (
    Path(__file__).resolve().parents[2]
    / "charts"
    / "tt-inference-server"
    / "values.yaml"
)


def _spec_engine_value(spec: ModelSpec) -> str:
    return cast(str, spec.inference_engine)


def _mapper_for(spec: ModelSpec) -> HelmValuesMapper:
    return MAPPERS[_spec_engine_value(spec)]


def assert_unique(specs: Iterable[ModelSpec]) -> None:
    """Each (model_name, device, engine, impl) must be unique."""
    seen: Dict[Tuple[str, str, str, str], str] = {}
    for spec in specs:
        key = (
            model_name_from_spec(spec),
            device_key(spec.device_type),
            _spec_engine_value(spec),
            spec.impl.impl_id,
        )
        if key in seen:
            raise GenerateHelmValuesError(
                f"Duplicate (model_name, device, engine, impl) found: {key}. "
                f"First seen as model_id={seen[key]}, then as model_id={spec.model_id}."
            )
        seen[key] = spec.model_id


def assert_single_default_impl(specs: Iterable[ModelSpec]) -> None:
    """When a (model, device, engine) group has more than one impl, exactly
    one must be marked default_impl=True. Single-impl groups are trivially
    the default and don't need the flag.
    """
    groups: Dict[Tuple[str, str, str], List[ModelSpec]] = defaultdict(list)
    for spec in specs:
        key = (
            model_name_from_spec(spec),
            device_key(spec.device_type),
            _spec_engine_value(spec),
        )
        groups[key].append(spec)
    for key, group in groups.items():
        if len(group) <= 1:
            continue
        defaults = [s for s in group if s.device_model_spec.default_impl]
        if len(defaults) != 1:
            impls = [(s.impl.impl_id, s.device_model_spec.default_impl) for s in group]
            raise GenerateHelmValuesError(
                f"({key}): {len(group)} impls but {len(defaults)} marked default_impl=True. "
                f"Impls: {impls}. Fix workflows/model_spec.py."
            )


def filter_specs(
    specs: Iterable[ModelSpec],
    *,
    model_names: Optional[Sequence[str]] = None,
    device_names: Optional[Sequence[str]] = None,
    engines: Optional[Sequence[str]] = None,
    include_multihost: bool = False,
) -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for spec in specs:
        if not include_multihost and is_multihost(spec.device_type):
            logger.debug(
                "skipping multihost spec %s (use --include-multihost)", spec.model_id
            )
            continue
        if model_names and model_name_from_spec(spec) not in model_names:
            continue
        if device_names and device_key(spec.device_type) not in device_names:
            continue
        if engines and _spec_engine_value(spec).lower() not in engines:
            continue
        out.append(spec)
    return out


def compute_default_engine_per_model(specs: Iterable[ModelSpec]) -> Dict[str, str]:
    """For each model that has impls in multiple engines, return the engine
    that should win as defaultEngine. Single-engine models get a defaultEngine
    too (the only choice) so the chart always has one to fall back to.

    Picks by precedence (vllm > media > forge).
    """
    by_model: Dict[str, set] = defaultdict(set)
    for spec in specs:
        engine_label = _mapper_for(spec).engine
        by_model[model_name_from_spec(spec)].add(engine_label)

    out: Dict[str, str] = {}
    for model_name, engines in by_model.items():
        ranked = sorted(engines, key=lambda e: ENGINE_PRECEDENCE.index(e))
        out[model_name] = ranked[0]
    return out


def map_specs(
    specs: Iterable[ModelSpec],
) -> List[Tuple[HelmModelSpec, HelmValuesMapper]]:
    out: List[Tuple[HelmModelSpec, HelmValuesMapper]] = []
    for spec in specs:
        engine_value = _spec_engine_value(spec)
        mapper = MAPPERS.get(engine_value)
        if mapper is None:
            logger.warning(
                "no mapper for engine %s (spec %s); skipping",
                engine_value,
                spec.model_id,
            )
            continue
        out.append((mapper.map(spec), mapper))
    return out


def generate(
    *,
    values_path: Path,
    specs: Iterable[ModelSpec],
    dry_run: bool = False,
) -> int:
    """Map specs into HelmModelSpec instances and merge into values_path.

    Returns the number of specs that resulted in a file change.
    """
    spec_list = list(specs)
    assert_unique(spec_list)
    assert_single_default_impl(spec_list)
    default_engines = compute_default_engine_per_model(spec_list)

    doc = load_values(values_path)
    mapped_list = map_specs(spec_list)

    changed_count = 0
    for mapped, mapper in mapped_list:
        result: MergeResult = merge_spec(doc, mapped, mapper.owned_leaf_paths())
        if result.changed:
            changed_count += 1
        logger.info(
            "%s/%s/%s/%s: %s",
            mapped.model_name,
            mapped.engine,
            mapped.device_name,
            mapped.impl_id,
            format_change_summary(result),
        )

    for model_name, engine in default_engines.items():
        if set_default_engine(doc, model_name, engine):
            logger.info("%s: defaultEngine=%s", model_name, engine)

    if dry_run:
        sys.stdout.write(dumps_values(doc))
    else:
        dump_values(doc, values_path)

    if dry_run:
        logger.info("Updated %d / %d specs", changed_count, len(mapped_list))
    else:
        logger.info(
            "Updated %d / %d specs; wrote %s",
            changed_count,
            len(mapped_list),
            values_path,
        )
    return changed_count


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m workflows.helm_generator",
        description=(
            "Generate or update charts/tt-inference-server/values.yaml entries "
            "from workflows/model_spec.py ModelSpec templates."
        ),
    )
    p.add_argument(
        "--model",
        action="append",
        default=[],
        dest="models",
        metavar="NAME",
        help="Filter by model name; repeatable.",
    )
    p.add_argument(
        "--device",
        action="append",
        default=[],
        dest="devices",
        metavar="NAME",
        help="Filter by device key (lowercase); repeatable.",
    )
    p.add_argument(
        "--engine",
        action="append",
        choices=list(ENGINE_PRECEDENCE),
        default=[],
        dest="engines",
        help="Filter by engine; repeatable.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resulting YAML to stdout instead of writing to file.",
    )
    p.add_argument(
        "--include-multihost",
        action="store_true",
        help="Include DUAL_GALAXY / QUAD_GALAXY specs (skipped by default).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(name)s] %(message)s",
    )
    specs = filter_specs(
        MODEL_SPECS.values(),
        model_names=args.models or None,
        device_names=args.devices or None,
        engines=args.engines or None,
        include_multihost=args.include_multihost,
    )
    try:
        generate(
            values_path=DEFAULT_VALUES_PATH,
            specs=specs,
            dry_run=args.dry_run,
        )
    except GenerateHelmValuesError as e:
        logger.error("%s", e)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
