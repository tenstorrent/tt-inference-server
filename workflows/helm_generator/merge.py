# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Sequence, Set, Tuple

from workflows.helm_generator.schema import HelmModelSpec

PathTuple = Tuple[str, ...]


@dataclass
class MergeResult:
    changed: bool = False
    inserted_model: bool = False
    inserted_engine: bool = False
    inserted_device: bool = False
    inserted_impl: bool = False
    updated_paths: List[PathTuple] = field(default_factory=list)


_MISSING = object()


def _normalize_for_compare(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Mapping):
        return {k: _normalize_for_compare(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_compare(v) for v in value]
    return value


def _equal(a: Any, b: Any) -> bool:
    return _normalize_for_compare(a) == _normalize_for_compare(b)


def _ensure_models_map(doc: Mapping) -> Any:
    if "models" not in doc:
        raise KeyError("values.yaml is missing top-level 'models:' key")
    return doc["models"]


def _set_path(node: Any, path: Sequence[str], value: Any) -> None:
    cursor = node
    for key in path[:-1]:
        if key not in cursor or not isinstance(cursor[key], Mapping):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[path[-1]] = value


def _get_path(node: Any, path: Sequence[str]) -> Any:
    cursor: Any = node
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return _MISSING
        cursor = cursor[key]
    return cursor


def _pluck_path(d: Mapping, path: Sequence[str]) -> Any:
    cursor: Any = d
    for key in path:
        cursor = cursor[key]
    return cursor


def _path_in_dict(d: Mapping, path: Sequence[str]) -> bool:
    cursor: Any = d
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return False
        cursor = cursor[key]
    return True


def merge_spec(
    doc: Any,
    mapped: HelmModelSpec,
    owned_impl_paths: Set[PathTuple],
) -> MergeResult:
    """Merge a single HelmModelSpec into a loaded values.yaml document.

    Navigates models.<m>.<engine>.<d>.impls.<impl_id>; inserts intermediate
    structure as needed. When an impl block already exists, overwrites only
    the leaf paths in owned_impl_paths; other keys (user-added) are preserved.

    Sets device-level defaultImpl when mapped.is_default is True. Does NOT
    touch model-level defaultEngine — the CLI handles that after the batch.
    """
    result = MergeResult()
    models = _ensure_models_map(doc)
    impl_block = mapped.config.to_yaml_dict()

    if mapped.model_name not in models:
        models[mapped.model_name] = {
            mapped.engine: {
                mapped.device_name: {
                    "defaultImpl": mapped.impl_id,
                    "impls": {mapped.impl_id: impl_block},
                }
            }
        }
        result.changed = True
        result.inserted_model = True
        return result

    model_entry = models[mapped.model_name]

    if mapped.engine not in model_entry:
        model_entry[mapped.engine] = {
            mapped.device_name: {
                "defaultImpl": mapped.impl_id,
                "impls": {mapped.impl_id: impl_block},
            }
        }
        result.changed = True
        result.inserted_engine = True
        return result

    engine_entry = model_entry[mapped.engine]

    if mapped.device_name not in engine_entry:
        engine_entry[mapped.device_name] = {
            "defaultImpl": mapped.impl_id,
            "impls": {mapped.impl_id: impl_block},
        }
        result.changed = True
        result.inserted_device = True
        return result

    device_entry = engine_entry[mapped.device_name]
    impls = device_entry.setdefault("impls", {})

    if mapped.impl_id not in impls:
        impls[mapped.impl_id] = impl_block
        result.changed = True
        result.inserted_impl = True
        if mapped.is_default:
            if device_entry.get("defaultImpl") != mapped.impl_id:
                device_entry["defaultImpl"] = mapped.impl_id
                result.updated_paths.append(("defaultImpl",))
        return result

    existing_impl = impls[mapped.impl_id]
    for path in sorted(owned_impl_paths):
        if not _path_in_dict(impl_block, path):
            continue
        gen_value = _pluck_path(impl_block, path)
        cur_value = _get_path(existing_impl, path)
        if cur_value is _MISSING or not _equal(cur_value, gen_value):
            _set_path(existing_impl, path, gen_value)
            result.changed = True
            result.updated_paths.append(path)

    if mapped.is_default and device_entry.get("defaultImpl") != mapped.impl_id:
        device_entry["defaultImpl"] = mapped.impl_id
        result.changed = True
        result.updated_paths.append(("defaultImpl",))

    return result


def set_default_engine(doc: Any, model_name: str, engine: str) -> bool:
    """Set models.<m>.defaultEngine at the top of the model block.

    Returns True if the file changed. Idempotent.
    """
    models = _ensure_models_map(doc)
    if model_name not in models:
        return False
    entry = models[model_name]
    if entry.get("defaultEngine") == engine:
        return False
    entry["defaultEngine"] = engine
    return True


def format_path(path: PathTuple) -> str:
    return ".".join(path)


def format_change_summary(result: MergeResult) -> str:
    if result.inserted_model:
        return "inserted new model entry"
    if result.inserted_engine:
        return "inserted new engine block"
    if result.inserted_device:
        return "inserted new device block"
    if result.inserted_impl:
        return "inserted new impl block"
    if not result.changed:
        return "no change"
    keys = ", ".join(format_path(p) for p in result.updated_paths)
    return f"updated {len(result.updated_paths)} keys ({keys})"


__all__: List[str] = [
    "MergeResult",
    "merge_spec",
    "set_default_engine",
    "format_change_summary",
    "format_path",
]
