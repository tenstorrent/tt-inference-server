import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

REPORTS_SCHEMA_PATH = Path(__file__).with_name("reports-schema.json")
REPORTS_SCHEMA_VERSION = "https://json-schema.org/draft/2020-12/schema"


def _import_jsonschema():
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError(
            "Report schema validation requires the 'jsonschema' package. "
            "Install the workflow dependencies that include it before running "
            "workflows/run_reports.py."
        ) from exc
    return jsonschema


@lru_cache(maxsize=1)
def load_reports_schema() -> Dict[str, Any]:
    with REPORTS_SCHEMA_PATH.open("r", encoding="utf-8") as schema_file:
        return json.load(schema_file)


def _deduplicate_schemas(schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique_schemas = []
    seen = set()
    for schema in schemas:
        schema_key = json.dumps(schema, sort_keys=True)
        if schema_key not in seen:
            unique_schemas.append(schema)
            seen.add(schema_key)
    return unique_schemas


def _infer_schema(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {key: _infer_schema(item) for key, item in value.items()},
            "required": list(value.keys()),
            "additionalProperties": False,
        }

    if isinstance(value, list):
        if not value:
            item_schema: Dict[str, Any] = {}
        else:
            item_schemas = _deduplicate_schemas([_infer_schema(item) for item in value])
            item_schema = (
                item_schemas[0] if len(item_schemas) == 1 else {"anyOf": item_schemas}
            )
        return {
            "type": "array",
            "items": item_schema,
        }

    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}

    raise TypeError(f"Unsupported value for schema generation: {type(value)!r}")


def _generic_object_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
    }


def _apply_report_schema_overrides(schema: Dict[str, Any]) -> Dict[str, Any]:
    properties = schema.get("properties", {})

    if "metadata" in properties:
        metadata_properties = properties["metadata"].get("properties", {})
        if "server_mode" in metadata_properties:
            metadata_properties["server_mode"] = {
                "type": "string",
                "enum": ["API", "docker"],
            }
        for field_name in ("tt_metal_commit", "vllm_commit"):
            if field_name in metadata_properties:
                metadata_properties[field_name] = {
                    "type": ["string", "null"],
                }

    if "evals" in properties:
        properties["evals"] = {
            "oneOf": [
                {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {
                                "type": "object",
                                "additionalProperties": True,
                            }
                        ]
                    },
                },
                _generic_object_schema(),
            ]
        }

    if "stress_tests" in properties:
        properties["stress_tests"] = {
            "type": ["array", "object", "null"],
            "items": _generic_object_schema(),
            "additionalProperties": True,
        }

    if "benchmarks_summary" in properties:
        properties["benchmarks_summary"] = {
            "type": "array",
            "items": _generic_object_schema(),
        }

    if "benchmarks" in properties:
        properties["benchmarks"] = {
            "type": "array",
            "items": _generic_object_schema(),
        }

    if "aiperf_benchmarks_detailed" in properties:
        properties["aiperf_benchmarks_detailed"] = {
            "type": "array",
            "items": _generic_object_schema(),
        }

    if "server_tests" in properties:
        properties["server_tests"] = {
            "type": "array",
            "items": _generic_object_schema(),
        }

    if "parameter_support_tests" in properties:
        properties["parameter_support_tests"] = {
            "oneOf": [
                _generic_object_schema(),
                {
                    "type": "array",
                    "items": _generic_object_schema(),
                },
            ]
        }

    if "spec_tests" in properties:
        properties["spec_tests"] = _generic_object_schema()

    if "benchmark_target_evaluation" in properties:
        properties["benchmark_target_evaluation"] = _generic_object_schema()

    if "acceptance_blockers" in properties:
        properties["acceptance_blockers"] = {
            "type": "object",
            "additionalProperties": {
                "type": "string",
            },
        }

    return schema


def generate_reports_schema(report_data: Dict[str, Any]) -> Dict[str, Any]:
    inferred_schema = _infer_schema(report_data)
    schema = {
        "$schema": REPORTS_SCHEMA_VERSION,
        "title": "Workflow Reports Output",
        "description": (
            "Schema for report_data_*.json generated by workflows/run_reports.py."
        ),
        **inferred_schema,
    }
    return _apply_report_schema_overrides(schema)


def _load_report_data(
    report_source: Union[Dict[str, Any], str, Path],
) -> Dict[str, Any]:
    if isinstance(report_source, dict):
        return report_source

    report_path = Path(report_source)
    with report_path.open("r", encoding="utf-8") as report_file:
        return json.load(report_file)


def write_reports_schema(
    report_source: Union[Dict[str, Any], str, Path],
    output_path: Union[str, Path] = REPORTS_SCHEMA_PATH,
) -> Path:
    """Generate a schema from report JSON and write it to disk."""
    schema_path = Path(output_path)
    report_data = _load_report_data(report_source)
    generated_schema = generate_reports_schema(report_data)
    schema_path.write_text(
        f"{json.dumps(generated_schema, indent=2)}\n",
        encoding="utf-8",
    )
    load_reports_schema.cache_clear()
    return schema_path


def _format_validation_error(error: Exception) -> str:
    absolute_path = getattr(error, "absolute_path", [])
    location = " -> ".join(str(part) for part in absolute_path) or "<root>"
    message = getattr(error, "message", str(error))
    return f"report_data schema validation failed at {location}: {message}"


def validate_report_data(
    report_data: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> None:
    jsonschema = _import_jsonschema()
    active_schema = load_reports_schema() if schema is None else schema

    try:
        jsonschema.validate(instance=report_data, schema=active_schema)
    except jsonschema.ValidationError as exc:
        logger.error(_format_validation_error(exc))
        # TODO: enable raising after schema is aligned with upstream and downstream consumers
        # raise RuntimeError(_format_validation_error(exc)) from exc


def validate_report_file(report_file: Path) -> None:
    try:
        report_data = json.loads(report_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse generated report JSON at {report_file}: {exc}"
        ) from exc

    validate_report_data(report_data)
