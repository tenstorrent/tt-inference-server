# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Case selection, schema wrapping, and the request prompt are ported from the
# Kimi Vendor Verifier (https://github.com/MoonshotAI/Kimi-Vendor-Verifier,
# tests/tool_call_json_schema/validator.py), Copyright (c) 2026 Moonshot AI,
# MIT License. See test_fixtures/tool_call_schema_cases/{SOURCE.md,LICENSE}.

"""Tool-call JSON-schema conformance: case loading, requests, and grading.

Each case is a JSON Schema from walle's validator test data. The schema is
wrapped as the ``value`` property of the ``parameters`` object of a single
``strict`` tool, the model is asked to call that tool, and the returned
``function.arguments`` are validated against the wrapped schema with
``jsonschema`` (Draft 2020-12). Every case runs in non-streaming and streaming
mode; streamed tool-call deltas are reassembled before validation.

The pytest suite (``llm_module/test_tool_call_json_schema.py``) drives this
module; ``ToolCallSchemaConformanceTest`` grades its report against a pass-rate
threshold. Everything here is network-free except :class:`ChatCompletionsClient`,
so it is unit-testable with a fake sender.
"""

from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

DEFAULT_CASE_DIR = (
    Path(__file__).resolve().parents[1]
    / "test_fixtures"
    / "tool_call_schema_cases"
    / "validator_cases"
)

TOOL_NAME = "kvv_walle_case"
REQUEST_MODES = ("non-stream", "stream")

SELECTIONS = ("all", "explicit", "object")
# Tenstorrent deployments have no guided decoding, so "required" is rejected
# there; "auto" lets the model answer in text, which counts as a failure.
TOOL_CHOICES = ("auto", "required")
# none: send no thinking field. kimi: {"thinking": {"type": ...}}.
# opensource: {"chat_template_kwargs": {"thinking": bool}}.
THINK_MODES = ("none", "kimi", "opensource")

DEFAULT_SELECTION = "all"
DEFAULT_TOOL_CHOICE = "auto"
DEFAULT_THINK_MODE = "none"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_CASE_RETRIES = 0
DEFAULT_WORKERS = 16
DEFAULT_REQUEST_TIMEOUT_S = 600.0
# Pause between attempts of the same case (the source suite used
# pytest-rerunfailures with --reruns-delay 2).
DEFAULT_RETRY_DELAY_S = 2.0
# Sampling params sent with default_sampling: the fixed values from the Kimi
# K2.7 Code quickstart (https://platform.kimi.ai/docs/guide/kimi-k2-7-code-quickstart)
# plus top_k 50. Without them some deployments sample from the full
# distribution (or reuse one fixed seed), which shows up as corrupted tool
# names and JSON rather than as model errors.
DEFAULT_SAMPLING_PARAMS: Dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 50,
    "n": 1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}

PASSED = "passed"
FAILED = "failed"

# Failure causes, recorded per case and counted in the report.
CAUSE_NO_TOOL_CALL = "no_tool_call"
CAUSE_WRONG_TOOL_NAME = "wrong_tool_name"
CAUSE_INVALID_JSON = "invalid_json"
CAUSE_SCHEMA_VIOLATION = "schema_violation"
CAUSE_HTTP_ERROR = "http_error"
CAUSE_CONNECTION_ERROR = "connection_error"
CAUSE_MALFORMED_RESPONSE = "malformed_response"
CAUSE_LOCAL_SCHEMA_ERROR = "local_schema_error"

TOOL_KEYWORDS = (
    "tool",
    "function",
    "tool_call",
    "function_call",
    "arguments",
    "parameters",
)

# Verbatim from the source suite, including its missing space in
# "minimumruntime", so pass rates stay comparable with Kimi Vendor Verifier runs.
USER_PROMPT = (
    f"Call the {TOOL_NAME} tool exactly once with minimum"
    "runtime arguments that satisfy its parameter schema, "
    "try your best to create the arguments. "
    "Do not copy or describe the JSON Schema itself. Do not "
    "include schema keywords like type, properties, required, "
    "or additionalProperties unless the schema explicitly "
    "requires them as argument property names. If the schema "
    "defines a top-level value argument, provide the minimal "
    "valid value for it. Always include every required "
    "property. Respect minItems, minProperties, enum, const, minimum, and "
    "minLength constraints. Prefer empty arrays and empty "
    "objects only when those constraints allow them. Do not "
    "answer with plain text."
)
TOOL_DESCRIPTION = (
    "Submit minimal JSON arguments that validate against this JSON Schema."
)


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatorCase:
    """One line of a walle ``valid.jsonl`` file."""

    suite: str
    line: int
    schema_text: str

    @property
    def case_id(self) -> str:
        return f"{self.suite}:{self.line}"


@dataclass(frozen=True)
class SelectedCase:
    """A runnable case: the wrapped tool ``parameters`` schema to send."""

    case: ValidatorCase
    schema: Any
    selection_reason: str

    @property
    def case_id(self) -> str:
        return self.case.case_id

    @property
    def uses_ref(self) -> bool:
        """Whether the sent schema types anything through ``$ref``.

        Servers that rebuild typed arguments from a text tool-call format
        (MiniMax's XML) must follow ``$ref`` -> ``$defs`` to coerce values;
        see ``SuiteSettings.exclude_ref_schemas``.
        """
        return bool(_collect_refs(self.schema))


def load_cases(case_dir: Path = DEFAULT_CASE_DIR) -> List[ValidatorCase]:
    """Read every non-blank line of ``<case_dir>/<suite>/valid.jsonl``."""
    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"case directory not found: {case_dir}")

    cases: List[ValidatorCase] = []
    for suite_path in sorted(p for p in case_dir.iterdir() if p.is_dir()):
        valid_path = suite_path / "valid.jsonl"
        with valid_path.open(encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                schema_text = raw_line.strip()
                if schema_text:
                    cases.append(
                        ValidatorCase(suite_path.name, line_number, schema_text)
                    )
    return cases


def _has_non_finite_number(value: Any) -> bool:
    if isinstance(value, float):
        return value != value or value in (float("inf"), float("-inf"))
    if isinstance(value, list):
        return any(_has_non_finite_number(item) for item in value)
    if isinstance(value, dict):
        return any(_has_non_finite_number(item) for item in value.values())
    return False


def _parse_for_transport(case: ValidatorCase) -> Tuple[bool, Any, str]:
    try:
        schema = json.loads(case.schema_text)
    except json.JSONDecodeError as exc:
        return False, None, f"schema text is not valid JSON: {exc.msg}"
    if _has_non_finite_number(schema):
        return False, None, "schema contains non-finite numeric value"
    return True, schema, ""


def _schema_text_has_tool_keyword(schema: Any) -> bool:
    text = json.dumps(schema, ensure_ascii=False).lower()
    return any(keyword in text for keyword in TOOL_KEYWORDS)


def _is_object_parameter_schema(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    schema_type = schema.get("type")
    return (
        schema_type == "object"
        or (isinstance(schema_type, list) and "object" in schema_type)
    ) and isinstance(schema.get("properties"), dict)


def _collect_refs(obj: Any) -> List[str]:
    refs: List[str] = []
    if isinstance(obj, dict):
        if "$ref" in obj:
            refs.append(obj["$ref"])
        for value in obj.values():
            refs.extend(_collect_refs(value))
    elif isinstance(obj, list):
        for item in obj:
            refs.extend(_collect_refs(item))
    return refs


def _rewrite_root_refs(obj: Any, target_ref: str) -> Any:
    """Rewrite JSON Schema root self-references (``"$ref": "#"``)."""
    if isinstance(obj, dict):
        return {
            key: (
                target_ref
                if key == "$ref" and value == "#"
                else _rewrite_root_refs(value, target_ref)
            )
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_rewrite_root_refs(item, target_ref) for item in obj]
    return obj


def wrap_schema_as_parameter_property(schema: Any) -> Any:
    """Use the case schema as the schema of a required ``value`` argument.

    Tool ``parameters`` must be an object, but most cases are not, so every
    case becomes ``{"type": "object", "properties": {"value": <case>}, ...}``.
    ``$defs`` are hoisted to the wrapper and root ``$ref: "#"`` recursion is
    redirected to a ``$defs`` entry holding the original schema.
    """
    if not isinstance(schema, dict):
        return schema

    wrapped: Dict[str, Any] = {
        "type": "object",
        "required": ["value"],
        "additionalProperties": False,
    }

    if "#" in _collect_refs(schema):
        defs = dict(schema.get("$defs", {}))
        def_name = "__case_schema"
        while def_name in defs:
            def_name = f"_{def_name}"
        target_ref = f"#/$defs/{def_name}"
        rewritten = _rewrite_root_refs(schema, target_ref)
        defs.update(rewritten.get("$defs", {}))
        defs[def_name] = {
            k: v for k, v in rewritten.items() if k not in ("$defs", "$id")
        }
        wrapped["properties"] = {"value": {"$ref": target_ref}}
        wrapped["$defs"] = defs
    else:
        wrapped["properties"] = {
            "value": {k: v for k, v in schema.items() if k not in ("$defs", "$id")}
        }
        if "$defs" in schema:
            wrapped["$defs"] = schema["$defs"]

    if "$id" in schema:
        wrapped["$id"] = schema["$id"]
    return wrapped


def _is_primitive_type(schema_type: Any) -> bool:
    if isinstance(schema_type, str):
        return schema_type in ("string", "number", "integer", "boolean", "null")
    if isinstance(schema_type, list):
        return any(_is_primitive_type(item) for item in schema_type)
    return False


def _has_non_recursive_option(obj: Any) -> bool:
    """Return True if ``obj`` allows a value that does not recurse."""
    if isinstance(obj, dict):
        if _is_primitive_type(obj.get("type")):
            return True
        if "anyOf" in obj:
            return any(_has_non_recursive_option(s) for s in obj["anyOf"])
        if "oneOf" in obj:
            return any(_has_non_recursive_option(s) for s in obj["oneOf"])
        if "enum" in obj:
            return True
        return any(_has_non_recursive_option(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_non_recursive_option(item) for item in obj)
    return False


def _has_recursive_ref_without_termination(schema: Any) -> bool:
    """Detect a required, self-referencing ``$defs`` entry with no way out."""
    if not isinstance(schema, dict):
        return False
    defs = schema.get("$defs", {})
    for ref in _collect_refs(schema):
        if not ref.startswith("#/$defs/"):
            continue
        def_name = ref.split("/")[-1]
        if def_name not in defs:
            continue
        def_schema = defs[def_name]
        for inner_ref in _collect_refs(def_schema):
            if inner_ref != ref or _has_non_recursive_option(def_schema):
                continue
            required = set(def_schema.get("required", []))
            for prop_name, prop_schema in def_schema.get("properties", {}).items():
                if ref in _collect_refs(prop_schema) and prop_name in required:
                    return True
    return False


def _has_exotic_property_keys(schema: Any) -> bool:
    """Detect property keys that tool-call parsers mangle.

    Leading/trailing whitespace and empty keys get stripped, and walle's
    literal escape sequences in keys (``\\n``, ``\\t``, ...) never round-trip.
    """
    if not isinstance(schema, dict):
        return False
    for key in schema.get("properties", {}).keys():
        if key != key.strip() or key == "":
            return True
        if any(seq in key for seq in (r"\n", r"\t", r"\r", r"\b", r"\f")):
            return True
    return False


def _strip_keyword_recursive(obj: Any, keyword: str) -> Any:
    if isinstance(obj, dict):
        return {
            k: _strip_keyword_recursive(v, keyword)
            for k, v in obj.items()
            if k != keyword
        }
    if isinstance(obj, list):
        return [_strip_keyword_recursive(item, keyword) for item in obj]
    return obj


def _schema_shape(schema: Any) -> str:
    if not isinstance(schema, dict):
        return type(schema).__name__
    if not schema:
        return "empty_parameter_schema"
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return "union_parameter_schema"
    if isinstance(schema_type, str):
        return f"{schema_type}_parameter_schema"
    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in schema:
            return f"{keyword}_parameter_schema"
    if "$ref" in schema:
        return "ref_parameter_schema"
    return "schema_parameter_schema"


def classify_case(case: ValidatorCase) -> Tuple[Any, str]:
    """Return ``(wrapped_schema, selection_reason)``.

    ``wrapped_schema`` is ``None`` when the case is not runnable; the reason
    then says why (``unsupported_by_transport: ...`` / ``skipped_...``).
    """
    transportable, schema, reason = _parse_for_transport(case)
    if not transportable:
        return None, f"unsupported_by_transport: {reason}"

    # Skip checks run on the original schema, before wrapping changes its shape.
    if isinstance(schema, dict):
        if schema.get("type") == "string":
            min_len = schema.get("minLength")
            if min_len is not None and min_len > 1000:
                return None, "skipped_extreme_minlength_not_supported"
        if _has_exotic_property_keys(schema):
            return None, "skipped_exotic_property_keys"
        if _has_recursive_ref_without_termination(schema):
            return None, "skipped_recursive_ref"

    if _schema_text_has_tool_keyword(schema):
        selection_reason = "explicit_tool_keyword"
    elif _is_object_parameter_schema(schema):
        selection_reason = "object_parameter_schema"
    else:
        selection_reason = _schema_shape(schema)

    schema = wrap_schema_as_parameter_property(schema)
    # ``default`` destabilises decoders (required properties get omitted).
    schema = _strip_keyword_recursive(schema, "default")
    return schema, selection_reason


def select_cases(
    cases: Sequence[ValidatorCase],
    *,
    selection: str = DEFAULT_SELECTION,
    max_cases: Optional[int] = None,
    exclude_ref: bool = False,
) -> List[SelectedCase]:
    """Keep the runnable cases matching ``selection`` (all / explicit / object).

    ``exclude_ref`` also drops every case whose schema uses ``$ref``
    (``SelectedCase.uses_ref``), before ``max_cases`` is counted.
    """
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}, got {selection!r}")
    selected: List[SelectedCase] = []
    for case in cases:
        if max_cases is not None and len(selected) >= max_cases:
            break
        schema, reason = classify_case(case)
        if schema is None:
            continue
        if selection == "explicit" and reason != "explicit_tool_keyword":
            continue
        if selection == "object" and reason != "object_parameter_schema":
            continue
        candidate = SelectedCase(case, schema, reason)
        if exclude_ref and candidate.uses_ref:
            continue
        selected.append(candidate)
    return selected


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SuiteSettings:
    """Knobs shared by the pytest suite and the spec-test wrapper."""

    tool_choice: str = DEFAULT_TOOL_CHOICE
    think_mode: str = DEFAULT_THINK_MODE
    thinking: bool = False
    selection: str = DEFAULT_SELECTION
    max_cases: Optional[int] = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    case_retries: int = DEFAULT_CASE_RETRIES
    workers: int = DEFAULT_WORKERS
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_S
    # Added to every request when set (see resolve_sampling_params).
    sampling_params: Optional[Dict[str, Any]] = None
    # Give every request a unique prompt prefix so none can be served from
    # the server's prefix cache.
    cache_bypass: bool = False
    # Do not send cases whose schema types values through $ref -> $defs (35 of
    # the 204 with the default selection). For servers whose tool-call parser
    # cannot resolve $ref, e.g. MiniMax: its tool calls are XML, so every value
    # arrives as text and the parser converts it to the schema's type, but the
    # MiniMax parsers in ai-dynamo 1.3.0 (dynamo-parsers 5.0.0) and vLLM 0.30.0
    # never follow $ref, so 3 comes back as "3" and an array as {"item": [...]}
    # even when the model wrote the right value. An upstream bug (GPU vLLM
    # returns the same output), not a model or device one.
    exclude_ref_schemas: bool = False

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("tool_choice", self.tool_choice, TOOL_CHOICES),
            ("think_mode", self.think_mode, THINK_MODES),
            ("selection", self.selection, SELECTIONS),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        if self.max_cases is not None and self.max_cases < 1:
            raise ValueError(f"max_cases must be >= 1, got {self.max_cases}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.case_retries < 0:
            raise ValueError(f"case_retries must be >= 0, got {self.case_retries}")
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")
        if self.request_timeout <= 0:
            raise ValueError(f"request_timeout must be > 0, got {self.request_timeout}")
        if self.sampling_params is not None and not isinstance(
            self.sampling_params, dict
        ):
            raise ValueError(
                f"sampling_params must be a dict, got {self.sampling_params!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Requests and responses
# --------------------------------------------------------------------------


def resolve_sampling_params(
    default_sampling: bool, overrides: Any = None
) -> Optional[Dict[str, Any]]:
    """The sampling params to send, or None to send none.

    ``overrides`` (a dict, or a JSON object as a string) is merged over
    DEFAULT_SAMPLING_PARAMS when ``default_sampling`` is set, and sent on its
    own otherwise. A null value is sent as JSON null (e.g. ``{"seed": null}``).
    """
    if isinstance(overrides, str):
        overrides = json.loads(overrides) if overrides.strip() else None
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError(f"sampling params must be a JSON object, got {overrides!r}")
    params = {
        **(DEFAULT_SAMPLING_PARAMS if default_sampling else {}),
        **(overrides or {}),
    }
    return params or None


def thinking_extra_body(thinking: bool, think_mode: str) -> Dict[str, Any]:
    """Top-level request fields that toggle thinking for ``think_mode``."""
    if think_mode == "none":
        return {}
    if think_mode == "opensource":
        return {"chat_template_kwargs": {"thinking": thinking}}
    return {"thinking": {"type": "enabled" if thinking else "disabled"}}


def build_request(
    schema: Any, settings: SuiteSettings, *, stream: bool
) -> Dict[str, Any]:
    """Chat-completions payload offering ``schema`` as the only tool.

    With ``settings.cache_bypass`` the tool description starts with a fresh
    id. Chat templates that render tool declarations before the messages
    (Kimi's does) then have no prompt prefix that can match one the server
    has cached.
    """
    description = TOOL_DESCRIPTION
    if settings.cache_bypass:
        description = f"[request {uuid.uuid4().hex}] {description}"
    payload: Dict[str, Any] = {
        "messages": [{"role": "user", "content": USER_PROMPT}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "description": description,
                    "parameters": schema,
                    "strict": True,
                },
            }
        ],
        "tool_choice": settings.tool_choice,
        "max_tokens": settings.max_tokens,
    }
    if stream:
        payload["stream"] = True
    payload.update(settings.sampling_params or {})
    payload.update(thinking_extra_body(settings.thinking, settings.think_mode))
    return payload


class ResponseError(Exception):
    """A response that could not be turned into tool calls."""

    def __init__(self, cause: str, message: str):
        super().__init__(message)
        self.cause = cause


def reassemble_stream(lines: Iterable[Any]) -> Dict[str, Any]:
    """Rebuild the first choice of a streamed chat completion from SSE lines.

    Mirrors ``_stream_chat_completion`` in ``test_vllm_qwen3_streaming.py``:
    ``tool_calls`` deltas are merged per ``index``, ``function.arguments`` is
    concatenated, and ``content`` / ``finish_reason`` are kept for messages.
    Raises :class:`ResponseError` on an in-stream error event or bad JSON.
    """
    content_parts: List[str] = []
    tool_calls: Dict[int, Dict[str, str]] = {}
    finish_reason = None
    chunks = 0

    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ResponseError(
                CAUSE_MALFORMED_RESPONSE,
                f"stream chunk is not valid JSON ({exc.msg}): {data}",
            ) from exc
        if not isinstance(chunk, dict):
            continue
        if chunk.get("error"):
            raise ResponseError(
                CAUSE_HTTP_ERROR, f"stream error event: {json.dumps(chunk)}"
            )
        chunks += 1
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}

        if isinstance(delta.get("content"), str):
            content_parts.append(delta["content"])

        for tc in delta.get("tool_calls") or []:
            slot = tool_calls.setdefault(
                int(tc.get("index") or 0), {"name": "", "arguments": ""}
            )
            fn = tc.get("function") or {}
            name = fn.get("name")
            # OpenAI-style deltas may split the name; some servers instead
            # repeat the full name on every delta, so don't double it up.
            if isinstance(name, str) and name and slot["name"] != name:
                slot["name"] += name
            if isinstance(fn.get("arguments"), str):
                slot["arguments"] += fn["arguments"]

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    return {
        "content": "".join(content_parts),
        "tool_calls": [tool_calls[i] for i in sorted(tool_calls)],
        "finish_reason": finish_reason,
        "chunks": chunks,
    }


def parse_completion(body: Any) -> Dict[str, Any]:
    """Normalise a non-streamed completion to :func:`reassemble_stream`'s shape."""
    if not isinstance(body, dict):
        raise ResponseError(
            CAUSE_MALFORMED_RESPONSE, f"response body is not an object: {body!r}"
        )
    choices = body.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ResponseError(
            CAUSE_MALFORMED_RESPONSE,
            f"response has no choices: {json.dumps(body, ensure_ascii=False)}",
        )
    message = choices[0].get("message") or {}
    tool_calls = []
    for tc in message.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        tool_calls.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
    return {
        "content": message.get("content") or "",
        "tool_calls": tool_calls,
        "finish_reason": choices[0].get("finish_reason"),
    }


def select_tool_arguments(completion: Dict[str, Any]) -> str:
    """Return the ``kvv_walle_case`` arguments string, or raise ResponseError."""
    tool_calls = completion.get("tool_calls") or []
    if not tool_calls:
        raise ResponseError(
            CAUSE_NO_TOOL_CALL,
            "response did not include tool_calls "
            f"(finish_reason={completion.get('finish_reason')}); "
            f"content={completion.get('content', '')!r}",
        )
    for call in tool_calls:
        if call.get("name") != TOOL_NAME:
            continue
        arguments = call.get("arguments")
        if arguments is None:
            raise ResponseError(
                CAUSE_INVALID_JSON,
                f"tool call {TOOL_NAME} did not include function.arguments",
            )
        if isinstance(arguments, str):
            return arguments
        return json.dumps(arguments, ensure_ascii=False)
    names = ", ".join(repr(call.get("name")) for call in tool_calls)
    raise ResponseError(
        CAUSE_WRONG_TOOL_NAME,
        f"response did not call {TOOL_NAME}; tool_calls names: {names}",
    )


def validate_arguments(schema: Any, arguments: str) -> Tuple[Optional[str], str]:
    """Validate ``arguments`` against ``schema``: ``(cause or None, message)``."""
    # Imported here so llm_module/conftest.py (shared by every llm_module
    # suite) can import this module's defaults without needing jsonschema.
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError, ValidationError

    try:
        instance = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return (
            CAUSE_INVALID_JSON,
            f"tool call arguments are not valid JSON ({exc.msg}); "
            f"arguments={arguments!r}",
        )
    try:
        Draft202012Validator(schema).validate(instance)
    except SchemaError as exc:
        return CAUSE_LOCAL_SCHEMA_ERROR, f"jsonschema rejected the case schema: {exc}"
    except ValidationError as exc:
        path = getattr(exc, "json_path", "$")
        return (
            CAUSE_SCHEMA_VIOLATION,
            f"tool call arguments do not match schema at {path}: {exc.message}; "
            f"arguments={arguments}",
        )
    return None, "tool call arguments matched schema"


# --------------------------------------------------------------------------
# Running cases
# --------------------------------------------------------------------------

# (payload, stream) -> HTTP response. Production uses ChatCompletionsClient.
Sender = Callable[[Dict[str, Any], bool], Any]


class ChatCompletionsClient:
    """POST payloads to a chat-completions endpoint, like ``api_client``.

    Same auth and model handling as the ``api_client`` fixture in
    ``test_fixtures/conftest.py`` (bearer token, ``model`` injected when
    missing), but returns the raw response so callers can read the HTTP
    status and stream it, and is safe to share across worker threads.
    """

    def __init__(
        self,
        endpoint_url: str,
        *,
        model_name: Optional[str] = None,
        bearer_token: Optional[str] = None,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_S,
    ):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if bearer_token:
            self.headers["Authorization"] = f"Bearer {bearer_token}"

    def __call__(self, payload: Dict[str, Any], stream: bool) -> requests.Response:
        if self.model_name and "model" not in payload:
            payload = {**payload, "model": self.model_name}
        return requests.post(
            self.endpoint_url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
            stream=stream,
        )


@dataclass(frozen=True)
class Attempt:
    """Outcome of one request for one case."""

    cause: Optional[str]
    message: str
    http_status: Optional[int] = None
    arguments: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.cause is None


def evaluate_once(
    send: Sender, selected: SelectedCase, mode: str, settings: SuiteSettings
) -> Attempt:
    """Send one request for ``selected`` in ``mode`` and grade the reply."""
    stream = mode == "stream"
    payload = build_request(selected.schema, settings, stream=stream)
    try:
        response = send(payload, stream)
    except requests.RequestException as exc:
        return Attempt(CAUSE_CONNECTION_ERROR, f"{type(exc).__name__}: {exc}")

    status = getattr(response, "status_code", None)
    try:
        if status is not None and status >= 400:
            return Attempt(
                CAUSE_HTTP_ERROR, f"HTTP {status}: {response.text}", http_status=status
            )
        if stream:
            # SSE is UTF-8 by definition, but servers often send a bare
            # ``text/event-stream`` content type, for which requests falls
            # back to ISO-8859-1 and garbles non-ASCII text (e.g. emoji
            # property names).
            response.encoding = "utf-8"
            completion = reassemble_stream(response.iter_lines(decode_unicode=True))
        else:
            completion = parse_completion(response.json())
        arguments = select_tool_arguments(completion)
    except ResponseError as exc:
        return Attempt(exc.cause, str(exc), http_status=status)
    except ValueError as exc:
        # Includes requests' JSONDecodeError for a non-JSON body. A consumed
        # stream has no .text left to show.
        body = "" if stream else f": {getattr(response, 'text', '')}"
        return Attempt(
            CAUSE_MALFORMED_RESPONSE,
            f"malformed response ({type(exc).__name__}: {exc}){body}",
            http_status=status,
        )
    except requests.RequestException as exc:
        # e.g. ChunkedEncodingError when the stream is cut mid-response.
        return Attempt(
            CAUSE_CONNECTION_ERROR, f"{type(exc).__name__}: {exc}", http_status=status
        )
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()

    cause, message = validate_arguments(selected.schema, arguments)
    return Attempt(cause, message, http_status=status, arguments=arguments)


@dataclass(frozen=True)
class CaseResult:
    """Final outcome of one (case, mode) after retries."""

    case_id: str
    suite: str
    line: int
    mode: str
    selection_reason: str
    status: str
    cause: Optional[str]
    message: str
    attempts: int
    http_status: Optional[int] = None
    arguments: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.status == PASSED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_case(
    send: Sender,
    selected: SelectedCase,
    mode: str,
    settings: SuiteSettings,
    *,
    retry_delay: float = DEFAULT_RETRY_DELAY_S,
) -> CaseResult:
    """Run one case, retrying up to ``settings.case_retries`` times on failure.

    The case passes if any attempt passes; otherwise the last attempt's
    failure is reported.
    """
    attempt = None
    attempts = 0
    for attempts in range(1, settings.case_retries + 2):
        attempt = evaluate_once(send, selected, mode, settings)
        if attempt.passed:
            break
        if attempts <= settings.case_retries and retry_delay > 0:
            time.sleep(retry_delay)
    return CaseResult(
        case_id=selected.case_id,
        suite=selected.case.suite,
        line=selected.case.line,
        mode=mode,
        selection_reason=selected.selection_reason,
        status=PASSED if attempt.passed else FAILED,
        cause=attempt.cause,
        message=attempt.message,
        attempts=attempts,
        http_status=attempt.http_status,
        arguments=attempt.arguments,
    )


def run_cases(
    send: Sender,
    tasks: Sequence[Tuple[SelectedCase, str]],
    settings: SuiteSettings,
    *,
    retry_delay: float = DEFAULT_RETRY_DELAY_S,
) -> List[CaseResult]:
    """Run ``(case, mode)`` tasks on ``settings.workers`` threads, in order."""
    if not tasks:
        return []
    workers = min(settings.workers, len(tasks))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(
                lambda task: run_case(
                    send, task[0], task[1], settings, retry_delay=retry_delay
                ),
                tasks,
            )
        )
