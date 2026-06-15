# SPDX-License-Identifier: Apache-2.0
"""Custom vLLM tool-call parser for Tenstorrent-served Qwen3.6-27B.

WHY THIS EXISTS
---------------
The model's chat template prescribes the qwen3 XML tool format
(`<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`), but on
device (validated greedily, thinking-off) this experimental bring-up reliably emits the function
tag correctly yet writes the PARAMETER in an attribute dialect:

    <tool_call>
    <function=get_current_weather>
    <parameter_name="location">Boston</parameter>
    </function>
    </tool_call>

The shipped `qwen3_xml` and `qwen3_coder` parsers both look for the literal `<parameter=` prefix,
so they extract the function name but find NO parameters -> `tool_calls: []`. This parser accepts
BOTH dialects:
  * `<parameter_name="KEY">VALUE</parameter>`   (what the model emits)
  * `<parameter=KEY>\nVALUE\n</parameter>`       (the template's prescribed form)
so tool calling works in either case.

MAKING tool_choice="auto" DRIFT-PROOF (the OpenCode fix)
-------------------------------------------------------
Clients like OpenCode send `tool_choice: "auto"` and cannot be made to send guided/named tool
choice. On this bring-up the SERVED traced/paged decode drifts, and `auto` is decoded WITHOUT any
grammar (vLLM's `get_json_schema_from_tools` returns None for "auto"), so the drift garbles the
tool-call bytes -> `tool_calls: []`, and in streaming the half-open `<tool_call>` makes vLLM emit
nothing at all ("no output"). `tool_choice: "required"`/named are bulletproof precisely because
they attach a JSON-schema grammar (host-side xgrammar logit masking, immune to device drift).

`adjust_request` below closes that gap for "auto" WITHOUT forcing a tool call every turn: it
attaches a STRUCTURAL TAG so text stays free until the model emits `<tool_call>`, at which point
the body is constrained to a valid tool-call JSON object `{"name": ..., "arguments": {...}}`. The
grammar therefore forces clean JSON inside `<tool_call>` regardless of decode drift, and `_parse`
reads that JSON first (XML dialects remain as a fallback for the un-grammared path).

Load it (no image rebuild) with:
    --tool-parser-plugin <path-to-this-file> --tool-call-parser qwen36_xml --enable-auto-tool-choice
"""
import json

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager

logger = init_logger(__name__)

_TOOLCALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
# function name: <function=NAME>  OR  <function name="NAME">
_FUNC = re.compile(r"<function(?:=|\s+name=\"?)([^>\"]+)\"?\s*>", re.DOTALL)
# parameters, model dialect: <parameter_name="KEY">VALUE</parameter>
_PARAM_ATTR = re.compile(r"<parameter_name=\"([^\"]+)\"\s*>(.*?)</parameter>", re.DOTALL)
# parameters, template dialect: <parameter=KEY>\nVALUE\n</parameter>
_PARAM_EQ = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)


@ToolParserManager.register_module("qwen36_xml")
class Qwen36XMLToolParser(ToolParser):
    """Lenient qwen3-XML parser accepting the model's `<parameter_name="k">v</parameter>` dialect
    in addition to the template's `<parameter=k>v</parameter>`."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    # ---- gate a tool-call grammar onto "auto" so it is drift-proof --------------------------------
    def _toolcall_schema(self, request):
        """Union JSON schema for ONE tool call: {"name": <one tool>, "arguments": <its params>}."""
        options = []
        for tool in request.tools or []:
            try:
                fn = tool.function
                params = fn.parameters or {"type": "object", "properties": {}}
            except Exception:
                continue
            options.append(
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": [fn.name]},
                        "arguments": params,
                    },
                    "required": ["name", "arguments"],
                }
            )
        if not options:
            return None
        return options[0] if len(options) == 1 else {"anyOf": options}

    def adjust_request(self, request):
        """For tool_choice="auto" (what OpenCode sends), attach a structural-tag grammar that
        constrains anything inside <tool_call>...</tool_call> to a valid tool-call JSON object,
        while leaving normal prose unconstrained. required/named keep the base behaviour."""
        try:
            tool_choice = getattr(request, "tool_choice", None)
            if tool_choice not in ("auto", None):
                # "required" / named / forced-function: the base parser already attaches the
                # JSON-schema grammar that makes those modes drift-proof. Don't interfere.
                return super().adjust_request(request)
            if not getattr(request, "tools", None):
                return request
            # Respect an explicit structured-output / response_format request from the client.
            if getattr(request, "structured_outputs", None) is not None:
                return request
            if getattr(request, "response_format", None) is not None:
                return request
            schema = self._toolcall_schema(request)
            if schema is None:
                return request
            s_tag = {
                "type": "structural_tag",
                "structures": [
                    {"begin": "<tool_call>", "schema": schema, "end": "</tool_call>"}
                ],
                "triggers": ["<tool_call>"],
            }
            request.structured_outputs = StructuredOutputsParams(
                structural_tag=json.dumps(s_tag)
            )
        except Exception as e:  # never break a request because the grammar couldn't be built
            logger.warning("qwen36_xml: skipped auto-mode tool grammar: %s", e)
        return request

    # ---- schema-aware value coercion (string -> int/float/bool/obj per the tool's param type) ----
    def _type_map(self, request):
        out = {}
        try:
            for tool in request.tools or []:
                fn = tool.function
                props = (fn.parameters or {}).get("properties", {})
                out[fn.name] = {k: (v or {}).get("type", "string") for k, v in props.items()}
        except Exception:
            pass
        return out

    @staticmethod
    def _coerce(value, ptype):
        v = value.strip()
        try:
            if ptype in ("integer", "number"):
                return json.loads(v)
            if ptype == "boolean":
                return v.lower() == "true"
            if ptype in ("object", "array"):
                return json.loads(v)
        except Exception:
            return value
        return value

    @staticmethod
    def _parse_json_block(block):
        """Parse a JSON tool call (`{"name": ..., "arguments"|"parameters": {...}}`) from a
        <tool_call> body. This is what the structural-tag grammar forces under tool_choice="auto".
        Returns (name, args) or None when the body is not such a JSON object."""
        s = block.strip()
        if not s.startswith("{"):
            return None
        try:
            obj = json.loads(s)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            return None
        args = obj.get("arguments")
        if args is None:
            args = obj.get("parameters", {})
        if not isinstance(args, dict):
            return None
        return (name, args)  # values are already typed by the grammar; no coercion needed

    def _parse(self, text, request):
        type_map = self._type_map(request)
        calls = []
        for block in _TOOLCALL.findall(text):
            # Preferred: grammar-forced JSON body (tool_choice="auto" + structural tag).
            jcall = self._parse_json_block(block)
            if jcall is not None:
                calls.append(jcall)
                continue
            # Fallback: XML dialects (un-grammared path / template form).
            fm = _FUNC.search(block)
            if not fm:
                continue
            name = fm.group(1).strip()
            args = {}
            for k, v in _PARAM_ATTR.findall(block):
                args[k.strip()] = v
            for k, v in _PARAM_EQ.findall(block):
                args.setdefault(k.strip(), v.strip())
            ptypes = type_map.get(name, {})
            args = {k: self._coerce(v, ptypes.get(k, "string")) for k, v in args.items()}
            calls.append((name, args))
        return calls

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        if "<tool_call>" not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        try:
            parsed = self._parse(model_output, request)
        except Exception as e:  # never crash the server on a parse error
            logger.exception("qwen36_xml tool parse failed: %s", e)
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        if not parsed:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        tool_calls = [
            ToolCall(
                id=make_tool_call_id(),
                type="function",
                function=FunctionCall(name=name, arguments=json.dumps(args, ensure_ascii=False)),
            )
            for name, args in parsed
        ]
        pre = model_output.split("<tool_call>", 1)[0]
        # strip stray thinking scaffold the model sometimes emits before the call
        pre = pre.replace("<think>", "").replace("</think>", "").strip()
        return ExtractedToolCallInformation(
            tools_called=True, tool_calls=tool_calls, content=pre or None
        )

    def extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    ):
        # Minimal, crash-safe streaming: stream plain text until a tool call opens, buffer while
        # inside it, and emit the fully-parsed tool call(s) once </tool_call> completes. (The
        # non-streaming path above is the primary one; this just keeps streaming clients functional.)
        try:
            if "<tool_call>" not in current_text:
                return DeltaMessage(content=delta_text) if delta_text else None
            # inside/after a tool call: only act when a call has just completed
            if "</tool_call>" not in current_text:
                return None
            if previous_text and current_text.count("</tool_call>") == previous_text.count("</tool_call>"):
                return None  # no NEW completed call in this delta
            info = self.extract_tool_calls(current_text, request)
            if not info.tools_called:
                return None
            from vllm.entrypoints.openai.engine.protocol import DeltaFunctionCall, DeltaToolCall

            deltas = [
                DeltaToolCall(
                    index=i,
                    id=tc.id,
                    type="function",
                    function=DeltaFunctionCall(name=tc.function.name, arguments=tc.function.arguments),
                )
                for i, tc in enumerate(info.tool_calls)
            ]
            return DeltaMessage(tool_calls=deltas)
        except Exception:
            return None


# Eager registration (belt-and-suspenders). The @register_module decorator registers LAZILY by
# module path; assigning into the eager registry guarantees `get_tool_parser("qwen36_xml")` resolves
# regardless of how vLLM loads this file.
ToolParserManager.tool_parsers["qwen36_xml"] = Qwen36XMLToolParser
