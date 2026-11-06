# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Run vLLM's structured-output benchmark and check that the grammar applied.

    python structured_output_client.py --script <benchmark_serving_structured_output.py>
        [--chat-template-kwargs JSON] <benchmark arguments>

The benchmark script is vLLM's own (fetched into the benchmark venv), loaded
from ``--script`` and run unchanged except for two hooks:

* ``--chat-template-kwargs`` renders every prompt as a chat turn with these
  template arguments: xgrammar_bench prompts (already chat-templated by the
  script) get them added, and the raw-text prompts of the other datasets
  become a single user turn. With a reasoning parser on the server, vLLM
  constrains the output only after the reasoning has ended, judged by the
  prompt; a prompt that leaves the model thinking (a raw prompt, or a template
  whose thinking is on by default) leaves the whole output unconstrained, so
  the driver passes the arguments that turn thinking off for such models.
* The requests the script picks for structured output are recorded, and once
  it has written its result file, ``tt_structured_output`` is added with what
  the structured outputs show: ``grammar_active_rate`` (a JSON object from
  the first non-whitespace token on, which an applied JSON grammar forces),
  ``complete_json_rate`` (exactly one JSON object) and ``schema_valid_rate``
  (that object satisfies its schema). The script's own ``correct_rate`` only
  looks for a ``{...}`` substring anywhere, so prose wrapped around a markdown
  code block passes it.

Self-contained on purpose: the driver launches it by path from the benchmark
venv's interpreter.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

JSON_DATASETS = ("json", "json-unique", "xgrammar_bench")


class ChatTemplateTokenizer:
    """``tokenizer`` with ``chat_template_kwargs`` added to every chat render."""

    def __init__(self, tokenizer: Any, chat_template_kwargs: Dict[str, Any]) -> None:
        self._tokenizer = tokenizer
        self._chat_template_kwargs = dict(chat_template_kwargs)

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any:
        return self._tokenizer.apply_chat_template(
            conversation, **{**kwargs, **self._chat_template_kwargs}
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)


class RecordingRandom:
    """The script's ``random`` module, recording the indices ``sample`` picks.

    The script picks its structured-output requests with one
    ``random.sample(range(n), k)`` and keeps them to itself."""

    def __init__(self, random_module: Any) -> None:
        self._random = random_module
        self.samples: List[List[int]] = []

    def sample(self, population: Any, k: int, **kwargs: Any) -> List[Any]:
        picked = self._random.sample(population, k, **kwargs)
        self.samples.append(list(picked))
        return picked

    def __getattr__(self, name: str) -> Any:
        return getattr(self._random, name)


def chat_sample_requests(sample_requests: Any, chat_template_kwargs: Dict[str, Any]):
    """Wrap the script's ``sample_requests`` to render chat prompts."""

    def wrapped(tokenizer: Any, args: argparse.Namespace) -> List[Any]:
        chat = ChatTemplateTokenizer(tokenizer, chat_template_kwargs)
        requests = sample_requests(chat, args)
        if args.dataset != "xgrammar_bench":
            for request in requests:
                request.prompt = chat.apply_chat_template(
                    [{"role": "user", "content": request.prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                request.prompt_len = len(tokenizer(request.prompt).input_ids)
        return requests

    return wrapped


def _json_object(text: str) -> Optional[Any]:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _schema_valid(schema: Any, value: Any) -> bool:
    import jsonschema

    if isinstance(schema, str):
        schema = json.loads(schema)
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError:
        return False
    return True


def output_checks(
    outputs: Sequence[Dict[str, Any]],
    structured: Sequence[int],
    schemas: Sequence[Any],
    dataset: str,
) -> Dict[str, Any]:
    """Rates over the structured requests' outputs (``None`` without any, or
    for a non-JSON dataset). ``outputs`` are the script's ``outputs`` records,
    in request order; ``schemas`` the requests' schemas, in the same order."""
    checks: Dict[str, Any] = {
        "dataset": dataset,
        "structured_requests": len(structured),
        "grammar_active_rate": None,
        "complete_json_rate": None,
        "schema_valid_rate": None,
    }
    if not structured or dataset not in JSON_DATASETS:
        return checks
    active = complete = valid = 0
    for index in structured:
        text = outputs[index].get("generated") or ""
        active += text.lstrip().startswith("{")
        value = _json_object(text)
        if value is None:
            continue
        complete += 1
        valid += _schema_valid(schemas[index], value)
    n = len(structured)
    checks.update(
        grammar_active_rate=active / n,
        complete_json_rate=complete / n,
        schema_valid_rate=valid / n,
    )
    return checks


def _load_script(path: Path) -> Any:
    # The script imports backend_request_func from its own directory.
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str]) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--script", required=True, type=Path)
    pre.add_argument("--chat-template-kwargs", type=json.loads, default=None)
    ours, rest = pre.parse_known_args(argv)

    script = _load_script(ours.script)
    args = script.create_argument_parser().parse_args(rest)
    if not args.save_results or not args.result_filename:
        raise SystemExit("--save-results and --result-filename are required")

    recorder = RecordingRandom(script.random)
    script.random = recorder
    requests: List[Any] = []
    sample_requests = script.sample_requests
    if ours.chat_template_kwargs is not None:
        sample_requests = chat_sample_requests(
            sample_requests, ours.chat_template_kwargs
        )

    def recorded_sample_requests(tokenizer: Any, args: argparse.Namespace):
        requests.extend(sample_requests(tokenizer, args))
        return requests

    script.sample_requests = recorded_sample_requests
    script.main(args)

    if len(recorder.samples) != 1:
        raise RuntimeError(
            f"expected one structured-output selection, saw {len(recorder.samples)}"
        )
    path = Path(args.result_filename)
    if args.result_dir:
        path = Path(args.result_dir) / path
    result = json.loads(path.read_text())
    checks = output_checks(
        result["outputs"],
        sorted(recorder.samples[0]),
        [request.schema for request in requests],
        args.dataset,
    )
    checks["chat_template_kwargs"] = ours.chat_template_kwargs
    result["tt_structured_output"] = checks
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=4))
    os.replace(tmp, path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
