#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
"""Retry lm-eval samples whose response is empty (resps == [[""]]) via curl.

Optionally regrade retried responses with the same lm-eval task metric
(e.g. r1_gpqa_diamond -> process_results_gpqa / exact_match).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_tokenizer_cache: Dict[str, Any] = {}
_log_lock = threading.Lock()


def log_print(*args, **kwargs) -> None:
    """Print with flush so output appears when stdout is piped (e.g. to tee)."""
    kwargs.setdefault("flush", True)
    with _log_lock:
        print(*args, **kwargs)


# Line-buffer stdout when piped; otherwise Python may hold output for a long time.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


def is_empty_resp(resps: Any) -> bool:
    return resps == [[""]] or resps == [""] or resps == ""


def extract_messages(sample: dict) -> List[dict]:
    gen_args = sample.get("arguments", {}).get("gen_args_0", {})
    raw_messages = gen_args.get("arg_0", [None])[0]
    if isinstance(raw_messages, str):
        return json.loads(raw_messages)
    if isinstance(raw_messages, list):
        return raw_messages
    raise ValueError(f"doc_id={sample.get('doc_id')}: could not parse messages")


def resolve_max_tokens(sample: dict, cli_max_tokens: Optional[int]) -> int:
    """Map lm-eval sample limits to the OpenAI API ``max_tokens`` field.

    lm-eval logs ``max_gen_toks`` in sample arguments; the chat completions API
  expects ``max_tokens``. This helper reads the sample limit but the curl payload
    must only ever send ``max_tokens``.
    """
    if cli_max_tokens is not None:
        return cli_max_tokens
    gen_kwargs = sample.get("arguments", {}).get("gen_args_0", {}).get("arg_1", {})
    for sample_key in ("max_tokens", "max_gen_toks"):
        if sample_key in gen_kwargs:
            return int(gen_kwargs[sample_key])
    return 32 * 1024


def build_payload(
    sample: dict,
    model: str,
    *,
    stream: bool,
    max_tokens: int,
) -> dict:
    gen_kwargs = sample.get("arguments", {}).get("gen_args_0", {}).get("arg_1", {})
    # OpenAI-compatible chat API field is max_tokens (never max_gen_toks).
    payload = {
        "model": model,
        "messages": extract_messages(sample),
        "stream": stream,
        "max_tokens": max_tokens,
    }
    if stream:
        # Ask providers to emit token usage on the final stream chunk when supported.
        payload["stream_options"] = {"include_usage": True}
    if "temperature" in gen_kwargs:
        payload["temperature"] = gen_kwargs["temperature"]
    if "top_p" in gen_kwargs:
        payload["top_p"] = gen_kwargs["top_p"]
    if "top_k" in gen_kwargs:
        payload["top_k"] = gen_kwargs["top_k"]
    return payload


def summarize_messages_for_print(messages: List[dict]) -> List[dict]:
    summarized = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            content_summary = f"<{len(content)} chars>"
        elif isinstance(content, list):
            content_summary = f"<{len(content)} parts>"
        else:
            content_summary = f"<{type(content).__name__}>"
        summarized.append(
            {
                "role": message.get("role"),
                "content": content_summary,
                **{
                    key: value
                    for key, value in message.items()
                    if key not in {"role", "content"}
                },
            }
        )
    return summarized


def summarize_payload_for_print(payload: dict) -> dict:
    """Request params for logging, with message bodies redacted."""
    summary = {
        key: value
        for key, value in payload.items()
        if key != "messages"
    }
    summary["messages"] = summarize_messages_for_print(payload.get("messages", []))
    return summary


def append_result_record(
    output_path: Path,
    record: dict,
    *,
    write_lock: Optional[threading.Lock] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _write() -> None:
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()

    if write_lock is None:
        _write()
    else:
        with write_lock:
            _write()


def extract_content(response: dict) -> str:
    """Final answer text only (OpenAI ``message.content``)."""
    choices = response.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if content:
        return content
    if choice.get("text"):
        return choice["text"]
    return ""


def extract_reasoning_content(response: dict) -> str:
    """Thinking/reasoning text (Kimi ``message.reasoning_content``), if present."""
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = (choices[0].get("message") or {})
    return message.get("reasoning_content") or ""


def _stream_chunk_parts(choice: dict) -> tuple[str, str, bool]:
    """Return (reasoning_delta, content_delta, is_chat_chunk) from one SSE chunk."""
    delta = choice.get("delta") or {}
    reasoning = delta.get("reasoning_content") or ""
    content = delta.get("content") or ""
    if (
        reasoning
        or content
        or "reasoning_content" in delta
        or "content" in delta
    ):
        return reasoning, content, True
    text = choice.get("text", "")
    return "", text, bool(text)


class SseAccumulator:
    """Incrementally accumulate OpenAI-style SSE chat/completions streams."""

    def __init__(self) -> None:
        self.accumulated_reasoning: Dict[int, str] = {}
        self.accumulated_content: Dict[int, str] = {}
        self.uses_chat_chunks = False
        self.usage: Optional[dict] = None
        self.finish_reason: Optional[str] = None

    def feed_line(self, raw_line: str) -> None:
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            return
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            index = choice.get("index", 0)
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
            reasoning, content, is_chat_chunk = _stream_chunk_parts(choice)
            self.uses_chat_chunks = self.uses_chat_chunks or is_chat_chunk
            if reasoning:
                self.accumulated_reasoning[index] = (
                    self.accumulated_reasoning.get(index, "") + reasoning
                )
            if content:
                self.accumulated_content[index] = (
                    self.accumulated_content.get(index, "") + content
                )

    def _primary_index(self) -> int:
        indices = set(self.accumulated_content) | set(self.accumulated_reasoning)
        if not indices:
            return 0
        return min(indices)

    def primary_content(self) -> str:
        index = self._primary_index()
        return self.accumulated_content.get(index, "")

    def primary_reasoning(self) -> str:
        index = self._primary_index()
        return self.accumulated_reasoning.get(index, "")

    def to_response(self) -> dict:
        indices = sorted(set(self.accumulated_content) | set(self.accumulated_reasoning))
        choices = []
        for index in indices:
            choice: Dict[str, Any] = {"index": index}
            if self.finish_reason:
                choice["finish_reason"] = self.finish_reason
            if self.uses_chat_chunks:
                message: Dict[str, str] = {}
                reasoning = self.accumulated_reasoning.get(index, "")
                content = self.accumulated_content.get(index, "")
                if reasoning:
                    message["reasoning_content"] = reasoning
                if content:
                    message["content"] = content
                choice["message"] = message
            else:
                choice["text"] = self.accumulated_content.get(index, "")
            choices.append(choice)
        response = {"choices": choices}
        if self.usage:
            response["usage"] = self.usage
        return response


def extract_usage(response: Optional[dict]) -> Optional[dict]:
    if not response:
        return None
    usage = response.get("usage")
    return usage if isinstance(usage, dict) else None


def extract_finish_reason(response: Optional[dict]) -> Optional[str]:
    if not response:
        return None
    choices = response.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


def is_likely_truncated(
    *,
    max_tokens: int,
    usage: Optional[dict],
    finish_reason: Optional[str],
) -> bool:
    if finish_reason == "length":
        return True
    completion_tokens = (usage or {}).get("completion_tokens")
    return completion_tokens is not None and completion_tokens >= max_tokens


def estimate_token_count(text: str, tokenizer_model: str) -> Optional[int]:
    if not text:
        return 0
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    tokenizer = _tokenizer_cache.get(tokenizer_model)
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_model,
                trust_remote_code=True,
            )
        except Exception:
            return None
        _tokenizer_cache[tokenizer_model] = tokenizer

    try:
        return len(tokenizer.encode(text))
    except Exception:
        return None


def format_length_stats(
    content: str,
    *,
    usage: Optional[dict] = None,
    estimated_tokens: Optional[int] = None,
) -> str:
    content_length = len(content)
    byte_length = len(content.encode("utf-8"))
    parts = [f"{content_length} chars", f"{byte_length} bytes"]

    if usage:
        completion_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        total_tokens = usage.get("total_tokens")
        if completion_tokens is not None:
            parts.append(f"api completion_tokens={completion_tokens}")
        if prompt_tokens is not None:
            parts.append(f"api prompt_tokens={prompt_tokens}")
        if total_tokens is not None:
            parts.append(f"api total_tokens={total_tokens}")

    if estimated_tokens is not None:
        parts.append(f"estimated_tokens={estimated_tokens}")

    return ", ".join(parts)


def parse_sse_response(body: str) -> dict:
    """Accumulate OpenAI-style SSE chat/completions stream into a response dict."""
    accumulator = SseAccumulator()
    for raw_line in body.splitlines():
        accumulator.feed_line(raw_line)
    return accumulator.to_response()


def parse_completion_response(body: str, *, stream: bool) -> dict:
    if stream:
        return parse_sse_response(body)
    return json.loads(body)


def _print_request_params(payload: dict) -> None:
    log_print(
        "request params: "
        + json.dumps(summarize_payload_for_print(payload), indent=2)
    )


def _print_request_end_summary(
    *,
    doc_id: int,
    content: str,
    reasoning_length: int,
    stream: bool,
    usage: Optional[dict] = None,
    estimated_tokens: Optional[int] = None,
    status_code: int,
    truncated: bool,
    finish_reason: Optional[str],
    max_tokens: int,
) -> None:
    stream_label = " (stream)" if stream else ""
    log_print(f"=== doc_id={doc_id} done ===")
    log_print(
        f"answer stats{stream_label}: "
        f"{format_length_stats(content, usage=usage, estimated_tokens=estimated_tokens)}"
    )
    if reasoning_length:
        log_print(f"reasoning stats: {reasoning_length} chars (stored in reasoning_content)")
    if truncated:
        completion_tokens = (usage or {}).get("completion_tokens")
        log_print(
            f"WARNING: response likely truncated at max_tokens={max_tokens} "
            f"(finish_reason={finish_reason!r}, api completion_tokens={completion_tokens})"
        )
    if status_code == 200 and content:
        log_print("OK")
    else:
        log_print(f"FAILED status={status_code}")


def curl_chat_completion(
    base_url: str,
    api_key: str,
    payload: dict,
    timeout_sec: int,
) -> tuple[int, str, Optional[dict]]:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/v1/chat/completions"

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as payload_file:
        json.dump(payload, payload_file)
        payload_path = payload_file.name

    stream = bool(payload.get("stream"))
    try:
        cmd = [
            "curl",
            "-sS",
            "-X",
            "POST",
            url,
            "-H",
            f"Authorization: Bearer {api_key}",
            "-H",
            "Content-Type: application/json",
            "--max-time",
            str(timeout_sec),
            "-d",
            f"@{payload_path}",
        ]
        if stream:
            # Flush chunks as they arrive; status code is emitted after the body.
            cmd.extend(["-N", "-w", "\nHTTP_STATUS:%{http_code}"])
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            accumulator = SseAccumulator()
            status_code = 0
            status_prefix = "HTTP_STATUS:"
            for line in proc.stdout:
                if line.startswith(status_prefix):
                    status_code = int(line.strip()[len(status_prefix) :])
                    continue
                accumulator.feed_line(line)

            stderr = proc.stderr.read() if proc.stderr is not None else ""
            return_code = proc.wait()
            if return_code != 0:
                return return_code, stderr.strip() or "curl failed", None
            if status_code != 200:
                return status_code, accumulator.primary_content() or stderr.strip(), None

            parsed = accumulator.to_response()
            if not extract_content(parsed) and not extract_reasoning_content(parsed):
                return status_code, "empty SSE stream (no chunks parsed)", None
            return status_code, "", parsed

        cmd.extend(["-w", "\n%{http_code}"])
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return proc.returncode, proc.stderr.strip() or "curl failed", None

        body, _, status_line = proc.stdout.rpartition("\n")
        status_code = int(status_line.strip() or "0")
        if status_code != 200:
            return status_code, body.strip(), None

        try:
            parsed = parse_completion_response(body, stream=False)
        except json.JSONDecodeError as exc:
            return status_code, f"invalid JSON response: {exc}: {body[:500]}", None

        return status_code, "", parsed
    finally:
        Path(payload_path).unlink(missing_ok=True)


def load_empty_samples(jsonl_path: Path) -> List[dict]:
    samples = []
    with jsonl_path.open() as handle:
        for line in handle:
            sample = json.loads(line)
            if is_empty_resp(sample.get("resps")):
                samples.append(sample)
    return samples


def load_all_samples(jsonl_path: Path) -> List[dict]:
    with jsonl_path.open() as handle:
        return [json.loads(line) for line in handle]


def load_retries(retries_path: Path) -> Dict[int, dict]:
    by_doc_id: Dict[int, dict] = {}
    with retries_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            by_doc_id[int(record["doc_id"])] = record
    return by_doc_id


def get_task_process_results(task_name: str) -> Callable:
    from lm_eval.tasks import TaskManager

    task = list(TaskManager().load_task_or_group(task_name).values())[0]
    return task.config.process_results


def grade_response(sample: dict, content: str, process_results_fn: Callable) -> dict:
    return process_results_fn(sample["doc"], [content])


def compute_regraded_summary(
    samples: List[dict],
    retries_by_doc_id: Dict[int, dict],
    process_results_fn: Callable,
    metric_key: str = "exact_match",
) -> dict:
    per_doc = []
    for sample in samples:
        doc_id = int(sample["doc_id"])
        retry = retries_by_doc_id.get(doc_id)
        if retry and retry.get("content"):
            metrics = grade_response(sample, retry["content"], process_results_fn)
            exact_match = int(metrics.get(metric_key, 0))
            source = "retry"
        else:
            exact_match = int(sample.get(metric_key, 0))
            source = "original"

        per_doc.append(
            {
                "doc_id": doc_id,
                "exact_match": exact_match,
                "target": sample.get("target"),
                "source": source,
            }
        )

    scores = [row["exact_match"] for row in per_doc]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "metric_key": metric_key,
        "num_samples": len(per_doc),
        "num_retried": sum(1 for row in per_doc if row["source"] == "retry"),
        "exact_match_mean": mean_score,
        "per_doc": per_doc,
    }


def write_regrade_report(summary: dict, output_path: Path) -> None:
    with output_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    log_print(
        f"Regraded exact_match mean: {summary['exact_match_mean']:.4f} "
        f"({summary['num_samples']} samples, {summary['num_retried']} retried)"
    )
    log_print(f"Wrote regrade report -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry empty lm-eval samples (resps == [[\"\"]]) using curl."
    )
    parser.add_argument(
        "jsonl_path",
        type=Path,
        help="Path to samples_*.jsonl produced by lm-eval",
    )
    parser.add_argument(
        "--base-url",
        default="https://console.tenstorrent.com",
        help="API base URL (default: https://console.tenstorrent.com)",
    )
    parser.add_argument(
        "--model",
        default="moonshotai/Kimi-K2.6",
        help="Model id sent in the request payload",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"),
        help="Bearer token (default: API_KEY or OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL path (default: <input>.retries.jsonl)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="OpenAI max_tokens cap for curl (default: mapped from sample max_gen_toks)",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=3600,
        help="curl --max-time per request in seconds (default: 3600)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Retry with stream=true; SSE data: lines are accumulated like lm-eval",
    )
    parser.add_argument(
        "--doc-id",
        type=int,
        action="append",
        help="Only retry specific doc_id(s); repeatable (includes inference errors, not just empty resps)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing .retries.jsonl instead of overwriting it",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print curl targets without sending requests",
    )
    parser.add_argument(
        "--regrade",
        action="store_true",
        help="After retrying, score responses with the lm-eval task metric",
    )
    parser.add_argument(
        "--regrade-only",
        type=Path,
        metavar="RETRIES_JSONL",
        help="Skip curl; regrade from an existing .retries.jsonl file",
    )
    parser.add_argument(
        "--task",
        default="r1_gpqa_diamond",
        help="lm-eval task name used for regrading (default: r1_gpqa_diamond)",
    )
    parser.add_argument(
        "--regrade-output",
        type=Path,
        help="Regrade report JSON path (default: <input>.regrade.json)",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help="HF model id for local token estimates (default: same as --model)",
    )
    parser.add_argument(
        "--no-estimate-tokens",
        action="store_true",
        help="Skip local tokenizer-based completion token estimates",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=16,
        help="Number of concurrent curl retries (default: 16)",
    )
    return parser.parse_args()


def retry_one_sample(
    sample: dict,
    *,
    base_url: str,
    api_key: str,
    model: str,
    stream: bool,
    max_tokens_override: Optional[int],
    timeout_sec: int,
    tokenizer_model: Optional[str],
    no_estimate_tokens: bool,
    output_path: Path,
    write_lock: threading.Lock,
) -> dict:
    doc_id = sample.get("doc_id")
    max_tokens = resolve_max_tokens(sample, max_tokens_override)
    payload = build_payload(
        sample,
        model,
        stream=stream,
        max_tokens=max_tokens,
    )
    log_print(f"\n=== doc_id={doc_id} (max_tokens={max_tokens}) ===")
    _print_request_params(payload)
    log_print("waiting for API response...")

    status_code, error_text, response = curl_chat_completion(
        base_url,
        api_key,
        payload,
        timeout_sec=timeout_sec,
    )

    content = extract_content(response) if response else ""
    reasoning_content = extract_reasoning_content(response) if response else ""
    content_length = len(content)
    reasoning_length = len(reasoning_content)
    usage = extract_usage(response)
    finish_reason = extract_finish_reason(response)
    truncated = is_likely_truncated(
        max_tokens=max_tokens,
        usage=usage,
        finish_reason=finish_reason,
    )
    estimated_tokens = None
    if not no_estimate_tokens and content:
        log_print("computing estimated token count...")
        estimated_tokens = estimate_token_count(content, tokenizer_model or model)

    request_params = summarize_payload_for_print(payload)
    record = {
        "doc_id": doc_id,
        "request_params": request_params,
        "status_code": status_code,
        "error": error_text or None,
        "content": content,
        "content_length": content_length,
        "reasoning_content": reasoning_content,
        "reasoning_length": reasoning_length,
        "max_tokens": max_tokens,
        "finish_reason": finish_reason,
        "truncated": truncated,
        "usage": usage,
        "completion_tokens": usage.get("completion_tokens") if usage else None,
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "total_tokens": usage.get("total_tokens") if usage else None,
        "estimated_completion_tokens": estimated_tokens,
        "raw_response": response,
    }
    append_result_record(output_path, record, write_lock=write_lock)
    log_print(f"saved doc_id={doc_id} -> {output_path}")
    _print_request_end_summary(
        doc_id=doc_id,
        content=content,
        reasoning_length=reasoning_length,
        stream=stream,
        usage=usage,
        estimated_tokens=estimated_tokens,
        status_code=status_code,
        truncated=truncated,
        finish_reason=finish_reason,
        max_tokens=max_tokens,
    )
    if status_code != 200 or (not content and not reasoning_content):
        log_print(error_text or json.dumps(response, indent=2)[:500])
    return record


def main() -> int:
    args = parse_args()
    jsonl_path = args.jsonl_path
    if not jsonl_path.exists():
        log_print(f"ERROR: file not found: {jsonl_path}", file=sys.stderr)
        return 1

    if args.regrade_only:
        retries_path = args.regrade_only
        if not retries_path.exists():
            log_print(f"ERROR: retries file not found: {retries_path}", file=sys.stderr)
            return 1
        retries_by_doc_id = load_retries(retries_path)
        process_results_fn = get_task_process_results(args.task)
        summary = compute_regraded_summary(
            load_all_samples(jsonl_path),
            retries_by_doc_id,
            process_results_fn,
        )
        regrade_output = (
            args.regrade_output
            or jsonl_path.with_suffix(jsonl_path.suffix + ".regrade.json")
        )
        write_regrade_report(summary, regrade_output)
        return 0

    if not args.dry_run and not args.api_key:
        log_print("ERROR: set API_KEY or pass --api-key", file=sys.stderr)
        return 1

    if args.doc_id:
        wanted = set(args.doc_id)
        samples = [
            sample
            for sample in load_all_samples(jsonl_path)
            if sample.get("doc_id") in wanted
        ]
        missing = wanted - {sample.get("doc_id") for sample in samples}
        if missing:
            log_print(f"WARNING: doc_id(s) not found in samples: {sorted(missing)}")
    else:
        samples = load_empty_samples(jsonl_path)

    if not samples:
        log_print("No samples to retry.")
        return 0

    output_path = args.output or jsonl_path.with_suffix(jsonl_path.suffix + ".retries.jsonl")
    parallel = max(1, args.parallel)
    log_print(
        f"Retrying {len(samples)} sample(s) -> {output_path} "
        f"(parallel={parallel})"
    )
    if not args.dry_run and not args.append:
        output_path.write_text("", encoding="utf-8")

    results: List[dict] = []
    if args.dry_run:
        for sample in samples:
            doc_id = sample.get("doc_id")
            max_tokens = resolve_max_tokens(sample, args.max_tokens)
            payload = build_payload(
                sample,
                args.model,
                stream=args.stream,
                max_tokens=max_tokens,
            )
            log_print(f"\n=== doc_id={doc_id} (max_tokens={max_tokens}) ===")
            _print_request_params(payload)
    else:
        write_lock = threading.Lock()
        tokenizer_model = args.tokenizer_model or args.model
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(
                    retry_one_sample,
                    sample,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    stream=args.stream,
                    max_tokens_override=args.max_tokens,
                    timeout_sec=args.timeout_sec,
                    tokenizer_model=tokenizer_model,
                    no_estimate_tokens=args.no_estimate_tokens,
                    output_path=output_path,
                    write_lock=write_lock,
                ): sample
                for sample in samples
            }
            for future in as_completed(futures):
                sample = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    doc_id = sample.get("doc_id")
                    log_print(
                        f"ERROR: doc_id={doc_id} raised {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    results.append(
                        {
                            "doc_id": doc_id,
                            "status_code": 0,
                            "error": str(exc),
                            "content": "",
                        }
                    )

    if args.dry_run:
        return 0

    ok = sum(1 for record in results if record["content"])
    log_print(f"\nFinished {len(results)} result(s) in {output_path} ({ok} non-empty)")
    for record in results:
        log_print(
            f"  doc_id={record['doc_id']}: "
            f"{format_length_stats(record.get('content', ''), usage=record.get('usage'), estimated_tokens=record.get('estimated_completion_tokens'))}"
        )

    if args.regrade and results:
        retries_by_doc_id = {int(record["doc_id"]): record for record in results}
        process_results_fn = get_task_process_results(args.task)
        summary = compute_regraded_summary(
            load_all_samples(jsonl_path),
            retries_by_doc_id,
            process_results_fn,
        )
        regrade_output = (
            args.regrade_output
            or jsonl_path.with_suffix(jsonl_path.suffix + ".regrade.json")
        )
        write_regrade_report(summary, regrade_output)
        for row in summary["per_doc"]:
            if row["source"] == "retry":
                mark = "✓" if row["exact_match"] else "✗"
                retry = retries_by_doc_id.get(int(row["doc_id"]), {})
                log_print(
                    f"  {mark} doc_id={row['doc_id']} target={row['target']} "
                    f"{format_length_stats(retry.get('content', ''), usage=retry.get('usage'), estimated_tokens=retry.get('estimated_completion_tokens'))}"
                )

    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
