# SPDX-License-Identifier: Apache-2.0
"""Bounded unscored B1 HTTP timing and native-chat regression evidence."""

import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request

QUALITATIVE_PROMPTS = [
    "Write a haiku about machine learning.",
    "Explain the difference between supervised and unsupervised learning in simple terms.",
    "Complete this story: Once upon a time, in a faraway kingdom, there lived a curious young inventor who discovered",
    "What are the three laws of thermodynamics?",
    'Translate the following to French: "Hello, how are you today?"',
    "Write a Python function to calculate the Fibonacci sequence.",
]


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


class Probe:
    def __init__(self, url, model, output):
        self.url, self.model = url.rstrip("/"), model
        self.output = Path(output) / "qwen38_b1_serving"
        self.output.mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.inputs = []
        self.deadline = time.monotonic() + 1800

    def request(self, path, body=None):
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
        if key:
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers,
        )
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("B1 HTTP probe exhausted its30minute request budget")
        return urllib.request.urlopen(request, timeout=min(600, remaining))

    def json(self, path, body=None):
        with self.request(path, body) as response:
            return json.load(response)

    def tokenize(self, content):
        body = dict(
            model=self.model,
            messages=[dict(role="user", content=content)],
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        tokens = self.json("/tokenize", body)["tokens"]
        if not tokens or not all(type(token) is int for token in tokens):
            raise ValueError("Native-chat tokenizer did not return integer IDs")
        self.inputs.append(
            dict(request=body, tokens=tokens, input_sha256=digest(tokens))
        )
        return tokens

    def exact_prompt(self, length):
        count = max(1, length - 40)
        for _ in range(10):
            tokens = self.tokenize(
                "Read this technical record, then explain its topic briefly.\n"
                + " observation" * count
            )
            if len(tokens) == length:
                return tokens
            count += length - len(tokens)
            if count < 1:
                break
        raise ValueError(f"Cannot render exact native-chat length{length}")

    def timed(self, tokens, repeat):
        body = dict(
            model=self.model,
            prompt=tokens,
            max_tokens=128,
            ignore_eos=True,
            temperature=0,
            top_k=1,
            top_p=1,
            seed=9472,
            return_token_ids=True,
            stream=True,
            stream_options={"include_usage": True},
        )
        started = time.perf_counter()
        first = last = None
        ids, texts, events = [], [], []
        usage = None
        with self.request("/v1/completions", body) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                value = json.loads(payload)
                now = time.perf_counter()
                if value.get("error"):
                    raise RuntimeError(value["error"])
                if value.get("usage"):
                    usage = value["usage"]
                for choice in value.get("choices", []):
                    current = choice.get("token_ids") or []
                    if current:
                        first = now if first is None else first
                        last = now
                        ids.extend(current)
                        events.append(
                            dict(elapsed_s=now - started, token_count=len(current))
                        )
                    texts.append(choice.get("text", ""))
        ended = time.perf_counter()
        if len(ids) != 128 or not first or last <= first:
            raise ValueError(
                f"Incomplete streamed token-ID/timing evidence: {len(ids)}"
            )
        if (
            not usage
            or usage.get("prompt_tokens") != len(tokens)
            or usage.get("completion_tokens") != 128
        ):
            raise ValueError(f"Incorrect HTTP usage: {usage}")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if cached:
            raise ValueError("Cold-prefill timing requires prefix cache disabled")
        row = dict(
            length=len(tokens),
            repeat=repeat,
            measured=repeat >= 0,
            input_sha256=digest(tokens),
            generation=body | {"prompt": "recorded_in_inputs.json"},
            tokens=ids,
            text="".join(texts),
            usage=usage,
            events=events,
            ttft_ms=1000 * (first - started),
            tpot_ms=1000 * (last - first) / 127,
            decode_tokens_per_second_per_user=127 / (last - first),
            request_wall_s=ended - started,
            output_throughput=128 / (ended - started),
        )
        self.rows.append(row)
        with (self.output / "calls.jsonl").open("a") as target:
            target.write(json.dumps(row) + "\n")
        return row

    def run(self):
        result = dict(
            status="running",
            scored=False,
            physical_batch=1,
            native_context=262144,
            outputs=128,
            timing="client SSE first-to-last token,127 intervals; includes serving/network; no profiler",
            prompt_mode="server native-chat /tokenize; enable_thinking=False",
            qualitative_prompt_source="shared readiness_check/vllm_prompts.txt six prompts",
        )
        try:
            models = self.json("/v1/models")["data"]
            if not any(
                row.get("id") == self.model and row.get("max_model_len") == 262144
                for row in models
            ):
                raise ValueError("Server does not attest native262144 context")
            result["models"] = models
            for length in (128, 4096, 129):
                tokens = self.exact_prompt(length)
                rows = [self.timed(tokens, repeat) for repeat in (-1, 0, 1, 2)]
                if any(row["tokens"] != rows[0]["tokens"] for row in rows):
                    raise ValueError(
                        f"Warmed repeated greedy outputs differ at{length}"
                    )
            result["qualitative"] = []
            for prompt in QUALITATIVE_PROMPTS:
                tokens = self.tokenize(prompt)
                generation = dict(
                    model=self.model,
                    prompt=tokens,
                    max_tokens=256,
                    temperature=0,
                    top_k=1,
                    top_p=1,
                    seed=9472,
                    return_token_ids=True,
                    stream=False,
                )
                response = self.json("/v1/completions", generation)
                if not response["choices"][0].get("token_ids"):
                    raise ValueError("Missing qualitative generated token IDs")
                result["qualitative"].append(
                    dict(
                        prompt=prompt,
                        input_sha256=digest(tokens),
                        response=response,
                        generation=generation,
                    )
                )
            result["summary"] = {}
            for length in (128, 4096, 129):
                rows = [
                    row
                    for row in self.rows
                    if row["length"] == length and row["measured"]
                ]
                result["summary"][str(length)] = {
                    key: {
                        "values": [row[key] for row in rows],
                        "median": statistics.median(row[key] for row in rows),
                    }
                    for key in (
                        "ttft_ms",
                        "tpot_ms",
                        "decode_tokens_per_second_per_user",
                        "output_throughput",
                    )
                }
            result["status"] = "pass_requires_manual_output_and_matched_arm_review"
            return result
        except BaseException as error:
            result.update(
                status="fail", error_type=type(error).__name__, error=str(error)
            )
            raise
        finally:
            (self.output / "inputs.json").write_text(json.dumps(self.inputs) + "\n")
            (self.output / "result.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )


def run(ctx, config):
    from report_module.schema import Block

    if config != {"schema_version": 1, "study": "qwen38-b1-compact-serving-v1"}:
        raise ValueError("Unexpected B1 serving probe configuration")
    probe = Probe(ctx.base_url, ctx.model_spec.hf_model_repo, ctx.output_path)
    result = probe.run()
    return Block(
        kind="qwen38_b1_serving",
        title="Unscored B1 serving regression probe",
        data=dict(
            success=True,
            scored=False,
            records=[result["summary"]],
            artifact=str(probe.output / "result.json"),
        ),
    )
