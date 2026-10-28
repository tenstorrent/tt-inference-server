# SPDX-License-Identifier: Apache-2.0
"""Unscored fixed-work prefix retention probe, explicitly selected by goal12 controls.

Uses the server's pinned tokenizer and native chat template, then sends exactly
those token IDs to completions. Predetermined transcript extensions isolate cache
mechanics from generated trajectories. One greedy output token is deliberately
different from the scored SWE policy; results cannot establish task quality.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import urllib.request


def sha(tokens):
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


def lcp(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def parse_metrics(raw):
    """Keep samples and labels; never combine histograms or independent engines."""
    prefixes = (
        "vllm:prefix_cache_",
        "vllm:num_requests_",
        "vllm:kv_cache_usage",
        "vllm:num_preemptions",
        "vllm:request_queue_time_",
        "vllm:prompt_tokens",
        "vllm:generation_tokens",
        "vllm:request_prefill_time_",
        "tt_prefix_cache_",
        "tt_num_requests_",
        "tt_kv_cache_",
    )
    rows = []
    for line in raw.splitlines():
        if line.startswith("#") or not line:
            continue
        parts = line.rsplit(None, 1)
        if len(parts) == 2 and parts[0].startswith(prefixes):
            try:
                value = float(parts[1])
            except ValueError:
                continue
            if math.isfinite(value):
                rows.append({"series": parts[0], "value": value})
    return rows


def validate_config(config):
    fields = {
        "schema_version",
        "study_namespace",
        "block_size",
        "base_lengths",
        "streams",
        "concurrency",
        "append_tokens",
        "churn_streams",
        "tool_gap_s",
        "request_budget_s",
    }
    if set(config) != fields or config["schema_version"] != 2:
        raise ValueError("Unknown or incomplete prefix-probe configuration")
    if (
        not isinstance(config["study_namespace"], str)
        or not config["study_namespace"].strip()
    ):
        raise ValueError(
            "A fixed cross-arm study_namespace is required; start a fresh server for every arm"
        )
    for name in fields - {"base_lengths", "tool_gap_s", "study_namespace"}:
        if type(config[name]) is not int:
            raise ValueError(f"{name} must be an integer")
    if not 1 <= config["streams"] <= 40 or not 1 <= config["concurrency"] <= 40:
        raise ValueError("Bounded probe supports at most40 client streams/concurrency")
    if (
        config["block_size"] not in (512, 1024, 2048)
        or not 1 <= config["append_tokens"] <= 2048
    ):
        raise ValueError("Invalid checkpoint or append geometry")
    if not 0 <= config["churn_streams"] <= 40 or not 0 <= config["tool_gap_s"] <= 60:
        raise ValueError("Invalid churn/gap bounds")
    if not 1 <= config["request_budget_s"] <= 3600:
        raise ValueError("Probe must retain its one-hour request budget")
    if (
        not isinstance(config["base_lengths"], list)
        or not 1 <= len(config["base_lengths"]) <= 2
    ):
        raise ValueError("Select one or two bounded input-length panels")
    if any(
        type(n) is not int or not 1024 <= n <= 32768 for n in config["base_lengths"]
    ):
        raise ValueError("Mechanism panels must lie within1K..32K")


class Probe:
    def __init__(self, *, base_url, model, config, output_dir):
        validate_config(config)
        self.url = base_url.rstrip("/")
        self.model, self.config = model, config
        self.root = Path(output_dir) / "qwen38_prefix_retention"
        self.root.mkdir(parents=True, exist_ok=False)
        self.session = config["study_namespace"]
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.outstanding = 0
        self.calls = []
        self.metrics = []
        self.prompts = []
        self.phase = "input_preparation"
        self.deadline = time.monotonic() + config["request_budget_s"]

    def http(self, path, payload=None, *, timeout=1800):
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
        if key:
            headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(
            self.url + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers=headers,
        )
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Fixed-work request budget exhausted")
        with urllib.request.urlopen(req, timeout=min(timeout, remaining)) as response:
            raw = response.read()
        return raw.decode() if path == "/metrics" else json.loads(raw)

    def write_inputs(self):
        (self.root / "inputs.json").write_text(
            json.dumps(dict(session=self.session, prompts=self.prompts))
        )

    def tokenize(self, content):
        body = self.http(
            "/tokenize",
            dict(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                add_generation_prompt=True,
                chat_template_kwargs={
                    "enable_thinking": True,
                    "reasoning_effort": "medium",
                    "preserve_thinking": True,
                },
            ),
        )
        ids = body.get("tokens")
        if not isinstance(ids, list) or not all(type(t) is int for t in ids):
            raise RuntimeError("Server /tokenize did not return integer token IDs")
        return ids

    def prompt(self, stream, target, *, family):
        # Private textual prefix plus salt prevents cross-stream reuse. The control
        # tokens remain exactly those produced by the checkpoint's chat template.
        header = f"Private research stream {stream} family {family}. Continue the technical record.\n"
        count = max(1, target - 100)
        for _ in range(10):
            content = header + " observation" * count
            ids = self.tokenize(content)
            if len(ids) == target:
                self.prompts.append(
                    dict(
                        stream=stream,
                        family=family,
                        tokens=ids,
                        token_count=len(ids),
                        sha256=sha(ids),
                        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                    )
                )
                return ids
            count += target - len(ids)
            if count < 1:
                break
        raise RuntimeError(
            f"Cannot render exact native-chat length {target}; last length {len(ids)}"
        )

    def call(self, *, panel, stream, turn, tokens, salt, previous=None, phase):
        before = time.time()
        reusable = lcp(previous, tokens) if previous is not None else 0
        block = self.config["block_size"]
        eligible = min((len(tokens) - 1) // block * block, reusable // block * block)
        row = dict(
            panel=panel,
            stream=stream,
            turn=turn,
            phase=phase,
            arrival_wall=before,
            prompt_tokens=len(tokens),
            input_sha256=sha(tokens),
            previous_sha256=sha(previous) if previous is not None else None,
            lcp_tokens=reusable,
            reusable_aligned_tokens=eligible,
            cache_salt=salt,
        )
        with self.lock:
            self.outstanding += 1
            row["client_outstanding_at_arrival"] = self.outstanding
        try:
            body = self.http(
                "/v1/completions",
                dict(
                    model=self.model,
                    prompt=tokens,
                    max_tokens=1,
                    ignore_eos=True,
                    temperature=0,
                    top_k=1,
                    top_p=1,
                    seed=9472,
                    cache_salt=salt,
                    return_token_ids=True,
                    stream=False,
                ),
            )
            usage = body.get("usage") or {}
            ids = body["choices"][0].get("token_ids")
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            if (
                usage.get("prompt_tokens") != len(tokens)
                or usage.get("completion_tokens") != 1
            ):
                raise RuntimeError(f"Unexpected HTTP token accounting: {usage}")
            if not isinstance(ids, list) or len(ids) != 1:
                raise RuntimeError("Missing actual generated token ID")
            if type(cached) is not int or not 0 <= cached <= len(tokens):
                raise RuntimeError("Missing or invalid HTTP cached-prompt token count")
            if previous is None and cached != 0:
                raise RuntimeError(
                    "Private cold stream unexpectedly shared cached tokens"
                )
            if previous is not None and cached > reusable:
                raise RuntimeError("Cached tokens exceed actual token-prefix identity")
            row.update(
                response=body,
                cached_prompt_tokens=cached,
                uncached_prompt_tokens=len(tokens) - cached,
                eligible_prefix_missed_tokens=max(0, eligible - cached),
                prior_common_prefix_uncached_tokens=max(0, reusable - cached),
                new_prompt_tokens_after_lcp=len(tokens) - reusable,
                retention_fraction_of_eligible=(
                    cached / eligible if eligible else None
                ),
            )
        except BaseException as error:
            row["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            row.update(end_wall=time.time(), elapsed_s=time.time() - before)
            with self.lock:
                self.outstanding -= 1
                self.calls.append(row)
                with (self.root / "calls.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
        return row

    def sample_metrics(self):
        while not self.stop.is_set():
            row = dict(wall=time.time(), phase=self.phase)
            try:
                raw = self.http("/metrics", timeout=10)
                row.update(samples=parse_metrics(raw), raw=raw)
            except Exception as error:
                row["error"] = str(error)
            with self.lock:
                row["client_outstanding"] = self.outstanding
                self.metrics.append(row)
                with (self.root / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
            self.stop.wait(1)

    def run(self):
        monitor = threading.Thread(target=self.sample_metrics, daemon=True)
        monitor.start()
        result = dict(
            status="running",
            scored=False,
            session=self.session,
            config=self.config,
            model=self.model,
            prompt_mode="server-pinned native chat /tokenize -> exact token-ID completions",
            generation_policy=dict(
                max_tokens=1, ignore_eos=True, temperature=0, top_k=1, seed=9472
            ),
            limitations=[
                "Predetermined extended user transcripts, not live agents or scored SWE.",
                "One capped greedy output differs from scored generation policy.",
                "Metrics are sampled; gaps and unknown series stay explicit.",
                "HTTP cached tokens establish reuse; allocator ownership comes from separate native inventory.",
            ],
        )
        target = self.root / "result.json"
        try:
            result["server_models"] = self.http("/v1/models")
            model_rows = result["server_models"].get("data", [])
            if not any(
                m.get("id") == self.model and m.get("max_model_len") == 262144
                for m in model_rows
            ):
                raise RuntimeError("Server did not attest native262144 model context")
            for panel, base in enumerate(self.config["base_lengths"]):
                streams, active = self.config["streams"], self.config["concurrency"]
                prefix = f"{self.session}-panel{panel}"
                prompts = {
                    i: [
                        self.prompt(
                            i, base + turn * self.config["append_tokens"], family=panel
                        )
                        for turn in range(3)
                    ]
                    for i in range(streams)
                }
                if any(
                    lcp(prompts[0][0], prompts[i][0]) >= 128 for i in range(1, streams)
                ):
                    raise RuntimeError(
                        "Private streams lack distinct early text tokens"
                    )
                self.write_inputs()

                def wave(turn):
                    self.phase = (
                        f"panel{panel}-"
                        + ("cold", "warm_after_gap", "after_churn")[turn]
                    )

                    def task(i):
                        return self.call(
                            panel=panel,
                            stream=i,
                            turn=turn,
                            tokens=prompts[i][turn],
                            previous=prompts[i][turn - 1] if turn else None,
                            salt=f"{prefix}-stream{i}",
                            phase=("cold", "warm_after_gap", "after_churn")[turn],
                        )

                    with ThreadPoolExecutor(max_workers=active) as pool:
                        return list(pool.map(task, range(streams)))

                wave(0)
                time.sleep(self.config["tool_gap_s"])
                warm_calls = wave(1)
                self.phase = f"panel{panel}-churn"
                churn = self.config["churn_streams"]
                churn_prompts = [
                    self.prompt(i + streams, base, family=panel) for i in range(churn)
                ]
                self.write_inputs()

                def churn_task(i):
                    return self.call(
                        panel=panel,
                        stream=streams + i,
                        turn=0,
                        tokens=churn_prompts[i],
                        salt=f"{prefix}-churn{i}",
                        phase="churn",
                    )

                with ThreadPoolExecutor(max_workers=active) as pool:
                    list(pool.map(churn_task, range(churn)))
                time.sleep(self.config["tool_gap_s"])
                wave(2)
                # Same exact native-chat IDs under a fresh private salt provide
                # a bounded cached/uncached output control after the retention panel.
                self.phase = f"panel{panel}-uncached_controls"
                for i in sorted({0, streams - 1}):
                    control = self.call(
                        panel=panel,
                        stream=i,
                        turn=1,
                        tokens=prompts[i][1],
                        salt=f"{prefix}-uncached-control{i}",
                        phase="uncached_control",
                    )
                    expected = warm_calls[i]["response"]["choices"][0]["token_ids"]
                    if control["response"]["choices"][0]["token_ids"] != expected:
                        raise RuntimeError(
                            f"Cached/uncached greedy token mismatch in stream{i}"
                        )
            if any("error" in m for m in self.metrics):
                result["metrics_errors"] = sum("error" in m for m in self.metrics)
            observed = {
                r["series"].split("{")[0]
                for m in self.metrics
                for r in m.get("samples", [])
            }
            for fragment in (
                "num_requests_running",
                "num_requests_waiting",
                "preemptions",
            ):
                if not any(fragment in name for name in observed):
                    raise RuntimeError(f"Required capacity metric absent: {fragment}")
            panels = {}
            for row in self.calls:
                key = f"panel{row['panel']}-{row['phase']}"
                total = panels.setdefault(
                    key,
                    dict(
                        requests=0,
                        prompt_tokens=0,
                        cached_prompt_tokens=0,
                        reusable_aligned_tokens=0,
                        uncached_prompt_tokens=0,
                        eligible_prefix_missed_tokens=0,
                        prior_common_prefix_uncached_tokens=0,
                        new_prompt_tokens_after_lcp=0,
                    ),
                )
                total["requests"] += 1
                for field in (
                    "prompt_tokens",
                    "cached_prompt_tokens",
                    "reusable_aligned_tokens",
                    "uncached_prompt_tokens",
                    "eligible_prefix_missed_tokens",
                    "prior_common_prefix_uncached_tokens",
                    "new_prompt_tokens_after_lcp",
                ):
                    total[field] += row[field]
            for total in panels.values():
                eligible = total["reusable_aligned_tokens"]
                total["retention_fraction_of_eligible"] = (
                    total["cached_prompt_tokens"] / eligible if eligible else None
                )
            result["panels"] = panels
            result["cached_uncached_sampled_greedy_controls_exact"] = True
            result.update(
                status="pass",
                request_count=len(self.calls),
                metrics_samples=len(self.metrics),
                metrics_series=sorted(observed),
                cached_prompt_tokens=sum(r["cached_prompt_tokens"] for r in self.calls),
                prompt_tokens=sum(r["prompt_tokens"] for r in self.calls),
                eligible_reuse_tokens=sum(
                    r["reusable_aligned_tokens"] for r in self.calls
                ),
                completion_tokens=len(self.calls),
            )
            return result
        except BaseException as error:
            result.update(
                status="fail", error_type=type(error).__name__, error=str(error)
            )
            raise
        finally:
            self.stop.set()
            monitor.join(timeout=12)
            self.write_inputs()
            result["elapsed_budget_used_s"] = self.config["request_budget_s"] - (
                self.deadline - time.monotonic()
            )
            target.write_text(json.dumps(result, indent=2) + "\n")


def run(ctx, config):
    from report_module.schema import Block

    probe = Probe(
        base_url=ctx.base_url,
        model=ctx.model_spec.hf_model_repo,
        config=config,
        output_dir=ctx.output_path,
    )
    result = probe.run()
    return Block(
        kind="qwen38_prefix_retention",
        title="Unscored Qwen prefix retention mechanism probe",
        data=dict(
            success=True,
            scored=False,
            records=[
                {
                    k: result[k]
                    for k in (
                        "status",
                        "request_count",
                        "prompt_tokens",
                        "cached_prompt_tokens",
                        "eligible_reuse_tokens",
                        "completion_tokens",
                    )
                }
            ],
            artifact=str(probe.root / "result.json"),
        ),
    )
