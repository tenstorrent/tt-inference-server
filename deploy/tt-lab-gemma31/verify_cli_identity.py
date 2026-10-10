#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Greedy token identity: the server's Gemma 4 31B output against the tt-lab CLI.

Two phases, because the native worker owns all four cards:

  reference  (server STOPPED) runs `tt-lab gemma --native-device --tokens FILE -n N`
             once per prompt, with the deployment env file's TT_* settings, on the
             exact token IDs the server builds for that chat request; saves JSON.
  check      (server RUNNING) sends the same chat requests to the API, streamed and
             not, sequentially and concurrently, and requires the text and
             completion token count to equal the CLI's. With --server-log it also
             reads the worker's per-request lines ("reused=N") and reports prefix
             cache hits on the shared-prefix prompts.

  python deploy/tt-lab-gemma31/verify_cli_identity.py reference --env deploy/tt-lab-gemma31/server-batch.env \
      --output /tmp/gemma31-cli-reference.json [--dry-run]
  python deploy/tt-lab-gemma31/verify_cli_identity.py check --reference /tmp/gemma31-cli-reference.json \
      --url http://127.0.0.1:8002 --concurrency 8 --server-log /tmp/gemma31-server.log

Token IDs are built as the server builds them: the chat template with
CHAT_TEMPLATE_KWARGS and add_generation_prompt, then encode without special
tokens (tt_lab_validation.validate_request). Use the tt-lab venv with the Gemma 4
tokenizer support (PYTHONPATH=/home/ttuser/workspace/gemma-reference/python).
"""
import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# Prompts: short factual, code, long-ish, and a shared-prefix pair (prefix caching).
DOCUMENT = " ".join(f"Record {i}: the sensor in bay {i % 17} read {(i * 37) % 101} units at hour {i % 24}."
                    for i in range(180))
PROMPTS = [
    dict(name="arithmetic", max_tokens=32, messages=[{"role": "user", "content": "What is 17 times 23? Answer briefly."}]),
    dict(name="capital", max_tokens=48, messages=[{"role": "user", "content": "Name the capital of Australia and one fact about it."}]),
    dict(name="code", max_tokens=160, messages=[{"role": "user", "content": "Write a Python function that reverses a string, with a docstring."}]),
    dict(name="story", max_tokens=256, messages=[{"role": "user", "content": "Write a short story about a lighthouse keeper who finds a map."}]),
    dict(name="list", max_tokens=120, messages=[{"role": "system", "content": "You are terse."},
                                                 {"role": "user", "content": "List five prime numbers above 100."}]),
    dict(name="doc-q1", max_tokens=64, prefix_group="doc",
         messages=[{"role": "user", "content": DOCUMENT + "\n\nWhich bay is mentioned in record 42?"}]),
    dict(name="doc-q2", max_tokens=64, prefix_group="doc",
         messages=[{"role": "user", "content": DOCUMENT + "\n\nWhat did the sensor read in record 99?"}]),
    dict(name="doc-q3", max_tokens=64, prefix_group="doc",
         messages=[{"role": "user", "content": DOCUMENT + "\n\nAt what hour was record 150 taken?"}]),
]


def load_env(path):
    env = {}
    for line in Path(path).read_text().splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            env[key] = value
    return env


def tokenizer_for(env):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(env["MODEL_WEIGHTS_PATH"], local_files_only=True)


def prompt_tokens(tokenizer, env, messages):
    kwargs = json.loads(env.get("CHAT_TEMPLATE_KWARGS", "{}") or "{}")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    return tokenizer.encode(text, add_special_tokens=False)


def reference(a):
    env = load_env(a.env)
    tokenizer = tokenizer_for(env)
    child_env = dict(os.environ)
    child_env.pop("TT_LAB_DEVICE", None)
    child_env.update({k: v for k, v in env.items() if k.startswith(("TT_GEMMA", "TT_LAB_HUGEPAGE"))})
    binary = a.binary or env["TT_LAB_BINARY"]
    prompts = list(PROMPTS)
    for n in a.long_tokens:
        # DOCUMENT is about 4,000 tokens; repeat it to at most about n prompt tokens.
        copies = max(1, n // 4000)
        prompts.append(dict(name=f"long-{n}", max_tokens=64, messages=[
            {"role": "user", "content": "\n\n".join([DOCUMENT] * copies) + "\n\nWhat did record 7 read?"}]))
    results = []
    for p in prompts:
        tokens = prompt_tokens(tokenizer, env, p["messages"])
        with tempfile.NamedTemporaryFile("w", suffix=".tokens", delete=False) as f:
            f.write(" ".join(map(str, tokens)) + "\n")
            token_file = f.name
        command = [binary, "gemma", "-m", env["TT_LAB_GGUF"], "--ttq", env["TT_LAB_TTQ"], "--native-device",
                   "--tokens", token_file, "-n", str(p["max_tokens"]), "--context", env["TT_LAB_CONTEXT"]]
        if a.dry_run:
            os.unlink(token_file)
            print(json.dumps(dict(name=p["name"], prompt_tokens=len(tokens), command=" ".join(command))))
            continue
        begin = time.monotonic()
        try:
            run = subprocess.run(command, env=child_env, capture_output=True, text=True, timeout=a.timeout)
        finally:
            os.unlink(token_file)
        if run.returncode:
            sys.exit(f"{p['name']}: CLI exited {run.returncode}\n{run.stderr[-2000:]}")
        generated = [int(line) for line in run.stdout.split() if re.fullmatch(r"-?\d+", line)]
        results.append(dict(name=p["name"], messages=p["messages"], max_tokens=p["max_tokens"],
                            prefix_group=p.get("prefix_group"), prompt_tokens=tokens, cli_tokens=generated,
                            text=tokenizer.decode(generated, skip_special_tokens=True),
                            finish="length" if len(generated) == p["max_tokens"] and generated[-1] not in (1, 50, 106)
                            else "stop", seconds=round(time.monotonic() - begin, 2)))
        print(json.dumps(dict(name=p["name"], prompt=len(tokens), generated=len(generated),
                              seconds=results[-1]["seconds"])), flush=True)
    if not a.dry_run:
        Path(a.output).write_text(json.dumps(dict(env=a.env, binary=binary, results=results), indent=1) + "\n")


def chat(url, model, ref, stream):
    body = dict(model=model, messages=ref["messages"], max_tokens=ref["max_tokens"], temperature=0, stream=stream)
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    begin = time.monotonic()
    first = None
    with urllib.request.urlopen(request, timeout=7200) as response:
        if not stream:
            data = json.load(response)
            return dict(text=data["choices"][0]["message"]["content"], finish=data["choices"][0]["finish_reason"],
                        completion_tokens=data["usage"]["completion_tokens"], seconds=time.monotonic() - begin)
        text, finish, usage = "", None, None
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                first = first or time.monotonic() - begin
                text += delta
            finish = chunk["choices"][0].get("finish_reason") or finish
            usage = chunk.get("usage") or usage
        return dict(text=text, finish=finish, completion_tokens=usage and usage["completion_tokens"],
                    seconds=time.monotonic() - begin, ttft=first)


def check(a):
    refs = json.loads(Path(a.reference).read_text())["results"]
    log_start = Path(a.server_log).stat().st_size if a.server_log else 0
    failures, rows = [], []

    def compare(ref, got, mode):
        ok = got["text"] == ref["text"] and got["completion_tokens"] == len(ref["cli_tokens"])
        rows.append(dict(name=ref["name"], mode=mode, ok=ok, seconds=round(got["seconds"], 2),
                         ttft=got.get("ttft") and round(got["ttft"], 3)))
        if not ok:
            failures.append(dict(name=ref["name"], mode=mode, cli=ref["text"], api=got["text"],
                                 cli_tokens=len(ref["cli_tokens"]), api_tokens=got["completion_tokens"]))

    # 1. Sequential, non-streamed then streamed.
    for stream in (False, True):
        for ref in refs:
            compare(ref, chat(a.url, a.model, ref, stream), "stream" if stream else "plain")
    # 2. Concurrent: every prompt `repeat` times at once, mixed streaming.
    jobs = [(ref, i % 2 == 0) for i, ref in enumerate(refs * a.repeat)]
    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool:
        futures = {pool.submit(chat, a.url, a.model, ref, stream): (ref, stream) for ref, stream in jobs}
        for future in concurrent.futures.as_completed(futures):
            ref, stream = futures[future]
            compare(ref, future.result(), f"concurrent-{a.concurrency}-{'stream' if stream else 'plain'}")
    report = dict(passed=not failures, checked=len(rows), failures=failures, rows=rows)
    # 3. Prefix cache: the worker logs "reused=N" for each request.
    if a.server_log:
        lines = Path(a.server_log).read_bytes()[log_start:].decode(errors="replace").splitlines()
        reused = [int(m.group(1)) for m in (re.search(r"Gemma native request:.*reused=(\d+)", x) for x in lines) if m]
        report["prefix_cache"] = dict(requests_logged=len(reused), hits=sum(r > 0 for r in reused),
                                      reused_tokens=sum(reused))
        # Hits are reported, not required: past 1,088 positions tt-lab can resume a slot only at a
        # 256-token point within one 64-position window block before its cached end (the local
        # window cannot roll back), so shared-document prompts with different questions miss.
        # tt-lab's gemma31_serve_batch_check.py --follow-up checks a hit's tokens natively.
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    if a.output:
        Path(a.output).write_text(json.dumps(report, indent=1) + "\n")
    sys.exit(0 if report["passed"] else 1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="phase", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--env", default=str(Path(__file__).with_name("server-batch.env")))
    r.add_argument("--binary", help="default: TT_LAB_BINARY from the env file")
    r.add_argument("--output", required=True)
    r.add_argument("--timeout", type=float, default=3600)
    r.add_argument("--dry-run", action="store_true", help="print the CLI commands; open no device")
    r.add_argument("--long-tokens", type=int, nargs="*", default=[],
                   help="extra long prompts of about this many tokens, e.g. 16000 30000 (needs MAX_MODEL_LENGTH)")
    c = sub.add_parser("check")
    c.add_argument("--reference", required=True)
    c.add_argument("--url", default="http://127.0.0.1:8002")
    c.add_argument("--model", default="google/gemma-4-31B-it")
    c.add_argument("--concurrency", type=int, default=8)
    c.add_argument("--repeat", type=int, default=2, help="copies of each prompt in the concurrent phase")
    c.add_argument("--server-log", help="server stderr log, to count prefix cache hits")
    c.add_argument("--output")
    a = p.parse_args()
    (reference if a.phase == "reference" else check)(a)


if __name__ == "__main__":
    main()
