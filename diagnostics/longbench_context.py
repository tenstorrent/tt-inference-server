# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""One-off paired LongBench replay. Diagnostic branch only; never qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
IMAGE = "ghcr.io/tenstorrent/tt-shield/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:3948820d5e28350fa94f8c4ccd7b2e53d1de8b756d745c1c39c9e8cd1af075b1"
TOKENIZER_SHA = "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4"
TOKENIZER_CONFIG_SHA = (
    "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_payload(tokens, kwargs):
    expected = {
        "max_gen_toks": 512,
        "temperature": 0,
        "do_sample": False,
        "until": [],
        "stream": False,
        "seed": 42,
    }
    if kwargs != expected:
        raise ValueError("Saved generation settings differ from the fixed comparison")
    return {
        "prompt": [tokens],
        "model": MODEL,
        "max_tokens": 512,
        "temperature": 0,
        "stop": [],
        "seed": 42,
        "stream": False,
    }


def prepare(source, tokenizer_path, manifest):
    from tokenizers import Tokenizer

    if sha(tokenizer_path) != TOKENIZER_SHA:
        raise ValueError("Tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    requests = []
    loaded = {}
    if len(manifest["cases"]) != 24:
        raise ValueError("Expected 24 fixed cases")
    for case in manifest["cases"]:
        name = Path(case["sample_file"]).name
        if name not in loaded:
            files = list(Path(source).rglob(name))
            if len(files) != 1 or sha(files[0]) != case["sample_file_sha256"]:
                raise ValueError(f"Missing, duplicate or changed sample file: {name}")
            rows = [
                json.loads(line) for line in files[0].read_text().split("\n") if line
            ]
            loaded[name] = {row["doc_id"]: row for row in rows}
            if len(loaded[name]) != len(rows):
                raise ValueError("Duplicate sample IDs")
        row = loaded[name][case["doc_id"]]
        if (
            row["doc"]["_id"] != case["source_id"]
            or row["prompt_hash"] != case["prompt_hash"]
        ):
            raise ValueError("Sample identity mismatch")
        args = row["arguments"]["gen_args_0"]
        tokens = tokenizer.encode(args["arg_0"], add_special_tokens=False).ids
        if len(tokens) != case["full_input_tokens"] or len(tokens) + 512 >= 131072:
            raise ValueError("Unexpected input length")
        for arm, ids in [("original_suffix", tokens[-1535:]), ("full_prompt", tokens)]:
            payload = build_payload(ids, args["arg_1"])
            if digest(payload) != case["arm_payload_sha256"][arm]:
                raise ValueError("Payload differs from the prepared comparison")
            requests.append(
                {
                    "case_id": case["case_id"],
                    "arm": arm,
                    "input_tokens": len(ids),
                    "payload": payload,
                    "payload_sha256": digest(payload),
                }
            )
    if len(requests) != 48 or len({r["case_id"] for r in requests}) != 24:
        raise ValueError("Diagnostic request count mismatch")
    return requests


def validate_response(data, prompt_tokens):
    if data.get("error") or len(data.get("choices", [])) != 1:
        raise ValueError("Expected one successful completion")
    choice = data["choices"][0]
    if (
        choice.get("index") != 0
        or not isinstance(choice.get("text"), str)
        or choice.get("finish_reason") not in ("stop", "length")
    ):
        raise ValueError("Invalid or unfinished completion")
    usage = data.get("usage", {})
    output = usage.get("completion_tokens")
    if (
        type(output) is not int
        or not 1 <= output <= 512
        or usage.get("prompt_tokens") != prompt_tokens
        or usage.get("total_tokens") != prompt_tokens + output
    ):
        raise ValueError("Returned token usage does not match the fixed request")


def validate_runtime(observed):
    if observed["image"] != IMAGE or not observed["weights_path"].endswith(
        "/snapshots/" + REVISION
    ):
        raise ValueError("Wrong image or checkpoint")
    log = observed["log"]
    required = [
        "models.demos.llama31_8b_qb2.tt.generator_vllm.LlamaForCausalLM",
        "gu4_head8_lm_head_hifi2",
    ]
    if any(text not in log for text in required):
        raise ValueError("Intended model generator/precision is not observed")
    for name, value in [
        ("max_model_len", "131072"),
        ("max_num_seqs", "32"),
        ("seed", "0"),
        ("revision", REVISION),
        ("tokenizer_revision", REVISION),
    ]:
        pattern = r"--" + name.replace("_", "[-_]") + r"[ =]+" + value + r"(?:\s|$)"
        if not re.search(pattern, log):
            raise ValueError("Runtime argument not verified: " + name)


def observe_runtime():
    ids = subprocess.check_output(
        [
            "docker",
            "ps",
            "--filter",
            "name=tt-inference-server-",
            "--format",
            "{{.ID}}",
        ],
        text=True,
    ).split()
    if len(ids) != 1:
        raise ValueError("Expected exactly one task server container")
    info = json.loads(
        subprocess.check_output(["docker", "inspect", ids[0]], text=True)
    )[0]
    env = dict(item.split("=", 1) for item in info["Config"]["Env"] if "=" in item)
    logs = subprocess.run(
        ["docker", "logs", "--tail", "2000", ids[0]],
        capture_output=True,
        text=True,
        check=True,
    )
    observed = {
        "container": info["Name"],
        "image": info["Config"]["Image"],
        "image_id": info["Image"],
        "weights_path": env.get("MODEL_WEIGHTS_DIR", ""),
        "log": logs.stdout + logs.stderr,
    }
    validate_runtime(observed)
    observed.pop(
        "log"
    )  # Never persist arbitrary container environment or log contents here.
    observed.update(
        generator="models.demos.llama31_8b_qb2.tt.generator_vllm.LlamaForCausalLM",
        precision="gu4_head8_lm_head_hifi2",
        max_model_len=131072,
        max_num_seqs=32,
        seed=0,
        checkpoint_revision=REVISION,
        tokenizer_revision=REVISION,
    )
    return observed


def replay(requests, output, transport):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Never overwrite or resume an uncertain device request.
    with (output / "responses.jsonl").open("x") as saved:
        for index, request in enumerate(requests, 1):
            start = time.monotonic()
            record = {k: v for k, v in request.items() if k != "payload"}
            try:
                record["response"] = transport(request["payload"])
            except Exception as error:
                record["transport_error_type"] = type(error).__name__
                record["elapsed_seconds"] = time.monotonic() - start
                saved.write(json.dumps(record) + "\n")
                saved.flush()
                os.fsync(saved.fileno())
                raise
            record["elapsed_seconds"] = time.monotonic() - start
            saved.write(json.dumps(record) + "\n")
            saved.flush()
            os.fsync(saved.fileno())
            validate_response(record["response"], request["input_tokens"])
            print(
                f"Diagnostic {index}/{len(requests)}: {request['case_id']} {request['arm']}, input={request['input_tokens']}, output={record['response']['usage']['completion_tokens']}",
                flush=True,
            )
    (output / "COMPLETE.json").write_text(
        json.dumps(
            {
                "diagnostic_only": True,
                "qualification": False,
                "requests": len(requests),
                "responses_sha256": sha(output / "responses.jsonl"),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(__file__).with_name("longbench_cases.json")
    manifest = json.loads(manifest_path.read_text())
    if args.tokenizer:
        tokenizer = args.tokenizer
    else:
        from huggingface_hub import hf_hub_download

        tokenizer = Path(hf_hub_download(MODEL, "tokenizer.json", revision=REVISION))
        config = Path(
            hf_hub_download(MODEL, "tokenizer_config.json", revision=REVISION)
        )
        if sha(config) != TOKENIZER_CONFIG_SHA:
            raise ValueError("Tokenizer configuration hash mismatch")
    requests = prepare(args.source, tokenizer, manifest)
    (args.output / "requests.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in requests)
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "tokenizer_identity.json").write_text(
        json.dumps({"revision": REVISION, "sha256": sha(tokenizer)}) + "\n"
    )
    if args.prepare_only:
        print("Prepared 48 exact payloads; no requests sent")
        return
    url = urlparse(args.base_url or "")
    if (
        url.scheme != "http"
        or url.hostname not in ("localhost", "127.0.0.1")
        or url.path not in ("", "/")
    ):
        raise ValueError("Diagnostic must use the task's local CI server")
    runtime = observe_runtime()
    (args.output / "runtime_identity.json").write_text(
        json.dumps(runtime, indent=2) + "\n"
    )
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]

    def transport(payload):
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)

    models_request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/models", headers=headers
    )
    with urllib.request.urlopen(models_request, timeout=30) as response:
        models = json.load(response)
    if MODEL not in [model.get("id") for model in models.get("data", [])]:
        raise ValueError("Expected served model is absent")
    (args.output / "served_models.json").write_text(json.dumps(models, indent=2) + "\n")
    replay(requests, args.output, transport)


if __name__ == "__main__":
    main()
