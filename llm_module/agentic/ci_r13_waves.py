# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unscored frozen-query replay, restricted to the diagnostic TTI branch."""

import gzip
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


def adapt_client(source):
    """Preserve request code exactly; redirect transport and relabel metadata."""
    replacements = [
        (
            "'http://localhost:8000/v1/chat/completions'",
            "INJECTED_ENDPOINT+'/v1/chat/completions'",
            1,
        ),
        (
            "'http://127.0.0.1:8000/v1/completions'",
            "INJECTED_ENDPOINT+'/v1/completions'",
            1,
        ),
        ("'http://127.0.0.1:8000'", "INJECTED_ENDPOINT", 2),
        ("headers={'Content-Type':'application/json'}", "headers=INJECTED_HEADERS", 4),
    ]
    adapted = source
    for before, after, count in replacements:
        if adapted.count(before) != count:
            raise ValueError(f"Frozen client transport count changed: {before}")
        adapted = adapted.replace(before, after)
    marker = "shared_rows=[]"
    metadata = """shared_metadata = {
 'model': shared_metadata['model'],
 'requested_revision': '842da3794eaa0b77d5f08bae87a17459d91ff475',
 'reference_local_metadata': shared_metadata,
 'revision_scope': 'Nested reference metadata describes the original local control only, not this CI filesystem. Actual CI loaded identity requires separate runtime/source artifacts. Endpoint prompt token arrays are retained; replay verifies reconstructed-token hashes.',
 'prompt_mode': 'chat',
 'endpoint': '/v1/chat/completions',
 'control': 'Paired immutable CI diagnostic required; no HF run or scored agent evaluation',
}
"""
    if adapted.count(marker) != 1:
        raise ValueError("Frozen client metadata marker changed")
    adapted = adapted.replace(marker, metadata + marker)
    reversed_source = adapted.replace(metadata, "", 1)
    for before, after, _ in replacements:
        reversed_source = reversed_source.replace(after, before)
    if reversed_source != source:
        raise ValueError("Request code changed during transport adaptation")
    prelude = """import os
INJECTED_ENDPOINT = os.environ['GEMMA_PROBE_ENDPOINT'].rstrip('/')
INJECTED_HEADERS = {'Content-Type': 'application/json'}
if os.environ.get('GEMMA_PROBE_AUTH_TOKEN'):
 INJECTED_HEADERS['Authorization'] = 'Bearer ' + os.environ['GEMMA_PROBE_AUTH_TOKEN']
"""
    return prelude + adapted


def validate_outputs(root, tag):
    def read(suffix):
        return json.loads((root / (tag + suffix)).read_text())

    replay = read("-replay.json")
    if [w["n"] for w in replay["waves"]] != [128, 256, 512]:
        raise ValueError("Incomplete replay waves")
    for wave in replay["waves"]:
        keys = {(r["actor"], r["turn"]) for r in wave["records"]}
        expected = {(a, t) for a in range(32) for t in range(wave["n"] // 32)}
        if keys != expected or len(wave["records"]) != wave["n"]:
            raise ValueError("Missing or duplicate query work")
    reconstructed = read("-reconstruction-check.json")
    if len(reconstructed) != 512 or not all(r["exact"] for r in reconstructed):
        raise ValueError("Incomplete reconstruction gate")
    sampled = read("-concurrent-sampling.json")
    if len(sampled) != 3 or any(
        len(w["prime"]) != 16 or len(w["data"]) != 16 for w in sampled
    ):
        raise ValueError("Incomplete controlled sampling gate")
    natural = read("-shared-qualitative.json")
    if len(natural["rows"]) != 6:
        raise ValueError("Incomplete natural-output gate")
    for row in natural["rows"]:
        choice = row["response"]["choices"][0]
        if choice["finish_reason"] != "stop" or not choice.get("token_ids"):
            raise ValueError("Natural output truncated or missing token IDs")
    return [{k: v for k, v in w.items() if k != "records"} for w in replay["waves"]]


def run_probe(server, context):
    if server.model != "google/gemma-4-31B-it":
        raise ValueError("Wrong model for frozen replay")
    data = Path(__file__).resolve().parent / "formation_probe_data"
    manifest = json.loads((data / "manifest.json").read_text())
    root = Path(context.output_dir) / "UNSCORED-formation-probe"
    root.mkdir(parents=True, exist_ok=False)
    tag = "ci-formation"
    try:
        for name, expected in manifest["files"].items():
            compressed = (data / (name + ".gz")).read_bytes()
            if hashlib.sha256(compressed).hexdigest() != expected["gzip_sha256"]:
                raise ValueError(f"Compressed source hash mismatch: {name}")
            payload = gzip.decompress(compressed)
            if (
                len(payload) != expected["bytes"]
                or hashlib.sha256(payload).hexdigest() != expected["sha256"]
            ):
                raise ValueError(f"Source hash mismatch: {name}")
            (root / name).write_bytes(payload)
        source = (root / "replay_with_sampling32_shared.py").read_text()
        adapted = adapt_client(source)
        client = root / "ci_client.py"
        client.write_text(adapted)
        (root / "source-provenance.json").write_text(
            json.dumps(
                {
                    "manifest": manifest,
                    "adapted_client_sha256": hashlib.sha256(
                        adapted.encode()
                    ).hexdigest(),
                    "adaptation": "Exact reverse-proven transport and reference-metadata changes only",
                },
                indent=2,
            )
        )
        env = dict(os.environ)
        env["GEMMA_PROBE_ENDPOINT"] = server.url_with_port.rstrip("/")
        env["GEMMA_PROBE_AUTH_TOKEN"] = server.auth_token or ""
        with (root / "client.log").open("w") as log:
            subprocess.run(
                [sys.executable, "-u", str(client), tag],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=2700,
                check=True,
            )
        waves = validate_outputs(root, tag)
        summary = {
            "unscored_policy_probe": True,
            "status": "completed",
            "accuracy": "NOT MEASURED: fixed-query diagnostic, no agent feedback",
            "correctness_scope": "Local gates passed; paired CI token-array comparison and human natural-output review still required",
            "waves": waves,
            "artifact_directory": str(root),
        }
        path = root / "summary.json"
        path.write_text(json.dumps(summary, indent=2))
        return summary, path
    except Exception:
        (root / "error.txt").write_text(traceback.format_exc())
        raise
