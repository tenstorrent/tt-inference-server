# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Phase 1 baseline: prove an embedding runner works from plain Python.

Run from the tt-media-server directory with the tt-metal venv active and
PYTHONPATH including the tt-metal repo root (for models.demos imports):

    PYTHONPATH=$TT_METAL_HOME python scripts/embedding_python_baseline.py [model]

where [model] is one of: bge-large (default), bge-m3, qwen3-8b.

Environment (MODEL / DEVICE) must be set before importing anything from
`config`, because config.settings builds its Settings singleton at import
time. This script sets them itself, so no exports are needed.
"""

import os
import sys
import time

# {cli name: (ModelNames value for MODEL env, runner class, HF model id)}
BASELINE_MODELS = {
    "bge-large": ("bge-large-en-v1.5", "BGELargeENRunner", "BAAI/bge-large-en-v1.5"),
    "bge-m3": ("bge-m3", "BGEM3Runner", "BAAI/bge-m3"),
    "qwen3-8b": (
        "Qwen3-Embedding-8B",
        "Qwen3Embedding8BRunner",
        "Qwen/Qwen3-Embedding-8B",
    ),
}

_choice = sys.argv[1] if len(sys.argv) > 1 else "bge-large"
if _choice not in BASELINE_MODELS:
    print(f"unknown model {_choice!r}; expected one of {sorted(BASELINE_MODELS)}")
    sys.exit(2)
_model_env, _runner_class, MODEL_ID = BASELINE_MODELS[_choice]

# Must happen before any tt-media-server import (Settings reads env at import).
os.environ.setdefault("MODEL", _model_env)
os.environ.setdefault("DEVICE", "n150")
os.environ.setdefault("DEVICE_IDS", "(0)")

import asyncio  # noqa: E402

from domain.text_embedding_request import TextEmbeddingRequest  # noqa: E402

import tt_model_runners.embedding_runner as _runners  # noqa: E402

PROMPT = "The quick brown fox jumps over the lazy dog."


def main() -> int:
    print(f"[baseline] constructing {_runner_class}('0')", flush=True)
    runner = getattr(_runners, _runner_class)("0")

    print("[baseline] set_device() ...", flush=True)
    t0 = time.time()
    runner.set_device()
    print(f"[baseline] set_device() done in {time.time() - t0:.1f}s", flush=True)

    print(
        "[baseline] warmup() (loads weights, may download them first) ...", flush=True
    )
    t0 = time.time()
    ok = asyncio.run(runner.warmup())
    print(f"[baseline] warmup() -> {ok} in {time.time() - t0:.1f}s", flush=True)
    if not ok:
        return 1

    print("[baseline] run() one request ...", flush=True)
    t0 = time.time()
    responses = runner.run([TextEmbeddingRequest(model=MODEL_ID, input=PROMPT)])
    print(f"[baseline] run() done in {time.time() - t0:.3f}s", flush=True)

    emb = responses[0].embedding
    print(f"[baseline] dim={len(emb)} total_tokens={responses[0].total_tokens}")
    print(f"[baseline] first 5 values: {emb[:5]}")

    runner.close_device()
    print("[baseline] SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
