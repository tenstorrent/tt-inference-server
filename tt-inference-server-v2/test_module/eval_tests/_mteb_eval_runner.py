# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Standalone MTEB runner executed inside the EVALS_EMBEDDING venv.

Invoked as a subprocess by ``embedding_eval_tests.run_embedding_eval`` so the
heavy ``mteb`` dependency stays out of the shared V2_RUN_SCRIPT venv (mirrors
the audio lmms-eval venv-targeting pattern). Emits parsed metrics as a JSON
object between the marker lines below.
"""

from __future__ import annotations

import argparse
import json
import sys

import mteb
import numpy as np
from mteb.models.model_implementations.openai_models import OpenAIModel
from mteb.models.model_meta import ModelMeta
from openai import OpenAI

MTEB_RESULT_START = "===MTEB_RESULT_JSON_START==="
MTEB_RESULT_END = "===MTEB_RESULT_JSON_END==="

_SCORE_KEYS = [
    "pearson",
    "spearman",
    "cosine_pearson",
    "cosine_spearman",
    "manhattan_pearson",
    "manhattan_spearman",
    "euclidean_pearson",
    "euclidean_spearman",
    "main_score",
    "languages",
]


def _parse_scores(results) -> dict:
    scores = results.task_results[0].scores["test"]
    if isinstance(scores, list) and len(scores) > 0:
        scores = scores[0]
    return {k: scores.get(k) for k in _SCORE_KEYS if k in scores}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--isl", type=int, required=True)
    ap.add_argument("--dimensions", type=int, required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    args = ap.parse_args()

    def single_string_encode(self, inputs, **kwargs):
        sentences = [text for batch in inputs for text in batch["text"]]
        all_embeddings = []
        for sentence in sentences:
            response = self._client.embeddings.create(
                input=sentence,
                model=args.model,
                encoding_format="float",
                dimensions=self._embed_dim if self._embed_dim else None,
            )
            all_embeddings.extend(self._to_numpy(response))
        return np.array(all_embeddings)

    client = OpenAI(base_url=f"{args.base_url}/v1", api_key=args.api_key)

    model = OpenAIModel(
        model_name=args.model,
        max_tokens=args.isl,
        embed_dim=args.dimensions,
        client=client,
    )
    model.encode = single_string_encode.__get__(model, type(model))

    model.mteb_model_meta = ModelMeta(
        name=args.model,
        revision=None,
        embed_dim=args.dimensions,
        max_tokens=args.isl,
        open_weights=False,
        loader=None,
        loader_kwargs={},
        framework=[],
        similarity_fn_name=None,
        use_instructions=None,
        release_date=None,
        languages=[],
        n_parameters=None,
        memory_usage_mb=None,
        license=None,
        public_training_code=None,
        public_training_data=None,
        training_datasets=None,
    )

    tasks = mteb.get_tasks(tasks=args.tasks)
    results = mteb.evaluate(
        model,
        tasks=tasks,
        encode_kwargs={"batch_size": 1},
        cache=None,
        overwrite_strategy="always",
    )

    print(MTEB_RESULT_START)
    print(json.dumps(_parse_scores(results)))
    print(MTEB_RESULT_END)
    return 0


if __name__ == "__main__":
    sys.exit(main())
