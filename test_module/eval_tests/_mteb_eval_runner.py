# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""MTEB runner executed inside the EVALS_EMBEDDING venv.

The heavy ``mteb`` dependency is provisioned into its own venv, so it cannot be imported into the
shared V2_RUN_SCRIPT venv. ``embedding_eval_tests.run_embedding_eval`` therefore
launches this module as a subprocess in that venv; the ``__main__`` guard below
is only a thin CLI adapter that hands off to :class:`MtebEvalRunner`.

Metrics are emitted as a JSON object between the marker lines so the parent
process can recover them from stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import mteb
import numpy as np
from mteb.models.model_implementations.openai_models import OpenAIModel
from mteb.models.model_meta import ModelMeta
from openai import OpenAI

MTEB_RESULT_START = "===MTEB_RESULT_JSON_START==="
MTEB_RESULT_END = "===MTEB_RESULT_JSON_END==="


class SingleStringOpenAIModel(OpenAIModel):
    """``OpenAIModel`` that embeds one sentence per request.

    MTEB hands ``encode`` a batch of inputs, but the served endpoint expects a
    single string per request. Overriding ``encode`` in a subclass replaces the
    previous instance-level monkeypatch (``model.encode = fn.__get__(...)``).
    """

    def __init__(
        self,
        *,
        model_name: str,
        max_tokens: int,
        embed_dim: int,
        client: OpenAI,
    ) -> None:
        super().__init__(
            model_name=model_name,
            max_tokens=max_tokens,
            embed_dim=embed_dim,
            client=client,
        )
        # OpenAIModel keeps the name privately; hold our own copy so ``encode``
        # does not depend on the base class's attribute naming.
        self._eval_model_name = model_name
        self.mteb_model_meta = self._build_model_meta(model_name, embed_dim, max_tokens)

    @staticmethod
    def _build_model_meta(
        model_name: str, embed_dim: int, max_tokens: int
    ) -> ModelMeta:
        return ModelMeta(
            name=model_name,
            revision=None,
            embed_dim=embed_dim,
            max_tokens=max_tokens,
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

    def encode(self, inputs, **kwargs) -> np.ndarray:
        sentences = [text for batch in inputs for text in batch["text"]]
        embeddings = []
        for sentence in sentences:
            response = self._client.embeddings.create(
                input=sentence,
                model=self._eval_model_name,
                encoding_format="float",
                dimensions=self._embed_dim if self._embed_dim else None,
            )
            embeddings.extend(self._to_numpy(response))
        return np.array(embeddings)


@dataclass
class MtebEvalConfig:
    """Parameters for a single MTEB evaluation run."""

    base_url: str
    model: str
    isl: int
    dimensions: int
    api_key: str
    tasks: list[str]

    @classmethod
    def from_argv(cls, argv: list[str] | None = None) -> "MtebEvalConfig":
        ap = argparse.ArgumentParser(description="Run an MTEB embedding evaluation.")
        ap.add_argument("--base-url", required=True)
        ap.add_argument("--model", required=True)
        ap.add_argument("--isl", type=int, required=True)
        ap.add_argument("--dimensions", type=int, required=True)
        ap.add_argument("--api-key", required=True)
        ap.add_argument("--tasks", nargs="+", required=True)
        args = ap.parse_args(argv)
        return cls(
            base_url=args.base_url,
            model=args.model,
            isl=args.isl,
            dimensions=args.dimensions,
            api_key=args.api_key,
            tasks=args.tasks,
        )


class MtebEvalRunner:
    """Runs an MTEB evaluation against an OpenAI-compatible embedding endpoint."""

    _SCORE_KEYS = (
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
    )

    def __init__(self, config: MtebEvalConfig) -> None:
        self.config = config

    def _build_model(self) -> SingleStringOpenAIModel:
        client = OpenAI(
            base_url=f"{self.config.base_url}/v1", api_key=self.config.api_key
        )
        return SingleStringOpenAIModel(
            model_name=self.config.model,
            max_tokens=self.config.isl,
            embed_dim=self.config.dimensions,
            client=client,
        )

    def run(self) -> dict:
        """Execute the evaluation and return the parsed metric scores."""
        model = self._build_model()
        tasks = mteb.get_tasks(tasks=self.config.tasks)
        results = mteb.evaluate(
            model,
            tasks=tasks,
            encode_kwargs={"batch_size": 1},
            cache=None,
            overwrite_strategy="always",
        )
        return self._parse_scores(results)

    @classmethod
    def _parse_scores(cls, results) -> dict:
        scores = results.task_results[0].scores["test"]
        if isinstance(scores, list) and scores:
            scores = scores[0]
        return {k: scores[k] for k in cls._SCORE_KEYS if k in scores}


def main(argv: list[str] | None = None) -> int:
    config = MtebEvalConfig.from_argv(argv)
    scores = MtebEvalRunner(config).run()
    print(MTEB_RESULT_START)
    print(json.dumps(scores))
    print(MTEB_RESULT_END)
    return 0


if __name__ == "__main__":
    sys.exit(main())
