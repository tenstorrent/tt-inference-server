# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ..context import MediaContext, require_health

logger = logging.getLogger(__name__)


OPENAI_API_KEY = "your-secret-key"
MTEB_TASKS = ["STS12"]
EMBEDDING_DIMENSIONS = 1000


def _embedding_model_config(ctx: MediaContext) -> tuple[str, int, int]:
    """Return (hf_model_repo, isl, dimensions) derived from model_spec env vars."""
    env = ctx.model_spec.device_model_spec.env_vars
    return (
        ctx.model_spec.hf_model_repo,
        int(env.get("VLLM__MAX_MODEL_LENGTH", 1024)),
        EMBEDDING_DIMENSIONS,
    )


def _parse_embedding_evals_output(results: Any) -> dict:
    """Parse MTEB evaluation results, extracting key metrics from scores['test']."""
    try:
        scores = results.task_results[0].scores["test"]
        if isinstance(scores, list) and len(scores) > 0:
            scores = scores[0]
    except Exception as e:
        logger.error(f"Could not extract scores['test']: {e}")
        raise

    keys = [
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
    report_data = {k: scores.get(k) for k in keys if k in scores}
    logger.info(f"Parsed evaluation results: {report_data}")
    return report_data


def _run_embedding_transcription_eval(ctx: MediaContext) -> dict:
    """Run MTEB eval against the embedding endpoint."""
    import mteb
    import numpy as np
    from mteb.models.model_implementations.openai_models import OpenAIModel
    from mteb.models.model_meta import ModelMeta
    from openai import OpenAI

    model_name, isl, dimensions = _embedding_model_config(ctx)

    def single_string_encode(self, inputs, **kwargs):
        sentences = [text for batch in inputs for text in batch["text"]]
        all_embeddings = []
        for sentence in sentences:
            response = self._client.embeddings.create(
                input=sentence,
                model=model_name,
                encoding_format="float",
                dimensions=self._embed_dim if self._embed_dim else None,
            )
            all_embeddings.extend(self._to_numpy(response))
        return np.array(all_embeddings)

    client = OpenAI(base_url=f"{ctx.base_url}/v1", api_key=OPENAI_API_KEY)

    model = OpenAIModel(
        model_name=model_name,
        max_tokens=isl,
        embed_dim=dimensions,
        client=client,
    )
    model.encode = single_string_encode.__get__(model, type(model))

    model_meta = ModelMeta(
        name=model_name,
        revision=None,
        embed_dim=dimensions,
        max_tokens=isl,
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
    model.mteb_model_meta = model_meta

    tasks = mteb.get_tasks(tasks=MTEB_TASKS)

    logger.info("Running embedding transcription evaluation with STS12...")
    results = mteb.evaluate(
        model,
        tasks=tasks,
        encode_kwargs={"batch_size": 1},
        cache=None,
        overwrite_strategy="always",
    )
    logger.info(f"Evaluation results: {results}")
    return _parse_embedding_evals_output(results)


def run_embedding_eval(ctx: MediaContext) -> dict:
    """Run evaluations for an embedding model."""
    logger.info(
        f"Running evals for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx)

    try:
        logger.info("Running embedding eval...")
        metrics = _run_embedding_transcription_eval(ctx)
    except Exception as e:
        logger.error(f"Eval execution encountered an error: {e}")
        raise

    logger.info("Generating evals report...")
    report_data = {
        "model": ctx.model_spec.model_name,
        "device": ctx.device.name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "task_type": "embedding",
        "task_name": ctx.all_params.tasks[0].task_name,
    }
    report_data.update(metrics)

    return report_data


__all__ = ["run_embedding_eval"]
