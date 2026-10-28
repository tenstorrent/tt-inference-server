# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Rerank service (request -> one pair request per document), the /rerank and /score responses, and the
nemotron rerank runner's tokenize-and-score path, with the model and scheduler stubbed out."""

import asyncio
import math
from unittest.mock import MagicMock, patch

import pytest
from domain.rerank_request import RerankPairRequest, RerankRequest, ScoreRequest
from domain.rerank_response import RerankPairResponse
from pydantic import ValidationError

MODEL = "nvidia/llama-nemotron-rerank-1b-v2"


@pytest.fixture(autouse=True)
def _rerank_runner_settings():
    """Request validation accepts the model names of the running runner: pin it to nemotron-rerank."""
    from config.settings import get_settings

    with patch.object(get_settings(), "model_runner", "nemotron-rerank"):
        yield


def _service():
    with patch("model_services.base_service.get_scheduler", return_value=MagicMock()):
        from model_services.rerank_service import RerankService

        return RerankService()


class FakeService:
    """Stands in for RerankService: one logit per document, from a fixed table."""

    def __init__(self, logits):
        self.logits = logits
        self.requests = []

    async def process_request(self, request):
        self.requests.append(request)
        return [
            RerankPairResponse(logit=self.logits[d], total_tokens=len(d))
            for d in request.document_texts()
        ]


class TestRerankService:
    def test_pre_process_splits_documents(self):
        svc = _service()
        req = RerankRequest(model=MODEL, query="q", documents=["a", {"text": "b"}])
        out = asyncio.run(svc.pre_process(req))
        assert out._segments == ["a", "b"]

    def test_empty_documents_rejected_at_validation(self):
        with pytest.raises(ValidationError):
            RerankRequest(model=MODEL, query="q", documents=[])

    def test_segment_request_is_one_pair(self):
        svc = _service()
        req = RerankRequest(model=MODEL, query="q", documents=["a", "b"])
        pair = svc.create_segment_request(req, "b", 1)
        assert isinstance(pair, RerankPairRequest)
        assert (pair.model, pair.query, pair.document) == (MODEL, "q", "b")
        assert pair._task_id != req._task_id

    def test_combine_keeps_document_order(self):
        svc = _service()
        results = [RerankPairResponse(1.0, 3), RerankPairResponse(-2.0, 4)]
        assert svc.combine_results(results) == results


class TestRerankApi:
    def test_rerank_sorts_and_cuts(self):
        from open_ai_api.rerank import rerank

        svc = FakeService({"low": -1.0, "high": 2.0, "mid": 0.5})
        req = RerankRequest(
            model=MODEL,
            query="q",
            documents=["low", "high", "mid"],
            top_n=2,
            return_documents=True,
        )
        out = asyncio.run(rerank(req, service=svc, api_key="k"))
        assert [r["index"] for r in out["results"]] == [1, 2]
        assert out["results"][0]["logit"] == 2.0
        assert out["results"][0]["relevance_score"] == pytest.approx(
            1 / (1 + math.exp(-2.0))
        )
        assert out["results"][0]["document"] == {"text": "high"}
        assert out["usage"]["total_tokens"] == len("low") + len("high") + len("mid")

    def test_short_and_hf_model_names_accepted(self):
        RerankRequest(model=MODEL, query="q", documents=["d"])
        ScoreRequest(model="llama-nemotron-rerank-1b-v2", text_1="q", text_2="d")

    def test_other_model_rejected_at_validation(self):
        with pytest.raises(ValidationError):
            RerankRequest(model="BAAI/bge-m3", query="q", documents=["d"])
        with pytest.raises(ValidationError):
            ScoreRequest(model="BAAI/bge-m3", text_1="q", text_2="d")

    def test_score_empty_list_rejected_at_validation(self):
        with pytest.raises(ValidationError):
            ScoreRequest(model=MODEL, text_1="q", text_2=[])

    def test_score_keeps_input_order(self):
        from open_ai_api.rerank import score

        svc = FakeService({"a": 3.0, "b": -3.0})
        out = asyncio.run(
            score(
                ScoreRequest(model=MODEL, text_1="q", text_2=["a", "b"]),
                service=svc,
                api_key="k",
            )
        )
        assert [d["index"] for d in out["data"]] == [0, 1]
        assert out["data"][0]["score"] > 0.9 > 0.1 > out["data"][1]["score"]

    def test_score_single_document(self):
        from open_ai_api.rerank import score

        out = asyncio.run(
            score(
                ScoreRequest(model=MODEL, text_1="q", text_2="a"),
                service=FakeService({"a": 0.0}),
                api_key="k",
            )
        )
        assert out["data"] == [{"index": 0, "object": "score", "score": 0.5}]


class TestNemotronRerankRunner:
    def _runner(self):
        from tt_model_runners.rerank_runner import NemotronRerankRunner

        with patch("tt_model_runners.base_device_runner.setup_runner_environment"):
            runner = NemotronRerankRunner("0")
        runner.max_model_len = 8
        runner.prompt_template = lambda q, p: f"question:{q} \n \n passage:{p}"
        runner.tokenizer = MagicMock(
            side_effect=lambda texts, truncation, max_length: {
                "input_ids": [list(range(min(len(t), max_length))) for t in texts]
            }
        )
        runner.model = MagicMock()
        runner.model.score_ids = MagicMock(
            side_effect=lambda ids: [float(len(x)) - 4 for x in ids]
        )
        return runner

    def test_run_scores_pairs_in_order(self):
        runner = self._runner()
        reqs = [
            RerankPairRequest(model=MODEL, query="q", document="d"),
            RerankPairRequest(
                model="llama-nemotron-rerank-1b-v2", query="q", document="d"
            ),
        ]
        out = runner.run(reqs)  # both the Hugging Face id and the short name are served
        ids = runner.model.score_ids.call_args[0][0]
        assert all(len(x) == 8 for x in ids)  # truncated to max_model_len
        assert [o.logit for o in out] == [4.0, 4.0]
        assert [o.total_tokens for o in out] == [8, 8]

    def test_run_rejects_other_models(self):
        runner = self._runner()
        with pytest.raises(ValueError):
            runner.run(
                [RerankPairRequest(model="BAAI/bge-m3", query="q", document="d")]
            )


def test_constants_register_nemotron_rerank():
    from config.constants import (
        MODEL_RUNNER_TO_MODEL_NAMES_MAP,
        MODEL_SERVICE_RUNNER_MAP,
        DeviceTypes,
        ModelConfigs,
        ModelNames,
        ModelRunners,
        ModelServices,
    )

    runner = ModelRunners.NEMOTRON_RERANK
    assert runner in MODEL_SERVICE_RUNNER_MAP[ModelServices.RERANK]
    assert MODEL_RUNNER_TO_MODEL_NAMES_MAP[runner] == {ModelNames.NEMOTRON_RERANK_1B_V2}
    config = ModelConfigs[(runner, DeviceTypes.P150)]
    # The worker sets TT_MM_THROTTLE_PERF only from a truthy value: "0", not 0.
    assert config["default_throttle_level"] == "0"
