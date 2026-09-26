"""Gemma 31B four-card runner settings: validation limits and device selection; no device open."""
import os
from pathlib import Path
import sys
import pytest

os.environ.update(MODEL_RUNNER='tt-lab-gemma', MODEL_SERVICE='llm', IS_GALAXY='false', DEVICE_IDS='(0)',
                  MODEL_WEIGHTS_PATH='/home/ttuser/workspace/gemma31-reference',
                  TT_LAB_SERVED_MODEL='google/gemma-4-31B-it', TT_LAB_CONTEXT='32768')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tt-media-server'))
from transformers import AutoTokenizer
from domain.completion_request import CompletionRequest
from domain.tt_lab_validation import validate_request


@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained(os.environ['MODEL_WEIGHTS_PATH'], local_files_only=True)


def test_31b_model_name(tokenizer):
    validate_request(CompletionRequest(model='google/gemma-4-31B-it', prompt='Hi', max_tokens=4), tokenizer)
    with pytest.raises(ValueError, match='serves only google/gemma-4-31B-it'):
        validate_request(CompletionRequest(model='google/gemma-4-26B-A4B-it', prompt='Hi', max_tokens=4), tokenizer)


def test_31b_context_limit(tokenizer):
    tokens, limit = validate_request(CompletionRequest(prompt=[2] * 30000, max_tokens=2000), tokenizer)
    assert len(tokens) == 30000 and limit == 2000
    with pytest.raises(ValueError, match='prompt \\+ max_tokens <= 32768'):
        validate_request(CompletionRequest(prompt=[2] * 32000, max_tokens=1000), tokenizer)


def test_four_cards_refuses_single_card_selection(monkeypatch):
    from tt_model_runners.tt_lab_runner import TTLabRunner
    monkeypatch.setenv('TT_LAB_FOUR_CARDS', '1')
    monkeypatch.setenv('TT_LAB_DEVICE', '1')
    with pytest.raises(ValueError, match='unset TT_LAB_DEVICE'):
        TTLabRunner('0')
