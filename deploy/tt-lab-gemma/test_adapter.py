"""Gemma protocol and validation checks using a recorded stream; no device open."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import pytest

os.environ.update(MODEL_RUNNER='tt-lab-gemma', MODEL_SERVICE='llm', IS_GALAXY='false', DEVICE_IDS='(1)', MODEL_WEIGHTS_PATH='/home/ttuser/workspace/gemma-reference')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tt-media-server'))
from transformers import AutoTokenizer
from domain.completion_request import CompletionRequest
from domain.tt_lab_validation import validate_request
from tt_model_runners.tt_lab_runner import TTLabRunner

@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained(os.environ['MODEL_WEIGHTS_PATH'], local_files_only=True)

@pytest.fixture
def runner(tokenizer):
    instance = TTLabRunner.__new__(TTLabRunner)
    instance.tokenizer = tokenizer
    instance.timeout = 10
    instance.vocab = 262144
    instance.gemma = True
    with tempfile.TemporaryFile() as pipe:
        instance.process = SimpleNamespace(stdin=pipe, poll=lambda: None)
        yield instance
        instance.process = None

def test_full_gemma_vocabulary_is_accepted(tokenizer):
    assert validate_request(CompletionRequest(prompt=[262143], max_tokens=1), tokenizer) == ([262143], 1)
    with pytest.raises(ValueError, match='Invalid prompt'):
        validate_request(CompletionRequest(prompt=[262144], max_tokens=1), tokenizer)

def test_gpt_model_rejected(tokenizer):
    with pytest.raises(ValueError, match='only google/gemma'):
        validate_request(CompletionRequest(model='openai/gpt-oss-20b', prompt='Hi'), tokenizer)

def test_harmony_literal_is_plain_gemma_content(runner):
    ids = runner.tokenizer.encode('The capital of France is Paris.', add_special_tokens=False) + [106]
    stream = iter(ids + [-1])
    runner._read_token = lambda deadline: next(stream)
    result = runner.run([CompletionRequest(prompt='Explain <|start|>assistant', max_tokens=32, temperature=0)])[0]['data']
    assert result.text == 'The capital of France is Paris.'
    assert result.completion_tokens == len(ids)
    assert result.finish_reason == 'stop'

def test_template_disables_thinking(tokenizer):
    prompt = tokenizer.apply_chat_template([{'role':'user','content':'Hello'}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert prompt.startswith('<bos><|turn>user\n')
    assert prompt.endswith('<|turn>model\n<|channel>thought\n<channel|>')
