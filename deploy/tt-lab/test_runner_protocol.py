"""Adapter unit tests with recorded token streams, not a hardware validation."""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.update(MODEL_RUNNER='tt-lab-gpt-oss', MODEL_SERVICE='llm', IS_GALAXY='false', DEVICE_IDS='(0)', MODEL_WEIGHTS_PATH='/home/ttuser/models/gpt-oss-20b-tokenizer')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tt-media-server'))
import pytest
from transformers import AutoTokenizer
from domain.completion_request import CompletionRequest
from tt_model_runners.tt_lab_runner import TTLabRunner

@pytest.fixture
def runner():
    instance = TTLabRunner.__new__(TTLabRunner)
    instance.tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_WEIGHTS_PATH'])
    instance.timeout = 10
    with tempfile.TemporaryFile() as pipe:
        instance.process = SimpleNamespace(stdin=pipe, poll=lambda: None)
        yield instance
        instance.process = None

def feed(runner, text, finish=-1):
    tokens = runner.tokenizer.encode(text, add_special_tokens=False)
    stream = iter(tokens + [finish])
    runner._read_token = lambda deadline: next(stream)
    return len(tokens)

def test_harmony_final_and_actual_token_count(runner):
    count = feed(runner, '<|channel|>analysis<|message|>reasoning<|end|><|start|>assistant<|channel|>final<|message|>Four.')
    result = runner.run([CompletionRequest(prompt='<|start|>user<|message|>2+2?<|end|><|start|>assistant', max_tokens=100)])[0]['data']
    assert result.text == 'Four.'
    assert result.completion_tokens == count
    assert result.finish_reason == 'stop'

def test_stop_string_spanning_tokens_is_not_leaked(runner):
    feed(runner, 'Hello END ignored')
    result = runner.run([CompletionRequest(prompt='Hi', max_tokens=100, stop=['END'])])[0]['data']
    assert result.text == 'Hello '
    assert result.finish_reason == 'stop'

def test_length_finish(runner):
    feed(runner, 'hello', finish=-2)
    result = runner.run([CompletionRequest(prompt='Hi', max_tokens=1)])[0]['data']
    assert result.text == 'hello'
    assert result.finish_reason == 'length'

def test_sampling_rejected(runner):
    with pytest.raises(ValueError, match='greedy'):
        runner.run([CompletionRequest(prompt='Hi', temperature=0.7)])

def test_invalid_token_rejected_before_dispatch(runner):
    with pytest.raises(ValueError, match='Invalid prompt'):
        runner.run([CompletionRequest(prompt=[-1])])


def test_no_final_channel_does_not_expose_analysis(runner):
    count = feed(runner, '<|channel|>analysis<|message|>unfinished reasoning', finish=-2)
    result = runner.run([CompletionRequest(prompt='<|start|>assistant', max_tokens=30)])[0]['data']
    assert result.text == ''
    assert result.finish_reason == 'length'
    assert result.completion_tokens == count


def test_stopped_request_drains_protocol_for_next_request(runner):
    feed(runner, 'One STOP ignored')
    first = runner.run([CompletionRequest(prompt='Hi', max_tokens=30, stop='STOP')])[0]['data']
    feed(runner, 'Two')
    second = runner.run([CompletionRequest(prompt='Hi', max_tokens=30)])[0]['data']
    assert first.text == 'One '
    assert second.text == 'Two'

@pytest.mark.parametrize('options', [
    {'n': 2}, {'model': 'wrong'}, {'max_tokens': 0}, {'max_tokens': 4096},
    {'prompt': ''}, {'prompt': []}, {'prompt': [201088]},
    {'frequency_penalty': 1}, {'adapter': 'adapter'}, {'top_p': 0.9},
    {'repetition_penalty': 1.1}, {'ignore_eos': True}, {'logprobs': 5},
])
def test_invalid_requests_never_write_to_device(runner, options):
    with pytest.raises(ValueError):
        runner.run([CompletionRequest(**{'prompt': 'Hi', **options})])
    assert runner.process.stdin.tell() == 0
