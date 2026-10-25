# SPDX-License-Identifier: Apache-2.0
import json
import pytest
from llm_module.attempt_timing import timed_attempt


@pytest.mark.parametrize('rc', [0, 1, 124])
def test_all_completed_outcomes_preserve_wall_time(tmp_path, rc):
    path = tmp_path / 'attempt.json'
    with timed_attempt(path, 'gpqa', 'lm_eval_harness') as record:
        assert json.loads(path.read_text())['finished_at'] is None
        record['rc'] = rc
    result = json.loads(path.read_text())
    assert result['rc'] == rc
    assert result['elapsed_seconds'] >= 0
    assert result['actual_hours'] == result['elapsed_seconds'] / 3600
    assert result['started_at'].endswith('+00:00')
    assert result['finished_at'] is not None


def test_exception_preserves_timing_without_secret_message(tmp_path):
    path = tmp_path / 'attempt.json'
    with pytest.raises(RuntimeError):
        with timed_attempt(path, 'swe', 'harbor_harness'):
            raise RuntimeError('credential-secret')
    result = json.loads(path.read_text())
    assert result['exception_type'] == 'RuntimeError'
    assert result['rc'] is None
    assert result['elapsed_seconds'] >= 0
    assert 'credential-secret' not in path.read_text()


def test_other_models_do_not_emit_receipts(tmp_path):
    path = tmp_path / 'attempt.json'
    with timed_attempt(path, 'other', 'harbor_harness', enabled=False) as record:
        record['rc'] = 0
    assert not path.exists()
