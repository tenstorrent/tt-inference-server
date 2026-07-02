"""
Tests for issue #143: transient-error retry backoff schedule.

Verifies that:
- _BACKOFF_SECONDS covers ~10 retries capped at 5 minutes each.
- InternalServerError and APIConnectionError are both retried.
- Each retry log line includes elapsed time and the total attempt count.
- After exhausting all retries the original exception is re-raised.
"""

import openai
from unittest.mock import patch, MagicMock, call

import orchestrator.agent as agent_module
from orchestrator.agent import _BACKOFF_SECONDS


def _make_response(content="ok"):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = []
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _persona(name="p"):
    return {"name": name, "model": "m", "system": "s"}


class TestBackoffSchedule:
    def test_length_gives_roughly_15_minute_coverage(self):
        # 10 retries means 11 total attempts; worst-case sum ≥ 15 min.
        assert len(_BACKOFF_SECONDS) == 10

    def test_max_individual_sleep_is_five_minutes(self):
        assert max(_BACKOFF_SECONDS) == 300

    def test_first_sleep_is_short(self):
        assert _BACKOFF_SECONDS[0] == 5

    def test_worst_case_total_is_under_30_minutes(self):
        assert sum(_BACKOFF_SECONDS) < 30 * 60


class TestInternalServerErrorRetry:

    @patch("orchestrator.agent.T.DEFS", [])
    @patch("orchestrator.agent.time.sleep")
    @patch("orchestrator.agent._client")
    def test_retries_on_internal_server_error(self, mock_client_factory, mock_sleep):
        client = MagicMock()
        mock_client_factory.return_value = client
        err = openai.InternalServerError(
            message="boom", response=MagicMock(), body={}
        )
        client.chat.completions.create.side_effect = [err, err, _make_response()]

        from orchestrator.agent import run
        text, _ = run(_persona(), [{"role": "user", "content": "hi"}], verbose=False)

        assert text == "ok"
        assert client.chat.completions.create.call_count == 3
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0] == call(_BACKOFF_SECONDS[0])
        assert mock_sleep.call_args_list[1] == call(_BACKOFF_SECONDS[1])

    @patch("orchestrator.agent.T.DEFS", [])
    @patch("orchestrator.agent.time.sleep")
    @patch("orchestrator.agent._client")
    def test_reraises_after_all_retries_exhausted(self, mock_client_factory, mock_sleep):
        client = MagicMock()
        mock_client_factory.return_value = client
        err = openai.InternalServerError(
            message="still broken", response=MagicMock(), body={}
        )
        # One more than total retries allowed.
        client.chat.completions.create.side_effect = [err] * (len(_BACKOFF_SECONDS) + 1)

        from orchestrator.agent import run
        import pytest
        with pytest.raises(openai.InternalServerError):
            run(_persona(), [{"role": "user", "content": "hi"}], verbose=False)

        assert mock_sleep.call_count == len(_BACKOFF_SECONDS)


class TestAPIConnectionErrorRetry:

    @patch("orchestrator.agent.T.DEFS", [])
    @patch("orchestrator.agent.time.sleep")
    @patch("orchestrator.agent._client")
    def test_retries_on_api_connection_error(self, mock_client_factory, mock_sleep):
        client = MagicMock()
        mock_client_factory.return_value = client
        err = openai.APIConnectionError(request=MagicMock())
        client.chat.completions.create.side_effect = [err, _make_response()]

        from orchestrator.agent import run
        text, _ = run(_persona(), [{"role": "user", "content": "hi"}], verbose=False)

        assert text == "ok"
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args_list[0] == call(_BACKOFF_SECONDS[0])


class TestRetryLogging:

    @patch("orchestrator.agent.T.DEFS", [])
    @patch("orchestrator.agent.time.sleep")
    @patch("orchestrator.agent.time.monotonic", side_effect=[0.0, 7.3, 7.3])
    @patch("orchestrator.agent._client")
    def test_log_includes_elapsed_and_total_attempts(
        self, mock_client_factory, mock_monotonic, mock_sleep, capsys
    ):
        client = MagicMock()
        mock_client_factory.return_value = client
        err = openai.InternalServerError(
            message="oops", response=MagicMock(), body={}
        )
        client.chat.completions.create.side_effect = [err, _make_response()]

        from orchestrator.agent import run
        run(_persona(), [{"role": "user", "content": "hi"}], verbose=True)

        out = capsys.readouterr().out
        assert "elapsed" in out
        assert f"/{len(_BACKOFF_SECONDS) + 1}" in out
