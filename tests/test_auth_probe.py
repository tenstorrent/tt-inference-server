"""Tests for utils/auth_probe.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.auth_probe import (  # noqa: E402
    ProbeResult,
    _fingerprint,
    _resolve_probe_url,
    assert_probe_ok,
    probe_bearer,
)


# ---------------------------------------------------------------------------
# URL normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base,chat,expected",
    [
        ("http://127.0.0.1:8000/v1", False, "http://127.0.0.1:8000/v1/completions"),
        (
            "http://127.0.0.1:8000/v1",
            True,
            "http://127.0.0.1:8000/v1/chat/completions",
        ),
        (
            "http://127.0.0.1:8000/v1/completions",
            False,
            "http://127.0.0.1:8000/v1/completions",
        ),
        (
            "http://127.0.0.1:8000/v1/chat/completions",
            True,
            "http://127.0.0.1:8000/v1/chat/completions",
        ),
        ("http://x:9000", False, "http://x:9000/completions"),
    ],
)
def test_resolve_probe_url_normalises_shapes(base, chat, expected):
    assert _resolve_probe_url(base, chat=chat) == expected


# ---------------------------------------------------------------------------
# Fingerprint hygiene
# ---------------------------------------------------------------------------


def test_fingerprint_is_short_hex_for_non_empty():
    fp = _fingerprint("a-bearer-token")
    assert len(fp) == 8
    assert all(c in "0123456789abcdef" for c in fp)


def test_fingerprint_is_literal_none_for_empty_or_missing():
    assert _fingerprint(None) == "none"
    assert _fingerprint("") == "none"


# ---------------------------------------------------------------------------
# probe_bearer: happy path
# ---------------------------------------------------------------------------


def _mock_response(status_code: int = 200, body=None, text: str = ""):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text or ""
    if body is not None:
        resp.json.return_value = body
    else:
        resp.json.side_effect = ValueError("not json")
    return resp


def test_probe_bearer_ok_on_200_with_choices():
    with patch("utils.auth_probe.requests.post") as post:
        post.return_value = _mock_response(
            200,
            body={"choices": [{"text": "."}]},
            text='{"choices": [{"text": "."}]}',
        )
        result = probe_bearer("http://x:8000/v1", "bearer", model="m")
    assert result.ok
    assert result.status_code == 200
    assert result.parsed_choices == 1
    assert result.bearer_fp != "none"


# ---------------------------------------------------------------------------
# probe_bearer: failure modes (the whole point of this module)
# ---------------------------------------------------------------------------


def test_probe_bearer_refuses_to_send_when_bearer_empty():
    # No HTTP call should be made.
    with patch("utils.auth_probe.requests.post") as post:
        result = probe_bearer("http://x:8000/v1", "")
    assert not result.ok
    assert result.error and "bearer is empty" in result.error
    post.assert_not_called()


def test_probe_bearer_fails_on_401():
    with patch("utils.auth_probe.requests.post") as post:
        post.return_value = _mock_response(401, text='{"error": "unauthorized"}')
        result = probe_bearer("http://x:8000/v1", "bad-bearer")
    assert not result.ok
    assert result.status_code == 401
    assert "non-200" in (result.error or "")
    assert "unauthorized" in result.body_preview


def test_probe_bearer_fails_on_200_with_no_choices_key():
    with patch("utils.auth_probe.requests.post") as post:
        post.return_value = _mock_response(
            200, body={"error": "invalid model"}, text='{"error": "invalid model"}'
        )
        result = probe_bearer("http://x:8000/v1", "bearer", model="bogus-model")
    assert not result.ok
    assert result.parsed_choices == 0
    assert "no usable 'choices'" in (result.error or "")


def test_probe_bearer_fails_on_200_with_empty_choices_list():
    with patch("utils.auth_probe.requests.post") as post:
        post.return_value = _mock_response(
            200, body={"choices": []}, text='{"choices": []}'
        )
        result = probe_bearer("http://x:8000/v1", "bearer")
    assert not result.ok
    assert result.parsed_choices == 0


def test_probe_bearer_fails_on_200_with_unparseable_body():
    with patch("utils.auth_probe.requests.post") as post:
        post.return_value = _mock_response(
            200, body=None, text="<html>maintenance</html>"
        )
        result = probe_bearer("http://x:8000/v1", "bearer")
    assert not result.ok
    assert "not JSON" in (result.error or "")


def test_probe_bearer_fails_on_connection_refused():
    with patch("utils.auth_probe.requests.post") as post:
        post.side_effect = requests.ConnectionError("refused")
        result = probe_bearer("http://x:8000/v1", "bearer")
    assert not result.ok
    assert result.status_code is None
    assert "transport error" in (result.error or "")


# ---------------------------------------------------------------------------
# assert_probe_ok
# ---------------------------------------------------------------------------


def test_assert_probe_ok_silent_on_ok():
    assert (
        assert_probe_ok(
            ProbeResult(
                ok=True,
                status_code=200,
                body_preview="",
                parsed_choices=1,
                bearer_fp="abcdef12",
                url="http://x/v1/completions",
            )
        )
        is None
    )


def test_assert_probe_ok_raises_with_diagnostic_on_failure():
    bad = ProbeResult(
        ok=False,
        status_code=401,
        body_preview='{"error": "unauthorized"}',
        parsed_choices=0,
        bearer_fp="cafef00d",
        url="http://x:8000/v1/completions",
        error="non-200 status; likely auth or model mismatch",
    )
    with pytest.raises(RuntimeError) as excinfo:
        assert_probe_ok(bad)
    msg = str(excinfo.value)
    assert "Preflight auth probe failed" in msg
    assert "cafef00d" in msg
    assert "401" in msg
    assert "Hint:" in msg
