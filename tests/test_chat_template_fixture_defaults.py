from types import SimpleNamespace

import pytest

from test_fixtures import conftest as fixtures


def _request(value):
    options = {"--model-name": "org/model", "--chat-template-kwargs": value}
    return SimpleNamespace(config=SimpleNamespace(
        getoption=lambda key, **_: options[key]))


def test_defaults_are_explicit_and_request_settings_win(monkeypatch):
    sent = []
    monkeypatch.setattr(fixtures, "_get_bearer_token", lambda: None)
    monkeypatch.setattr(fixtures.requests, "post", lambda url, **kwargs: (
        sent.append(kwargs) or SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {})))
    client = fixtures.api_client.__wrapped__(
        "http://example/v1/chat/completions",
        _request('{"enable_thinking": false}'))
    payload = {"messages": [], "temperature": 0.5, "max_tokens": 32}
    client(payload)
    assert sent[-1]["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent[-1]["json"]["temperature"] == 0.5
    assert sent[-1]["timeout"] == 30
    assert "chat_template_kwargs" not in payload
    client({**payload, "chat_template_kwargs": {"enable_thinking": True}})
    assert sent[-1]["json"]["chat_template_kwargs"] == {"enable_thinking": True}
    client({"prompt": "raw completion"})
    assert "chat_template_kwargs" not in sent[-1]["json"]


def test_unconfigured_chat_requests_preserve_existing_payload(monkeypatch):
    sent = []
    monkeypatch.setattr(fixtures, "_get_bearer_token", lambda: None)
    monkeypatch.setattr(fixtures.requests, "post", lambda url, **kwargs: (
        sent.append(kwargs) or SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {})))
    client = fixtures.api_client.__wrapped__(
        "http://example/v1/chat/completions", _request("{}"))
    payload = {"model": "org/model", "messages": [], "temperature": 0.9,
               "max_tokens": 1024, "seed": 1234}
    client(payload, timeout=None)
    assert sent[-1]["json"] == payload
    assert "chat_template_kwargs" not in sent[-1]["json"]
    assert sent[-1]["timeout"] is None


@pytest.mark.parametrize("value", ["[]", "null", '"false"'])
def test_non_object_defaults_rejected(value):
    with pytest.raises(ValueError, match="JSON object"):
        fixtures._chat_template_defaults(_request(value))
