"""Tests for temporary fixture download authorization."""

from minimax_mock.download_signer import DownloadSigner


def test_download_signatures_are_task_specific_and_expire():
    now = [1_700_000_000.0]
    signer = DownloadSigner(
        "test-signing-secret",
        ttl_seconds=60,
        clock=lambda: now[0],
    )

    authorization = signer.issue("task-1")

    assert authorization.expires == 1_700_000_060
    assert signer.verify("task-1", authorization.expires, authorization.signature)
    assert not signer.verify("task-2", authorization.expires, authorization.signature)
    assert not signer.verify("task-1", authorization.expires, "invalid")

    now[0] = authorization.expires + 1
    assert not signer.verify("task-1", authorization.expires, authorization.signature)
