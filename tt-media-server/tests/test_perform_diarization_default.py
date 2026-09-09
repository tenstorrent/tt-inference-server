# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The server-side default for perform_diarization.

Covers all three entry points, because each resolved the flag differently before:

* the multipart route (``Form``), which used to collapse "omitted" and
  "explicitly false" into the same value via ``perform_diarization or False``;
* the JSON route, which builds ``AudioProcessingRequest`` straight from the body
  and so only ever sees the model's own default;
* ``audio_worker_function``, which runs in a CpuWorkloadHandler worker *process* --
  an in-process default cannot be assumed to survive the queue, so that one is
  exercised across a real process boundary.
"""

import importlib.machinery
import multiprocessing as mp
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _reload_with(monkeypatch, value):
    """Re-import the settings + domain modules with PERFORM_DIARIZATION set."""
    import importlib

    monkeypatch.setenv("PERFORM_DIARIZATION", value)
    import config.settings as settings_mod

    importlib.reload(settings_mod)
    import domain.audio_processing_request as req_mod

    importlib.reload(req_mod)
    return settings_mod.settings, req_mod.AudioProcessingRequest


class TestSettingsDefault:
    def test_defaults_to_true(self):
        from config.settings import get_settings

        assert get_settings().perform_diarization is True

    @pytest.mark.parametrize("env,expected", [("false", False), ("true", True)])
    def test_env_override(self, monkeypatch, env, expected):
        settings, _ = _reload_with(monkeypatch, env)
        assert settings.perform_diarization is expected


class TestJsonRouteDefault:
    """The JSON route does AudioProcessingRequest(**json_body) with no massaging,
    so the model default is the only thing standing between a JSON client and
    diarization being silently off."""

    def test_model_default_follows_settings(self):
        from domain.audio_processing_request import AudioProcessingRequest

        req = AudioProcessingRequest(file=b"x")
        assert req.perform_diarization is True

    def test_explicit_false_still_wins(self):
        from domain.audio_processing_request import AudioProcessingRequest

        req = AudioProcessingRequest(file=b"x", perform_diarization=False)
        assert req.perform_diarization is False

    def test_model_default_follows_env(self, monkeypatch):
        _, Request = _reload_with(monkeypatch, "false")
        assert Request(file=b"x").perform_diarization is False


def _worker_probe(queue, passed):
    """Runs in a child process: report what audio_worker_function resolved to.

    Patches the branch inputs rather than the whole function so the resolution
    under test is the real one.
    """
    try:
        from unittest.mock import MagicMock

        import model_services.audio_service as svc

        seen = {}

        prepared = MagicMock()
        prepared.audio_array = "arr"
        prepared.duration = 5.0
        prepared.source_sample_rate = 16000
        prepared.source_channels = 1
        mgr = MagicMock()
        mgr.to_audio_array.return_value = prepared

        def _record(audio_array, enable_diarization):
            seen["diarization"] = enable_diarization
            return ["seg"]

        mgr.apply_diarization_with_vad.side_effect = _record

        kwargs = {} if passed is None else {"perform_diarization": passed}
        svc.audio_worker_function(mgr, b"bytes", True, **kwargs)
        queue.put(("ok", seen.get("diarization")))
    except Exception as e:  # surface the failure instead of hanging the parent
        queue.put(("err", f"{type(e).__name__}: {e}"))


@pytest.mark.parametrize(
    "passed,expected",
    [
        (None, True),  # caller omitted it -> server-side default
        (False, False),  # explicit false survives the boundary
        (True, True),
    ],
)
def test_worker_resolves_across_process_boundary(passed, expected):
    # model_services.audio_service reaches torch transitively via
    # cpu_workload_handler, and torch lives in the separate audio venv (see the
    # header of tt-media-server/requirements.txt).
    #
    # PathFinder, not importorskip or importlib.util.find_spec: conftest injects
    # MagicMock entries for torch into sys.modules, so `import torch` succeeds
    # here even when the package is absent, and util.find_spec raises
    # "torch.__spec__ is not set" on the mock. The spawned child inherits none of
    # those mocks, so either would mislead. PathFinder searches sys.path and
    # ignores sys.modules, which is what the child will actually do.
    if importlib.machinery.PathFinder().find_spec("torch") is None:
        pytest.skip("audio_service needs a real torch (installed in the audio venv)")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker_probe, args=(q, passed))
    p.start()
    try:
        status, value = q.get(timeout=120)
    finally:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            pytest.fail("worker process hung resolving perform_diarization")
    assert status == "ok", value
    assert value is expected
