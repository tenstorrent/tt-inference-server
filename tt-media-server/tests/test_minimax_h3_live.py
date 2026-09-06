# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Live hardware tests: MiniMax-H3 FL2VA and Ref2VA generation + DELETE on a running deployment.

These tests talk to a real tt-media-server over HTTP and occupy the mesh for
about an hour. They are **opt-in**: the module is skipped unless ``H3_LIVE_URL``
is set, so the default unit-test run is unaffected. The unit tests for the same
code paths live in ``test_job_manager.py`` and ``test_video_api.py``.

    H3_LIVE_URL=http://localhost:8000 pytest tests/test_minimax_h3_live.py -s -v

What one combination does (``TestFl2va`` keyframe layouts, ``TestRef2va``
reference mixes):

1. optionally start fresh workers for the task (see *deployment control*)
2. request 1 pays the kernel compile; ``H3_LIVE_REPEATS`` more requests are
   the warm measurements -- every request records API wall time and the mp4
   is downloaded and probed
3. DELETE contract on every job: 409 while live, 200 when terminal, then GET /
   download / DELETE answer 404, the job is gone from ``/v1/videos/jobs``; when
   the tests run on the server host the result file is gone from disk and the
   download left no remuxed copy behind in the temp dir

A MiniMax-H3 deployment serves one task. Without deployment control the tests
detect the served task (an empty body gets a "This deployment ..." 422 from the
endpoints it refuses and a field-validation 422 from the one it serves) and
skip the other class. With deployment control they switch tasks themselves.

Environment::

    H3_LIVE_URL             base URL of the server (required)
    H3_LIVE_API_KEY         bearer token (default: your-secret-key)
    H3_LIVE_REPEATS         warm generations after the compile request (default 3)
    H3_LIVE_POLL_TIMEOUT_S  per-job completion budget (default 2700)
    H3_LIVE_TASK            fl2va | ref2va -- what the server serves; probed when unset
    H3_LIVE_ASSETS_DIR      dir with img_512.png, vid_5s.mp4, aud_4s.mp3 and (optionally)
                            key_first.jpg, key_last.jpg; synthesised with ffmpeg when unset
    H3_LIVE_VIDEO_DIR       the server's TT_VIDEO_OUTPUT_DIR, when the tests run on the
                            server host -> enables the on-disk checks
    H3_LIVE_TMP_DIR         the server's temp dir for /download remux copies (default /tmp)
    H3_LIVE_REPORT          write the per-request timing table here as JSON

Deployment control (all optional, ``{task}`` is substituted)::

    H3_LIVE_START_CMD       start fresh workers + API for a task, e.g.
                            "bash /home/zni/h3-deploy/h3ctl.sh start {task}"
    H3_LIVE_WAIT_CMD        block until the deployment is ready, e.g.
                            "bash /home/zni/h3-deploy/h3ctl.sh wait-ready 1800"
    H3_LIVE_RESET_CMD       reset the chips on every host, e.g.
                            "bash /home/zni/h3-deploy/h3ctl.sh reset"
    H3_LIVE_RESET_SETTLE_S  seconds to let the inter-host links retrain after a reset (45)
    H3_LIVE_START_RETRIES   reset + start cycles after a failed start (2)
    H3_LIVE_RESET_FIRST     1 (default): reset the chips before this session's first start too.
                            Twice on the quad a run killed mid-request left the fabric in a state
                            where the next process hung 300 s into loading the vision tower
                            ("device timeout in fetch queue wait" at blocks.16.mlp.linear_fc2);
                            a reset first costs ~110 s once. 0 skips it.

When a start command is configured every combination begins with fresh workers
(a failed job does not release device memory, so anything measured after a
failure in the same process is noise -- see tt-inference-server#5044). The
chips are reset, not merely restarted, whenever the hardware may be wedged:

* a fresh start does not come up ready -> reset, let the links retrain, start
  again (``H3_LIVE_START_RETRIES`` cycles), fail the run if that does not help;
* a job failed or timed out (device memory is poisoned / a rank may hang) ->
  the next start is preceded by a reset;
* a job is still not terminal after the poll budget -> reset right away;
* at session end, if the last combination left the deployment poisoned, reset
  and restart it so the cluster is not left wedged for the next user.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests/ holds h3_live_common
from h3_live_common import (  # noqa: E402
    Disk,
    ENDPOINT,
    FL2VA_COMBOS,
    REF2VA_COMBOS,
    REF2VA_LIMIT_COMBOS,
    REF2VA_OVER_LIMIT,
    _cancel_then_delete,
    _delete_contract,
    _fresh_or_skip,
    _generate_repeatedly,
    _need_task,
    _routing,
    _wait_terminal,
    http,
    require_live,
)
from h3_live_common import assets, deployment, report, served_task  # noqa: E402,F401  (fixtures)

require_live()
pytestmark = pytest.mark.live




class TestFl2va:
    """Keyframe layouts on a MiniMax-H3 FL2VA deployment (MODEL_RUNNER=tt-minimax-h3-fl2va)."""

    @pytest.mark.h3_tier1
    def test_routing_and_delete_negatives(self, served_task, deployment):
        _need_task("fl2va", served_task, deployment)
        _routing("fl2va")

    @pytest.mark.parametrize("combo", FL2VA_COMBOS, ids=lambda c: c.name)
    def test_repeated_generation_and_delete(self, combo, assets, served_task, deployment, report, tmp_path):
        _fresh_or_skip("fl2va", served_task, deployment)
        _generate_repeatedly(combo, assets, deployment, report, tmp_path)

    def test_cancel_then_delete(self, assets, served_task, deployment):
        _need_task("fl2va", served_task, deployment)
        _cancel_then_delete("fl2va", FL2VA_COMBOS[0].body(assets), deployment)


# --------------------------------------------------------------------------- Ref2VA


class TestRef2va:
    """Reference mixes on a MiniMax-H3 Ref2VA deployment (MODEL_RUNNER=tt-minimax-h3-ref2va)."""

    @pytest.mark.h3_tier1
    def test_routing_and_delete_negatives(self, served_task, deployment):
        _need_task("ref2va", served_task, deployment)
        _routing("ref2va")

    @pytest.mark.parametrize("combo", REF2VA_COMBOS, ids=lambda c: getattr(c, "name", None))
    def test_repeated_generation_and_delete(self, combo, assets, served_task, deployment, report, tmp_path):
        _fresh_or_skip("ref2va", served_task, deployment)
        _generate_repeatedly(combo, assets, deployment, report, tmp_path)

    def test_cancel_then_delete(self, assets, served_task, deployment):
        _need_task("ref2va", served_task, deployment)
        _cancel_then_delete("ref2va", REF2VA_COMBOS[0].body(assets), deployment)

    # -- known limits, last: they poison the worker process ------------------------------

    @pytest.mark.parametrize("combo", REF2VA_LIMIT_COMBOS)
    def test_known_limit_mix_repeats(self, combo, assets, served_task, deployment, report, tmp_path):
        _fresh_or_skip("ref2va", served_task, deployment)
        _generate_repeatedly(combo, assets, deployment, report, tmp_path)

    def test_failed_job_is_deletable(self, assets, served_task, deployment, report):
        """Over the reference limit the job fails on device; a failed job is terminal and deletable."""
        _fresh_or_skip("ref2va", served_task, deployment)
        combo = REF2VA_OVER_LIMIT
        videos_before = Disk.videos()
        t0 = time.time()
        code, resp = http("POST", ENDPOINT["ref2va"], combo.body(assets), timeout=300)
        assert code in (200, 202) and isinstance(resp, dict), f"submit -> {code} {resp}"
        job_id = resp["id"]
        status, job, wall = _wait_terminal(job_id, t0, deployment)
        report.add(combo=combo.name, task="ref2va", request="compile", status=status, wall_s=wall, job_id=job_id,
                   error=str(job.get("error"))[:300] if isinstance(job, dict) else None)
        assert status in ("completed", "failed"), f"over-limit job ended {status}"
        if status == "failed":
            deployment.mark_poisoned(f"{combo.name} failed as expected")
            code, _ = http("GET", f"/v1/videos/generations/{job_id}/download", timeout=60)
            assert code == 404, f"download of a failed job -> {code}"
        _delete_contract(job_id)
        if Disk.enabled:
            assert Disk.videos() == videos_before
