# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Live MiniMax-H3 deployment health: the knobs and budgets that made or broke the quad deployments.

Tier 1 (~15 min with deployment control).  What each test guards, and the incident behind it:

* warmup envs -- ``TT_METAL_SHM_TRACKING_DISABLED=1`` + ``TT_METAL_LOGS_PATH`` cut the first request
  in a fresh process from 173 s to 42 s (2026-09-05); without them the k8s pods "warm up slowly";
* time-to-ready / first-request / warm-request budgets -- with cold-JIT detection, because a rebuild
  makes the first request 300+ s for reasons that have nothing to do with the deployment;
* a failed job must not poison the process -- the worker loop catches per-request errors, the next
  request must complete;
* (tier 3) all six trace-bucket rungs bound and then replayed in one process -- the 150 MB trace
  region must hold every capture (upstream validates with 450 MB).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h3_live_common as live  # noqa: E402
from h3_live_common import _fresh_or_skip, _need_task, require_live  # noqa: E402
from h3_live_common import assets, deployment, report, served_task  # noqa: E402,F401  (fixtures)
from h3_live_sequences import Spec, failures, run_one, run_sequence  # noqa: E402

require_live()
pytestmark = [pytest.mark.live]

OUT = Path(os.environ.get("H3_LIVE_OUT_DIR") or os.path.join(tempfile.gettempdir(), f"h3-live-{os.environ.get('USER', 'x')}")) / "deploy"
ON_SERVER_HOST = live.Disk.enabled or os.environ.get("H3_LIVE_ON_SERVER_HOST") == "1"

BUDGET_READY_S = float(os.environ.get("H3_LIVE_BUDGET_READY_S", "150"))
BUDGET_FIRST_S = float(os.environ.get("H3_LIVE_BUDGET_FIRST_S", "120"))        # warm kernel cache
BUDGET_FIRST_COLD_S = float(os.environ.get("H3_LIVE_BUDGET_FIRST_COLD_S", "600"))  # BuildKernels seen
BUDGET_WARM_S = float(os.environ.get("H3_LIVE_BUDGET_WARM_S", "60"))
COLD_JIT_KERNELS = int(os.environ.get("H3_LIVE_COLD_JIT_KERNELS", "500"))   # a new-canvas bind on a warm cache compiles ~140; a cold cache ~1800

REQUIRED_WORKER_ENV = {"TT_METAL_SHM_TRACKING_DISABLED": "1", "TT_METAL_LOGS_PATH": None}  # None: any non-empty value


def _worker_pids() -> list[int]:
    p = subprocess.run(["pgrep", "-f", "tt_model_runners.video_runner"], capture_output=True, text=True)
    return [int(x) for x in p.stdout.split() if x.isdigit()]


def _environ(pid: int) -> dict:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    return dict(kv.split("=", 1) for kv in raw.decode(errors="replace").split("\0") if "=" in kv)


@pytest.mark.h3_tier1
class TestHygiene:
    def test_check_passes_on_all_ranks(self, report):
        """`h3ctl.sh check`: identical tt-metal binary and tt-media-server tree on every rank, cache overlay
        mounted, weights readable.  MPI needs identical trees; a drifted rank fails in the middle of a run."""
        ctl = os.environ.get("H3_LIVE_CTL")
        if not ctl or not os.path.exists(ctl):
            pytest.skip("needs H3_LIVE_CTL (run_live_test.sh sets it from H3_DEPLOY_DIR/h3ctl.sh)")
        rc, out = live.sh(f"bash {ctl} check", timeout=300)
        report.add(combo="h3ctl check", task="-", request="check", status="passed" if rc == 0 else "failed", wall_s=0.0, job_id=None,
                   error=None if rc == 0 else out[-400:])
        assert rc == 0 and "CHECK PASSED" in out, f"h3ctl.sh check failed:\n{out[-1200:]}"


@pytest.mark.h3_tier1
class TestApiContract:
    @pytest.mark.xfail(strict=True, reason="the SP API process accepts out-of-policy duration_seconds (202) and the "
                       "request fails later as a job; the contract is a 422 at admission naming the served set "
                       "(see tests/test_minimax_h3_admission_gaps.py)")
    def test_out_of_policy_duration_is_refused_at_admission(self, served_task, deployment):
        _need_task("t2va", served_task, deployment)
        code, resp = live.http("POST", live.ENDPOINT["t2va"], {"prompt": live.PROMPT, "seed": 7, "duration_seconds": 3}, timeout=120)
        if code in (200, 202) and isinstance(resp, dict) and resp.get("id"):
            live._wait_terminal(resp["id"], time.time(), deployment)   # let the worker refuse it, keep the queue clean
            live.http("DELETE", f"/v1/videos/generations/{resp['id']}", timeout=60)
        assert code == 422, f"duration_seconds=3 -> {code} (expected 422 at admission): {str(resp)[:200]}"


@pytest.mark.h3_tier1
class TestDeploymentKnobs:
    def test_worker_env_has_the_warmup_knobs(self, served_task, deployment):
        """Every device worker on this host runs with the two envs that fixed the slow warmup."""
        if not ON_SERVER_HOST:
            pytest.skip("reads /proc/<pid>/environ: run on the server host (H3_LIVE_VIDEO_DIR or H3_LIVE_ON_SERVER_HOST=1)")
        _need_task("t2va", served_task, deployment)
        pids = _worker_pids()
        assert pids, "no tt_model_runners.video_runner process on this host"
        for pid in pids:
            env = _environ(pid)
            if not env:
                continue  # a process that exited between pgrep and read
            for key, want in REQUIRED_WORKER_ENV.items():
                assert key in env and env[key], f"worker {pid} lacks {key} (first request ~4x slower without it)"
                if want is not None:
                    assert env[key] == want, f"worker {pid}: {key}={env[key]!r}, expected {want!r}"


@pytest.mark.h3_tier1
class TestBudgets:
    def test_time_to_ready(self, served_task, deployment, report):
        """Fresh workers + API to model_ready within the budget (measured 30-60 s on quad1)."""
        if not deployment.controllable:
            pytest.skip("needs deployment control (H3_LIVE_START_CMD) to time a fresh start")
        if deployment.poisoned:
            deployment.fresh("t2va")   # absorb the session's first chip reset (~110 s); it is not startup time
        t0 = time.time()
        deployment.fresh("t2va")
        ready_s = round(time.time() - t0, 1)
        report.add(combo="time-to-ready", task="t2va", request="start", status="ready", wall_s=ready_s, job_id=None)
        assert ready_s <= BUDGET_READY_S, f"model_ready took {ready_s}s (budget {BUDGET_READY_S}s)"

    def test_first_and_warm_request(self, assets, served_task, deployment, report):
        """First request in the process (compile + capture path) and a warm replay, both judged for content.
        A cold kernel cache (BuildKernels lines in the worker log) switches the first-request budget."""
        _fresh_or_skip("t2va", served_task, deployment)
        first = run_one(Spec("t2va", "16:9", 5, label="first request"), assets, deployment, OUT / "budgets")
        report.add(**first.row())
        assert first.ok, first.describe()
        cold = (first.compiled_kernels or 0) >= COLD_JIT_KERNELS
        budget = BUDGET_FIRST_COLD_S if cold else BUDGET_FIRST_S
        assert first.wall_s <= budget, (
            f"first request took {first.wall_s}s (budget {budget}s, {'cold kernel cache: ' + str(first.compiled_kernels) + ' kernels compiled' if cold else 'warm kernel cache'}). "
            "173 s -> 42 s is the TT_METAL_SHM_TRACKING_DISABLED/TT_METAL_LOGS_PATH effect; check test_worker_env_has_the_warmup_knobs"
        )
        warm = run_one(Spec("t2va", "16:9", 5, label="warm request"), assets, deployment, OUT / "budgets")
        report.add(**warm.row())
        assert warm.ok, warm.describe()
        assert warm.wall_s <= BUDGET_WARM_S, f"warm request took {warm.wall_s}s (budget {BUDGET_WARM_S}s; 20-30 s on quad1)"


@pytest.mark.h3_tier1
class TestRecovery:
    def test_failed_job_does_not_poison_the_process(self, assets, served_task, deployment, report):
        """A request the worker refuses (duration outside 4..15 s -- the API admits it, see
        test_minimax_h3_admission_gaps.py) ends as a failed job; the next valid request must complete."""
        _need_task("t2va", served_task, deployment)
        t0 = time.time()
        code, resp = live.http("POST", live.ENDPOINT["t2va"], {"prompt": live.PROMPT, "seed": 7, "duration_seconds": 3}, timeout=120)
        if code == 422:
            pytest.skip("the API now rejects duration_seconds=3 at admission (good) -- pick another worker-side failure")
        assert code in (200, 202) and isinstance(resp, dict) and resp.get("id"), f"submit -> {code} {resp}"
        status, job, wall = live._wait_terminal(resp["id"], t0, deployment)
        err = str(job.get("error")) if isinstance(job, dict) else ""
        report.add(combo="worker-refused 3 s", task="t2va", request="failing", status=status, wall_s=wall, job_id=resp["id"], error=err[:300])
        assert status == "failed", f"3 s request ended {status} (expected the worker's policy refusal): {err[:200]}"
        assert "duration_seconds" in err, f"unexpected failure text: {err[:300]}"
        live.http("DELETE", f"/v1/videos/generations/{resp['id']}", timeout=60)
        after = run_one(Spec("t2va", "16:9", 5, label="after a failed job"), assets, deployment, OUT / "recovery")
        report.add(**after.row())
        assert after.ok, f"the process did not recover from a failed job: {after.describe()}"


@pytest.mark.h3_tier3
class TestTraceResidency:
    def test_all_rungs_bound_then_replayed(self, assets, served_task, deployment, report):
        """Bind all six rungs (smallest canvas/duration that lands on each), then serve each again so
        six captures are resident at once.  Guards the trace region (150 MB today, 450 MB upstream):
        an overflow fails the job in end_trace_capture and leaks its budget.  Status-only: the audio
        replay bug (xfail elsewhere) is expected to show in the second pass."""
        _fresh_or_skip("t2va", served_task, deployment)
        ladder = [Spec("t2va", "1:1", 5), Spec("t2va", "4:3", 5), Spec("t2va", "16:9", 5),
                  Spec("t2va", "16:9", 6), Spec("t2va", "16:9", 9), Spec("t2va", "16:9", 12)]
        results = run_sequence(ladder + ladder, assets, deployment, report, OUT / "trace-residency")
        not_done = [r.describe() for r in results if r.status != "completed"]
        assert not not_done, "requests did not complete with 6 rungs resident:\n  " + "\n  ".join(not_done)
        rungs = sorted({r.rung for r in results if r.rung})
        if rungs:
            assert rungs == [22528, 31744, 44032, 61440, 86016, 118784], f"rungs walked: {rungs}"
        content_bad = failures(results)
        if content_bad:
            print(f"[trace-residency] {len(content_bad)} outputs with corrupted content (known audio replay bug):\n  " + "\n  ".join(content_bad))
