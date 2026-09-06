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
from h3_live_sequences import Spec, failures, run_one, run_sequence, worker_log_alarms  # noqa: E402

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


@pytest.mark.h3_tier1
class TestLivenessUnderLoad:
    def test_liveness_stays_ready_during_a_generation(self, assets, served_task, deployment, report):
        """/tt-liveness must keep saying alive + model_ready while a 12 s generation runs (the k8s
        deployments died because a probe flipped during a 52 s job); queue_size is recorded per poll."""
        _need_task("t2va", served_task, deployment)
        spec = Spec("t2va", "16:9", 12, label="12 s under liveness polling")
        t0 = time.time()
        code, resp = live.http("POST", live.ENDPOINT["t2va"], spec.body(assets), timeout=120)
        assert code in (200, 202) and isinstance(resp, dict) and resp.get("id"), f"submit -> {code} {resp}"
        job_id = resp["id"]
        polls, bad = [], []
        while True:
            lc, lv = live.http("GET", "/tt-liveness", timeout=15)
            ok = lc == 200 and isinstance(lv, dict) and lv.get("status") == "alive" and lv.get("model_ready") is True
            polls.append({"t": round(time.time() - t0, 1), "code": lc, "ready": ok, "queue": lv.get("queue_size") if isinstance(lv, dict) else None})
            if not ok:
                bad.append(polls[-1])
            _, job = live.http("GET", f"/v1/videos/generations/{job_id}", timeout=30)
            if isinstance(job, dict) and job.get("status") in ("completed", "failed", "cancelled"):
                break
            if time.time() - t0 > 900:
                deployment.mark_poisoned("liveness test job did not finish", reset_now=True)
                pytest.fail("job did not reach a terminal state in 900 s")
            time.sleep(4)
        status = job.get("status")
        report.add(combo="liveness under load", task="t2va", request="12 s", status=status, wall_s=round(time.time() - t0, 1), job_id=job_id,
                   liveness_polls=len(polls), liveness_bad=len(bad), queue_sizes=[p["queue"] for p in polls])
        live.http("DELETE", f"/v1/videos/generations/{job_id}", timeout=60)
        assert status == "completed", f"job ended {status}: {str(job.get('error'))[:200]}"
        assert not bad, f"/tt-liveness degraded during the generation: {bad[:3]} (of {len(polls)} polls)"


@pytest.mark.h3_tier1
class TestApiRestart:
    def test_api_restart_keeps_serving(self, assets, served_task, deployment, report):
        """Restart only the uvicorn frontend while the workers keep running: the SHM rings are
        create-or-attach on both sides, so the next request must complete on the same worker process."""
        ctl = os.environ.get("H3_LIVE_CTL")
        if not ctl or not os.path.exists(ctl) or not deployment.controllable:
            pytest.skip("needs H3_LIVE_CTL with a restart-api subcommand")
        _need_task("t2va", served_task, deployment)
        t0 = time.time()
        rc, out = live.sh(f"bash {ctl} restart-api", timeout=120)
        assert rc == 0, f"restart-api failed: {out[-300:]}"
        deadline = time.time() + 180
        while time.time() < deadline:
            code, lv = live.http("GET", "/tt-liveness", timeout=15)
            if code == 200 and isinstance(lv, dict) and lv.get("model_ready") is True:
                break
            time.sleep(5)
        else:
            pytest.fail("API did not report model_ready within 180 s of restart-api")
        report.add(combo="api restart", task="t2va", request="restart", status="ready", wall_s=round(time.time() - t0, 1), job_id=None)
        res = run_one(Spec("t2va", "16:9", 5, label="after api restart"), assets, deployment, OUT / "api-restart")
        report.add(**res.row())
        assert res.ok, res.describe()


@pytest.mark.h3_tier3
class TestChaos:
    @pytest.mark.xfail(strict=True, reason="a dead rank is bounded only by TT_METAL_OPERATION_TIMEOUT (300 s per op) and the "
                       "5000 s request timeout: logs show 25-83 min to fail a job while /tt-liveness keeps saying model_ready")
    def test_killed_rank_fails_the_job_within_a_bound(self, assets, served_task, deployment, report):
        """Opt-in (H3_LIVE_CHAOS=1): kill the device worker on the second rank mid-generation.  The job
        must reach a terminal state within TT_METAL_OPERATION_TIMEOUT (300 s) plus margin instead of
        hanging, and the deployment must recover with a reset + fresh start."""
        if os.environ.get("H3_LIVE_CHAOS") != "1":
            pytest.skip("destructive: set H3_LIVE_CHAOS=1")
        hosts = (os.environ.get("H3_LIVE_HOSTS") or "").split()
        if len(hosts) < 2 or not deployment.controllable:
            pytest.skip("needs H3_LIVE_HOSTS (rank hosts) and deployment control")
        _fresh_or_skip("t2va", served_task, deployment)
        t0 = time.time()
        code, resp = live.http("POST", live.ENDPOINT["t2va"], Spec("t2va", "16:9", 15).body(assets), timeout=120)
        assert code in (200, 202) and resp.get("id"), f"submit -> {code} {resp}"
        job_id = resp["id"]
        time.sleep(15)
        victim = hosts[1]
        rc, out = live.sh(f"ssh -o BatchMode=yes -o ConnectTimeout=10 {victim} pkill -f [v]ideo_runner", timeout=60)
        print(f"  killed video_runner on {victim}: rc={rc}", flush=True)
        bound = float(os.environ.get("H3_LIVE_CHAOS_BOUND_S", "480"))
        status = None
        while time.time() - t0 < bound:
            _, job = live.http("GET", f"/v1/videos/generations/{job_id}", timeout=30)
            status = job.get("status") if isinstance(job, dict) else None
            if status in ("completed", "failed", "cancelled"):
                break
            time.sleep(10)
        wall = round(time.time() - t0, 1)
        _, lv = live.http("GET", "/tt-liveness", timeout=15)
        report.add(combo="chaos: rank 1 killed", task="t2va", request="15 s", status=status or "still running", wall_s=wall, job_id=job_id,
                   liveness=str(lv)[:200])
        deployment.mark_poisoned("chaos test killed a rank", reset_now=False)
        assert status == "failed", f"job is {status} after {wall}s with a dead rank (bound {bound}s): the API must fail it, not hang"


@pytest.mark.h3_tier1
class TestServedContract:
    """What the deployment actually serves versus what the API documents."""

    @pytest.mark.xfail(strict=True, reason="served requests run 20 denoise steps (shared DEFAULT_VIDEO_INFERENCE_STEPS) while "
                       "MINIMAX_H3_NUM_INFERENCE_STEPS=50 is the policy and the warmup schedule; the worker log says '20 steps'")
    def test_requests_run_at_the_policy_step_count(self, assets, served_task, deployment, report):
        _need_task("t2va", served_task, deployment)
        res = run_one(Spec("t2va", "16:9", 5, label="step-count probe"), assets, deployment, OUT / "contract")
        report.add(**res.row())
        assert res.ok, res.describe()
        if res.steps is None:
            pytest.skip("worker log not readable: step count unknown")
        assert res.steps == 50, f"served {res.steps} steps"

    @pytest.mark.xfail(strict=True, reason="the deployment runs on the compiled-in default API key ('your-secret-key'); "
                       "anyone on the VPN can drive the 128-chip mesh")
    def test_api_key_is_not_the_default(self):
        assert live.API_KEY != "your-secret-key"
        code, _ = live.http("GET", "/v1/videos/jobs", timeout=30)
        assert code == 200

    @pytest.mark.xfail(strict=True, reason="/v1/models returns 'MiniMaxAI/MiniMax-H3' for t2va, fl2va and ref2va alike; a client "
                       "cannot discover which task the deployment serves (the frontend knows it only via MODEL)")
    def test_models_endpoint_names_the_served_task(self, served_task, deployment):
        """Checked on an fl2va deployment: the id must say FL2VA (on t2va the shared id happens to look right)."""
        _need_task("fl2va", served_task, deployment)
        code, resp = live.http("GET", "/v1/models", timeout=30)
        assert code == 200 and isinstance(resp, dict), f"/v1/models -> {code}"
        ids = [m.get("id", "") for m in resp.get("data", [])]
        assert any("fl2va" in i.lower() for i in ids), f"{ids} do not name the served task fl2va"

    @pytest.mark.xfail(strict=True, reason="/tt-liveness reports device 'n150' on a 4x32 Blackhole quad (settings default; run_api.sh "
                       "sets no DEVICE) -- the field set is unpinned")
    def test_liveness_fields_describe_this_deployment(self):
        code, lv = live.http("GET", "/tt-liveness", timeout=30)
        assert code == 200 and isinstance(lv, dict)
        for key in ("status", "model_ready", "queue_size", "max_queue_size", "device_mesh_shape", "device", "runner_in_use"):
            assert key in lv, f"/tt-liveness lacks {key}: {sorted(lv)}"
        assert lv["device_mesh_shape"] == [4, 32]
        assert lv["device"] != "n150", f"device={lv['device']!r} on a Blackhole quad"

    @pytest.mark.xfail(strict=True, reason="while the workers warm up the API answers 405 'Model is not ready' to /tt-liveness and "
                       "to POST /generations (scheduler.py raises HTTPException(405)); the contract is 503 with model_ready:false")
    def test_not_ready_is_a_503(self, assets, served_task, deployment, report):
        if not deployment.controllable or not live.START_CMD:
            pytest.skip("needs deployment control to observe the warm-up window")
        rc, out = live.sh(live.START_CMD.format(task="t2va"), timeout=300)
        assert rc == 0, out[-300:]
        deployment.task, deployment.poisoned = "t2va", False
        codes = []
        deadline = time.time() + 240
        while time.time() < deadline:
            lc, lv = live.http("GET", "/tt-liveness", timeout=10)
            if lc == 200 and isinstance(lv, dict) and lv.get("model_ready") is True:
                break
            pc, presp = live.http("POST", live.ENDPOINT["t2va"], Spec("t2va", "16:9", 5).body(assets), timeout=30)
            codes.append((round(time.time() + 240 - deadline), lc, pc))
            if pc in (200, 202) and isinstance(presp, dict) and presp.get("id"):
                live.http("DELETE", f"/v1/videos/generations/{presp['id']}", timeout=30)
            time.sleep(3)
        if live.WAIT_CMD:
            live.sh(live.WAIT_CMD.format(task="t2va"), timeout=1800)
        report.add(combo="not-ready window", task="t2va", request="probe", status="observed", wall_s=0.0, job_id=None, codes=codes[:20])
        if not codes:
            pytest.skip("workers were ready before the first probe; warm-up window not observed")
        assert all(lc == 503 for _, lc, _ in codes), f"/tt-liveness during warm-up -> {sorted({lc for _, lc, _ in codes})}"
        assert all(pc == 503 for _, _, pc in codes), f"POST during warm-up -> {sorted({pc for _, _, pc in codes})}"


# keep this class LAST: it reads the worker log of the process the module has been exercising
@pytest.mark.h3_tier1
class TestWorkerLog:
    def test_no_unexpected_alarms_in_the_current_worker_log(self):
        """critical / TT_FATAL / Traceback / TIMEOUT / Permission denied lines other than the known-benign
        matmul-config fallbacks must not appear in a healthy run (a clean t2va run on 2026-09-06 had none)."""
        if not ON_SERVER_HOST:
            pytest.skip("reads the worker log on the server host")
        alarms = worker_log_alarms()
        assert not alarms, f"{len(alarms)} alarming worker-log lines:\n  " + "\n  ".join(alarms[:12])
