#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Checks that need the real server (tt_media_server_cpp), not the stub.

Run against a server started with MODEL_RUNNER_TYPE=video_mock.
tests/video/mock/run_server_tests.sh starts the server once per --phase:

  main        TT_VIDEO_MOCK_DELAY_MS=3000, TT_VIDEO_MOCK_WRITE_FILE=1,
              MAX_QUEUE_SIZE=3 (--max-queue 3): 401, 409, cancel queued /
              running, sync mode returns an mp4, async download, 429.
  sync-error  default mock (result is mock://, not a file): sync answers 500.
  limits      MAX_JOBS=2: the third job gets 503 (job limit).
  warmup      TT_VIDEO_MOCK_WARMUP_MS=6000: POST during warmup gets 503.
  cancel-race TT_MEDIA_TEST_PRECLAIM_DELAY_MS=1500 + TT_VIDEO_MOCK_DELAY_MS=3000:
              a cancel that lands after the worker popped the task but before
              CLAIMED must still stop it (the shell script then checks the
              server log: "(after claim)" and no MockVideoRunner run).
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contract_cases as cc  # noqa: E402

GEN = cc.GEN
note = cc.note


def raw(base, method, path, body=None, token=None):
    """(status, headers, parsed json) so Retry-After can be checked."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("content-type", "application/json")
    if token:
        req.add_header("authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode()
            return resp.status, resp.headers, json.loads(text) if text.strip() else {}
    except urllib.error.HTTPError as exc:
        text = exc.read().decode()
        try:
            payload = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            payload = {"raw": text}
        return exc.status, exc.headers, payload


def job_status(base, token, job_id):
    status, _, job = raw(base, "GET", f"{GEN}/{job_id}", token=token)
    return job.get("status") if status == 200 else None


def wait_for(base, token, job_id, wanted, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job_status(base, token, job_id) in wanted:
            return True
        time.sleep(0.3)
    return False


def check_auth(base, token):
    for label, tok in (("no token", None), ("wrong token", "wrong-" + token)):
        status, _, payload = raw(base, "POST", GEN, {"prompt": "x"}, tok)
        note(status == 401, f"401 POST {label}", f"status={status}")
        status, _, _ = raw(base, "GET", "/v1/videos/jobs", token=tok)
        note(status == 401, f"401 GET jobs {label}", f"status={status}")
    status, _, _ = raw(base, "GET", "/v1/videos/jobs", token=token)
    note(status == 200, "200 with a valid token", f"status={status}")


def check_queue_cancel_delete(base, token):
    ids = []
    for i in range(3):
        status, _, job = raw(base, "POST", GEN, {"prompt": f"p{i}", "seed": i}, token)
        ids.append(job.get("id"))
        note(status == 202, f"submit {i + 1}", f"status={status} id={job.get('id')}")
    first, second, third = ids

    running = wait_for(base, token, first, ("in_progress",))
    note(
        running,
        "the first job is in_progress",
        f"status={job_status(base, token, first)}",
    )

    # 409: DELETE on a running and on a queued job
    for label, jid in (("in_progress", first), ("queued", second)):
        status, _, payload = raw(base, "DELETE", f"{GEN}/{jid}", token=token)
        note(
            status == 409,
            f"409 DELETE of a {label} job",
            f"status={status} message={cc.error_message(payload)!r}",
        )

    # cancel while queued: terminal at once, no error
    status, _, payload = raw(base, "POST", f"{GEN}/{second}/cancel", token=token)
    note(
        status == 200 and payload.get("status") == "cancelled",
        "cancel of a queued job",
        f"status={status} job_status={payload.get('status')!r}",
    )
    status, _, payload = raw(base, "POST", f"{GEN}/{third}/cancel", token=token)
    note(
        status == 200 and payload.get("status") == "cancelled",
        "cancel of a queued job (2)",
        f"status={status}",
    )
    # second cancel of the same job: already terminal -> 404
    status, _, _ = raw(base, "POST", f"{GEN}/{second}/cancel", token=token)
    note(status == 404, "second cancel is 404", f"status={status}")

    # cancel the running one: cancelling now, cancelled when the worker returns
    status, _, payload = raw(base, "POST", f"{GEN}/{first}/cancel", token=token)
    note(
        status == 200 and payload.get("status") == "cancelling",
        "cancel of a running job -> cancelling",
        f"status={status} job_status={payload.get('status')!r}",
    )
    done = wait_for(base, token, first, ("cancelled",), timeout=30)
    note(
        done,
        "the running job ends as cancelled",
        f"status={job_status(base, token, first)}",
    )

    # nothing of the three produced an error; cancelled queued jobs stay cancelled
    for label, jid in (("1", first), ("2", second), ("3", third)):
        status, _, job = raw(base, "GET", f"{GEN}/{jid}", token=token)
        note(
            status == 200 and job.get("status") == "cancelled" and "error" not in job,
            f"job {label} cancelled without an error",
            f"status={status} job_status={job.get('status')!r} error={job.get('error')!r}",
        )
        status, _, _ = raw(base, "DELETE", f"{GEN}/{jid}", token=token)
        note(status == 200, f"DELETE of cancelled job {label}", f"status={status}")


def fetch_raw(base, path, token, method="GET", body=None):
    """(status, headers, raw bytes) for non-JSON answers such as the mp4."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("content-type", "application/json")
    if token:
        req.add_header("authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.status, exc.headers, exc.read()


def check_sync_ok(base, token):
    _, _, before = fetch_raw(base, "/v1/videos/jobs", token)
    count_before = len(json.loads(before))
    for label, path, body in (
        ("?async=false", GEN + "?async=false", {"prompt": "sync one"}),
        ("JSON async:false", GEN, {"prompt": "sync two", "async": False}),
    ):
        started = time.time()
        status, headers, data = fetch_raw(base, path, token, "POST", body)
        elapsed = time.time() - started
        note(
            status == 200
            and headers.get("Content-Type", "").startswith("video/mp4")
            and data[4:8] == b"ftyp",
            f"sync {label}: 200 + mp4",
            f"status={status} type={headers.get('Content-Type')!r} bytes={len(data)} "
            f"ftyp={data[4:8]!r} duration={elapsed:.1f}s",
        )
        gen_time = headers.get("X-Generation-Time", "")
        try:
            ok_time = float(gen_time) >= 0
        except ValueError:
            ok_time = False
        note(ok_time, f"sync {label}: X-Generation-Time", f"header={gen_time!r}")
        disposition = headers.get("Content-Disposition", "")
        note(
            "attachment" in disposition and ".mp4" in disposition,
            f"sync {label}: Content-Disposition",
            f"header={disposition!r}",
        )
    _, _, after = fetch_raw(base, "/v1/videos/jobs", token)
    note(
        len(json.loads(after)) == count_before,
        "sync creates no job record",
        f"jobs before={count_before} after={len(json.loads(after))}",
    )
    # ?async=true wins over JSON async:false
    status, _, job = raw(
        base, "POST", GEN + "?async=true", {"prompt": "x", "async": False}, token
    )
    note(
        status == 202 and cc.UUID_RE.match(job.get("id", "")) is not None,
        "?async=true wins over JSON async:false",
        f"status={status}",
    )
    wait_for(
        base, token, job.get("id"), ("completed", "failed", "cancelled"), timeout=30
    )
    # async download of the finished file
    status, headers, data = fetch_raw(base, f"{GEN}/{job.get('id')}/download", token)
    note(
        status == 200 and data[4:8] == b"ftyp",
        "async download: 200 mp4",
        f"status={status} bytes={len(data)} type={headers.get('Content-Type')!r}",
    )
    if b"moov" in data:
        # a real mp4 (TT_VIDEO_MOCK_WRITE_FILE=ffmpeg): the worker remuxed it
        # once when the video finished, so moov comes before mdat
        note(
            data.find(b"moov") < data.find(b"mdat"),
            "download: faststart (moov before mdat), the worker did the remux",
            f"moov={data.find(b'moov')} mdat={data.find(b'mdat')}",
        )
    raw(base, "DELETE", f"{GEN}/{job.get('id')}", token=token)


def range_get(base, token, jid, rng):
    req = urllib.request.Request(f"{base}{GEN}/{jid}/download")
    req.add_header("authorization", f"Bearer {token}")
    if rng:
        req.add_header("Range", rng)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.status, exc.headers, exc.read()


def check_download(base, token):
    # unknown job vs job that is not finished yet
    status, _, payload = raw(
        base,
        "GET",
        f"{GEN}/{cc.__dict__.get('ZERO_UUID', '00000000-0000-4000-8000-000000000000')}/download",
        token=token,
    )
    note(
        status == 404 and cc.error_message(payload) == "Video job not found",
        "download: unknown job -> 404",
        f"status={status} message={cc.error_message(payload)!r}",
    )
    status, _, job = raw(base, "POST", GEN, {"prompt": "dl"}, token)
    jid = job.get("id")
    status, _, payload = raw(base, "GET", f"{GEN}/{jid}/download", token=token)
    message = cc.error_message(payload)
    note(
        status == 409
        and "not completed" in message
        and ("queued" in message or "in_progress" in message),
        "download: job not finished -> 409",
        f"status={status} message={message!r}",
    )
    note(
        wait_for(base, token, jid, ("completed",), timeout=30),
        "download: job finished",
        "",
    )

    status, headers, full = range_get(base, token, jid, None)
    size = len(full)
    note(
        status == 200
        and headers.get("Accept-Ranges") == "bytes"
        and headers.get("Content-Type", "").startswith("video/mp4")
        and "attachment" in headers.get("Content-Disposition", "")
        and int(headers.get("Content-Length", -1)) == size,
        "download: 200 + Accept-Ranges, Content-Type, Content-Disposition, Content-Length",
        f"status={status} headers={dict(headers)}",
    )

    status, headers, part = range_get(base, token, jid, "bytes=0-99")
    note(
        status == 206
        and part == full[:100]
        and headers.get("Content-Range") == f"bytes 0-99/{size}"
        and int(headers.get("Content-Length", -1)) == 100,
        "download: Range 0-99 -> 206 + Content-Range",
        f"status={status} range={headers.get('Content-Range')!r} bytes={len(part)}",
    )
    status, headers, part = range_get(base, token, jid, "bytes=-10")
    note(
        status == 206
        and part == full[-10:]
        and headers.get("Content-Range") == f"bytes {size - 10}-{size - 1}/{size}",
        "download: Range -10 (end of the file) -> 206",
        f"status={status} range={headers.get('Content-Range')!r}",
    )
    status, headers, part = range_get(base, token, jid, "bytes=20-")
    note(
        status == 206 and part == full[20:],
        "download: Range 20- (to the end) -> 206",
        f"status={status} bytes={len(part)}",
    )
    status, headers, part = range_get(base, token, jid, f"bytes={size + 5}-")
    note(
        status == 416 and headers.get("Content-Range") == f"bytes */{size}",
        "download: Range beyond the file -> 416 + Content-Range */size",
        f"status={status} range={headers.get('Content-Range')!r}",
    )
    status, headers, part = range_get(base, token, jid, "bytes=0-9,20-29")
    note(
        status == 200 and part == full,
        "download: several ranges are ignored -> 200 whole file",
        f"status={status} bytes={len(part)}",
    )
    # a failed/cancelled job has nothing to download either: 409 with its status
    raw(base, "DELETE", f"{GEN}/{jid}", token=token)


def check_sync_error(base, token):
    status, _, payload = raw(base, "POST", GEN + "?async=false", {"prompt": "x"}, token)
    message = cc.error_message(payload)
    note(
        status == 500 and "file not found" in message,
        "sync: the result is not a file -> 500",
        f"status={status} message={message!r}",
    )
    status, _, payload = raw(
        base,
        "POST",
        GEN + "?async=false",
        {"prompt": "x", "num_inference_steps": 3},
        token,
    )
    note(status == 422, "sync: invalid request -> 422", f"status={status}")


def check_job_limit(base, token):
    ids = []
    for i in range(2):
        status, _, job = raw(base, "POST", GEN, {"prompt": f"l{i}"}, token)
        ids.append(job.get("id"))
        note(status == 202, f"limit: job {i + 1} accepted", f"status={status}")
    for jid in ids:
        wait_for(base, token, jid, ("completed", "failed"), timeout=30)
    status, headers, payload = raw(base, "POST", GEN, {"prompt": "l2"}, token)
    note(
        status == 503 and headers.get("Retry-After") == "60",
        "503 when the job limit is reached (MAX_JOBS=2)",
        f"status={status} retry_after={headers.get('Retry-After')!r} "
        f"message={cc.error_message(payload)!r}",
    )
    status, _, _ = raw(base, "DELETE", f"{GEN}/{ids[0]}", token=token)
    note(status == 200, "limit: DELETE frees a slot", f"status={status}")
    status, _, job = raw(base, "POST", GEN, {"prompt": "l3"}, token)
    note(status == 202, "limit: a new job after DELETE", f"status={status}")


def check_warmup(base, token):
    deadline = time.time() + 30
    live = None
    while time.time() < deadline:
        try:
            # liveness answers 503 (with model_ready=false) during warmup
            status, _, live = raw(base, "GET", "/tt-liveness")
            if status in (200, 503):
                break
        except Exception:
            pass
        time.sleep(0.2)
    note(
        live is not None and live.get("model_ready") is False,
        "warmup: model_ready=false",
        f"status={status} liveness={live}",
    )
    status, headers, payload = raw(base, "POST", GEN, {"prompt": "x"}, token)
    note(
        status == 503 and headers.get("Retry-After") == "30",
        "503 during warmup",
        f"status={status} retry_after={headers.get('Retry-After')!r} "
        f"message={cc.error_message(payload)!r}",
    )
    ready = False
    for _ in range(120):
        status, _, live = raw(base, "GET", "/tt-liveness")
        if status == 200 and live.get("model_ready") is True:
            ready = True
            break
        time.sleep(0.5)
    note(ready, "warmup finishes (model_ready=true)", f"liveness={live}")
    status, _, job = raw(base, "POST", GEN, {"prompt": "x"}, token)
    note(status == 202, "POST after warmup is 202", f"status={status}")


def check_cancel_race(base, token):
    # The worker pops the task at once and then waits 1.5 s before sending
    # CLAIMED (test hook), so for that time the job is still "queued".
    status, _, job = raw(base, "POST", GEN, {"prompt": "race"}, token)
    jid = job.get("id")
    time.sleep(0.5)
    before = job_status(base, token, jid)
    note(
        before == "queued",
        "race: the job is still queued while the worker waits",
        f"status={before!r}",
    )
    status, _, payload = raw(base, "POST", f"{GEN}/{jid}/cancel", token=token)
    note(
        status == 200 and payload.get("status") == "cancelled",
        "race: cancel -> cancelled",
        f"status={status} job_status={payload.get('status')!r}",
    )
    note(
        wait_for(base, token, jid, ("cancelled",), timeout=15),
        "race: the job stays cancelled",
        f"status={job_status(base, token, jid)!r}",
    )
    time.sleep(2.5)  # long enough for the worker to wake up and (not) start
    status, _, final = raw(base, "GET", f"{GEN}/{jid}", token=token)
    note(
        status == 200 and final.get("status") == "cancelled" and "error" not in final,
        "race: cancelled without an error",
        f"status={final.get('status')!r} error={final.get('error')!r}",
    )
    # the next job must run normally (the chip was not left busy)
    status, _, job2 = raw(base, "POST", GEN, {"prompt": "after race"}, token)
    done = wait_for(base, token, job2.get("id"), ("completed",), timeout=30)
    note(
        done,
        "race: the next job completes",
        f"status={job_status(base, token, job2.get('id'))!r}",
    )
    print(f"RACE_TASK_ID {jid}")


def check_queue_full(base, token, max_queue):
    ids = []
    last = None
    for i in range(max_queue + 1):
        status, headers, payload = raw(base, "POST", GEN, {"prompt": f"q{i}"}, token)
        last = (status, headers, payload)
        if status == 202:
            ids.append(payload.get("id"))
    status, headers, payload = last
    note(
        status == 429 and headers.get("Retry-After") == "60",
        f"429 when the queue is full (MAX_QUEUE_SIZE={max_queue})",
        f"status={status} retry_after={headers.get('Retry-After')!r} "
        f"message={cc.error_message(payload)!r}",
    )
    # free the queue again: cancel everything, wait, delete
    for jid in ids:
        raw(base, "POST", f"{GEN}/{jid}/cancel", token=token)
    for jid in ids:
        wait_for(base, token, jid, ("cancelled", "completed", "failed"), timeout=60)
        raw(base, "DELETE", f"{GEN}/{jid}", token=token)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--phase",
        default="main",
        choices=["main", "sync-error", "limits", "warmup", "cancel-race"],
    )
    parser.add_argument(
        "--max-queue",
        type=int,
        default=0,
        help="MAX_QUEUE_SIZE of the server; 0 skips the 429 check",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    token = os.environ.get("VIDEO_API_KEY", "your-secret-key")

    if args.phase == "warmup":
        check_warmup(base, token)
    else:
        status, _, live = raw(base, "GET", "/tt-liveness")
        if not (status == 200 and live.get("model_ready") is True):
            print("server is not ready")
            return 1
        if args.phase == "main":
            check_auth(base, token)
            check_queue_cancel_delete(base, token)
            check_sync_ok(base, token)
            check_download(base, token)
            if args.max_queue > 0:
                check_queue_full(base, token, args.max_queue)
        elif args.phase == "sync-error":
            check_sync_error(base, token)
        elif args.phase == "limits":
            check_job_limit(base, token)
        elif args.phase == "cancel-race":
            check_cancel_race(base, token)
    print(f"\n{cc.PASS} passed, {cc.FAIL} failed")
    return 0 if cc.FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
