#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""HTTP checks for the job based video API (/v1/videos/...).

POST /generations answers 202 with a job (id, status=queued); the client polls
GET /generations/{id} and fetches the mp4 from /generations/{id}/download.

The same cases run against a local HTTP stub of the service and against
tt_media_server_cpp. --require-file is for the device runner: the finished job
must download as a non-empty mp4 (the mock only has mock:// results, so its
download is 404).
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

PASS = 0
FAIL = 0


def checked_base_url(value):
    """Accept only http(s) URLs of this machine, or of a host named in
    VIDEO_TEST_ALLOWED_HOSTS (comma separated, for a test run against a remote T3K).

    The test scripts send requests to whatever --base-url says. The check keeps them
    from being pointed at an arbitrary host by mistake. The URL is rebuilt from its
    parts, so a path, query or credentials in the argument are dropped.
    """
    parsed = urllib.parse.urlparse(value)
    allowed = {"127.0.0.1", "localhost", "::1"}
    extra = os.environ.get("VIDEO_TEST_ALLOWED_HOSTS", "").split(",")
    allowed |= {h.strip() for h in extra if h.strip()}
    if parsed.scheme not in ("http", "https") or parsed.hostname not in allowed:
        raise argparse.ArgumentTypeError(
            f"{value!r}: only http(s) URLs of localhost are accepted; "
            "set VIDEO_TEST_ALLOWED_HOSTS=<host> to test a remote server"
        )
    return f"{parsed.scheme}://{parsed.netloc}"


def note(status, name, detail):
    global PASS, FAIL
    if status:
        PASS += 1
        print(f"PASS  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}")
    print(f"      {detail}")


def request(base, method, path, body=None, auth=None):
    data = None if body is None else body.encode()
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("content-type", "application/json")
    if auth:
        req.add_header("authorization", f"Bearer {auth}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode()
            return resp.status, json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            payload = {"raw": raw}
        return exc.status, payload


GEN = "/v1/videos/generations"
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
TERMINAL = ("completed", "failed", "cancelled")


def download(base, job_id, auth=None):
    """Raw GET of the mp4: (status, bytes)."""
    req = urllib.request.Request(f"{base}{GEN}/{job_id}/download")
    if auth:
        req.add_header("authorization", f"Bearer {auth}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.status, exc.read()


def submit(base, auth, body):
    """POST a job. Returns (status, payload)."""
    return request(base, "POST", GEN, json.dumps(body), auth)


def wait_terminal(base, auth, job_id, timeout):
    """Poll GET /generations/{id} until the job is terminal."""
    deadline = time.time() + timeout
    status, job = request(base, "GET", f"{GEN}/{job_id}", auth=auth)
    while status == 200 and job.get("status") not in TERMINAL:
        if time.time() > deadline:
            break
        time.sleep(1)
        status, job = request(base, "GET", f"{GEN}/{job_id}", auth=auth)
    return status, job


def is_job(payload):
    return (
        isinstance(payload.get("id"), str)
        and UUID_RE.match(payload["id"]) is not None
        and payload.get("job_type") == "video"
        and isinstance(payload.get("created_at"), int)
        and isinstance(payload.get("request_parameters"), dict)
    )


def error_message(payload):
    error = payload.get("error")
    if isinstance(error, dict):
        return error.get("message", "")
    return ""


def expect_error(base, auth, name, body, message):
    status, payload = request(base, "POST", "/v1/videos/generations", body, auth)
    ok = status == 422 and error_message(payload) == message
    note(
        ok,
        name,
        f"status={status} message={error_message(payload)!r} expected 422 {message!r}",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, type=checked_base_url)
    parser.add_argument(
        "--require-file",
        action="store_true",
        help="finished job must download as a non-empty mp4",
    )
    parser.add_argument(
        "--job-timeout",
        type=float,
        default=float(os.environ.get("VIDEO_JOB_TIMEOUT", "1800")),
        help="seconds to poll a job until it is terminal",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    auth = os.environ.get("VIDEO_API_KEY", "")

    status, live = request(base, "GET", "/tt-liveness")
    ready = status == 200 and live.get("model_ready") is True
    runner = live.get("runner_in_use") or live.get("runner") or ""
    note(
        ready and bool(runner),
        "liveness",
        f"status={status} model_ready={live.get('model_ready')} runner={runner or 'none'}",
    )
    if not ready:
        print("server is not ready, the remaining checks are not sent")
        return 1

    # 1. submit: 202 + job metadata, not a video path
    status, job = submit(
        base, auth, {"prompt": "a mountain stream", "seed": 1, "num_frames": 81}
    )
    job_id = job.get("id", "")
    note(
        status == 202
        and is_job(job)
        and job.get("status") == "queued"
        and job["request_parameters"].get("prompt") == "a mountain stream"
        and "video_path" not in job,
        "POST returns a job id",
        f"status={status} id={job_id!r} status_field={job.get('status')!r}",
    )

    # 2. polling until terminal
    status, job = wait_terminal(base, auth, job_id, args.job_timeout)
    note(
        status == 200
        and is_job(job)
        and job.get("id") == job_id
        and job.get("status") == "completed"
        and isinstance(job.get("completed_at"), int),
        "polling until completed",
        f"status={status} job_status={job.get('status')!r} error={job.get('error')!r}",
    )

    # 3. the job shows up in the list
    status, jobs = request(base, "GET", "/v1/videos/jobs", auth=auth)
    note(
        status == 200
        and isinstance(jobs, list)
        and any(j.get("id") == job_id for j in jobs),
        "job list",
        f"status={status} has_job={any(j.get('id') == job_id for j in jobs) if isinstance(jobs, list) else False}",
    )

    # 4. download: mp4 on a device runner, 404 for mock:// results
    status, data = download(base, job_id, auth)
    if args.require_file:
        is_mp4 = len(data) >= 8 and data[4:8] == b"ftyp"
        note(
            status == 200 and len(data) > 0 and is_mp4,
            "download mp4",
            f"status={status} bytes={len(data)} mp4={is_mp4}",
        )
        workers = live.get("worker_info")
        if isinstance(workers, dict) and workers:
            alive = [
                wid
                for wid, info in workers.items()
                if isinstance(info, dict) and info.get("is_alive")
            ]
            note(bool(alive), "worker alive", f"alive={alive or 'none'}")
        else:
            note(False, "worker alive", "liveness has no worker_info")
    else:
        note(
            status in (200, 404),
            "download",
            f"status={status} (404 = the result is not a file, e.g. mock://)",
        )

    # 5. cancel on a finished job is 404, like the Python server
    status, payload = request(base, "POST", f"{GEN}/{job_id}/cancel", auth=auth)
    note(
        status == 404,
        "cancel of a finished job",
        f"status={status} message={error_message(payload)!r}",
    )

    # 6. delete finished job, then it is gone
    status, payload = request(base, "DELETE", f"{GEN}/{job_id}", auth=auth)
    note(
        status == 200
        and payload.get("id") == job_id
        and payload.get("object") == "video"
        and payload.get("deleted") is True,
        "delete of a finished job",
        f"status={status} payload={payload!r}",
    )
    status, payload = request(base, "GET", f"{GEN}/{job_id}", auth=auth)
    note(status == 404, "job after delete", f"status={status}")
    status, payload = request(base, "DELETE", f"{GEN}/{job_id}", auth=auth)
    note(status == 404, "delete of a non-existent job", f"status={status}")

    for bad in ("999999999", "abc", "00000000-0000-4000-8000-000000000000"):
        status, payload = request(base, "GET", f"{GEN}/{bad}", auth=auth)
        note(
            status == 404 and error_message(payload) == "Video job not found",
            f"unknown job {bad}",
            f"status={status} message={error_message(payload)!r}",
        )

    expect_error(
        base,
        auth,
        "no prompt",
        "{}",
        "Missing required field: prompt",
    )
    expect_error(
        base,
        auth,
        "empty prompt",
        json.dumps({"prompt": "   "}),
        "prompt must not be empty",
    )
    expect_error(
        base,
        auth,
        "whitespace-only prompt",
        json.dumps({"prompt": "\t\n\r"}),
        "prompt must not be empty",
    )
    for key, value, message in (
        ("num_inference_steps", 0, "num_inference_steps must be between 4 and 50"),
        ("num_inference_steps", -3, "num_inference_steps must be between 4 and 50"),
        ("num_inference_steps", 3, "num_inference_steps must be between 4 and 50"),
        ("num_inference_steps", 51, "num_inference_steps must be between 4 and 50"),
        ("height", 0, "height must be positive"),
        ("width", -1, "width must be positive"),
        ("fps", 0, "fps must be positive"),
        ("duration", -2, "duration must be positive"),
        ("num_frames", 0, "num_frames must be positive"),
        ("num_frames", -1, "num_frames must be positive"),
        ("aspect_ratio", "", "aspect_ratio must not be empty"),
        ("prompt", 1, "prompt must be a string"),
        ("height", "480", "height must be an integer"),
        ("fps", "fast", "fps must be a number"),
        ("seed", "one", "seed must be an integer"),
    ):
        body = {"prompt": "a mountain stream", key: value}
        expect_error(base, auth, f"{key}={value!r}", json.dumps(body), message)

    status, payload = request(
        base,
        "POST",
        "/v1/videos/generations",
        json.dumps(
            {
                "prompt": "a mountain stream",
                "negative_prompt": None,
                "num_inference_steps": None,
                "seed": None,
                "aspect_ratio": None,
                "height": None,
                "width": None,
                "fps": None,
                "duration": None,
                "num_frames": None,
            }
        ),
        auth,
    )
    note(
        status == 202 and is_job(payload),
        "explicit null",
        f"status={status} id={payload.get('id')!r}",
    )

    status, payload = request(
        base,
        "POST",
        "/v1/videos/generations",
        json.dumps({"prompt": "a mountain stream", "seed": 0}),
        auth,
    )
    note(
        status == 202 and is_job(payload),
        "seed 0",
        f"status={status} id={payload.get('id')!r}",
    )

    status, payload = request(
        base,
        "POST",
        "/v1/videos/generations",
        json.dumps(
            {
                "prompt": "flowing water",
                "negative_prompt": "blurry",
                "num_inference_steps": 20,
                "seed": 42,
                "aspect_ratio": "16:9",
                "height": 480,
                "width": 832,
                "fps": 16,
                "duration": 5,
                "num_frames": 81,
            }
        ),
        auth,
    )
    note(
        status == 202 and is_job(payload),
        "all fields",
        f"status={status} id={payload.get('id')!r}",
    )

    status, payload = request(
        base,
        "POST",
        "/v1/videos/generations?async=banana",
        json.dumps({"prompt": "a mountain stream"}),
        auth,
    )
    note(
        status == 422 and "async" in error_message(payload),
        "async=banana",
        f"status={status} message={error_message(payload)!r}",
    )

    status, payload = request(
        base, "POST", "/v1/videos/generations", "this is not json", auth
    )
    note(
        status == 422 and error_message(payload) == "Invalid JSON body",
        "invalid JSON",
        f"status={status} message={error_message(payload)!r}",
    )

    status, payload = request(base, "GET", "/no-such-route", auth=auth)
    note(
        status == 404 and bool(error_message(payload)),
        "unknown route",
        f"status={status} message={error_message(payload)!r}",
    )

    print(f"\n{PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
