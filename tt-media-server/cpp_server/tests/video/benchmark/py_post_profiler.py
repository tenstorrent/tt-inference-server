# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# Run from tt-media-server with the same environment as the Python server, e.g.
#   PROF_OUT=/tmp/py_times.json python <path>/py_post_profiler.py
"""Starts the Python media server with timing wrappers around the POST /v1/videos/generations path.

Nothing in the repository is changed: the wrappers are applied at runtime, after
`main` is imported and before uvicorn starts. Timings (ns) are written to
$PROF_OUT as JSON when the process exits.

Measured:
  outer            ASGI middleware added last (outermost): request arrives -> response sent
  submit           open_ai_api.video._submit_video_request (the endpoint body)
  <helper names>   each helper the endpoint calls (media checks, url resolution, ...)
  model_dump       VideoGenerateRequest.model_dump
  create_job       BaseJobService.create_job (includes the two below)
  start_event      BaseJobService._createStartEvent
  jm_create_job    JobManager.create_job
  observe_request  observe_video_request (runs in the job task, not in the request path)
"""

import atexit
import functools
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import main  # noqa: E402  (builds the FastAPI app)
import open_ai_api.video as video  # noqa: E402
import model_services.base_job_service as bjs  # noqa: E402
import model_services.video_service as vs  # noqa: E402
import utils.job_manager as jm  # noqa: E402

TIMES = {}


def rec(name, ns):
    TIMES.setdefault(name, []).append(ns)


def wrap_sync(obj, attr, name):
    orig = getattr(obj, attr)

    @functools.wraps(orig)
    def w(*a, **k):
        t = time.perf_counter_ns()
        try:
            return orig(*a, **k)
        finally:
            rec(name, time.perf_counter_ns() - t)

    setattr(obj, attr, w)


def wrap_async(obj, attr, name):
    orig = getattr(obj, attr)

    @functools.wraps(orig)
    async def w(*a, **k):
        t = time.perf_counter_ns()
        try:
            return await orig(*a, **k)
        finally:
            rec(name, time.perf_counter_ns() - t)

    setattr(obj, attr, w)


wrap_async(video, "_submit_video_request", "submit")
for helper in (
    "_enforce_inline_media_total",
    "_check_reference_counts",
    "_redact_media_urls",
    "_check_request_media",
):
    wrap_sync(video, helper, helper)
for helper in ("_resolve_image_prompt_urls", "_resolve_media_source_urls"):
    wrap_async(video, helper, helper)
wrap_sync(video.VideoGenerateRequest, "model_dump", "model_dump")
wrap_async(bjs.BaseJobService, "create_job", "create_job")
wrap_sync(bjs.BaseJobService, "_createStartEvent", "start_event")
wrap_async(jm.JobManager, "create_job", "jm_create_job")
wrap_sync(vs, "observe_video_request", "observe_request")


import fastapi.routing as froute  # noqa: E402
import fastapi.dependencies.utils as fdeps  # noqa: E402

# FastAPI internals, looked up as module globals at call time.
wrap_async(froute, "solve_dependencies", "fa_solve_dependencies")
wrap_async(froute, "run_endpoint_function", "fa_run_endpoint")
wrap_async(froute, "serialize_response", "fa_serialize_response")
wrap_async(fdeps, "request_body_to_args", "fa_body_validation")
wrap_async(fdeps, "run_in_threadpool", "fa_sync_dependency_threadpool")

CURRENT = {"post": False}


class Timed:
    """Proxy around one ASGI layer: inclusive time of that layer and everything below."""

    def __init__(self, app, name):
        self.app = app
        self.name = name

    def __getattr__(self, item):
        return getattr(self.app, item)

    async def __call__(self, scope, receive, send):
        if not CURRENT["post"]:
            return await self.app(scope, receive, send)
        t = time.perf_counter_ns()
        try:
            await self.app(scope, receive, send)
        finally:
            rec("layer:" + self.name, time.perf_counter_ns() - t)


class Outer:
    def __init__(self, app):
        self.app = app
        self.wrapped = False

    def wrap_layers(self):
        # walk the middleware chain below this one and time each layer
        parent, depth = self, 0
        while hasattr(parent, "app") and parent.app is not None and depth < 20:
            child = parent.app
            if isinstance(child, Timed):
                break
            name = f"{depth}:{type(child).__name__}"
            parent.app = Timed(child, name)
            parent = child
            depth += 1
        self.wrapped = True

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or not scope.get("path", "").endswith("/videos/generations")
        ):
            return await self.app(scope, receive, send)
        if not self.wrapped:
            self.wrap_layers()
        CURRENT["post"] = True
        t = time.perf_counter_ns()
        try:
            await self.app(scope, receive, send)
        finally:
            rec("outer", time.perf_counter_ns() - t)
            CURRENT["post"] = False


main.app.add_middleware(Outer)


def dump():
    with open(os.environ["PROF_OUT"], "w") as f:
        json.dump(TIMES, f)


atexit.register(dump)


def _periodic():
    # uvicorn re-raises SIGTERM on shutdown, so atexit may not run: dump often.
    while True:
        time.sleep(3)
        try:
            dump()
        except Exception:
            pass


import threading  # noqa: E402

threading.Thread(target=_periodic, daemon=True).start()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(main.app, host="127.0.0.1", port=8000, lifespan="on")
