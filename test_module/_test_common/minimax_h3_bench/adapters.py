# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""HTTP transport and the tt-media-server MiniMax-H3 request shapes.

    t2va   -> POST /v1/videos/generations            {prompt, duration_seconds, seed, aspect_ratio}
    fl2va  -> POST /v1/videos/generations/i2v        + image_prompts[{image: b64, frame_pos: 0|-1}]
    ref2va -> POST /v1/videos/generations/ref2va     + references{images|videos|audios: [{b64}]}

The lifecycle is asynchronous: 202 + job id, poll ``GET /v1/videos/generations/{id}``
until a terminal status, fetch ``.../download``. The client never names a step
count -- the deployment runs a fixed 50 and refuses ``num_inference_steps``. Media
is inlined as base64 and checked against the deployment's own caps (read from its
openapi.json) before anything is sent, so an oversize input is reported as the
capability limit it is rather than as an opaque 422 after a large upload.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

from . import models as M

# minimax_h3_client.API_KEY_ENV_VARS is the master copy; the engine also runs standalone
# (no aiohttp), so the import is guarded and a unit test keeps the two tuples equal.
try:
    from ..minimax_h3_client import API_KEY_ENV_VARS
except ImportError:  # pragma: no cover - standalone use
    API_KEY_ENV_VARS = ("API_KEY", "MINIMAX_API_KEY", "TT_MINIMAX_API_KEY")

ROUTE = {
    "t2va": "/v1/videos/generations",
    "fl2va": "/v1/videos/generations/i2v",
    "ref2va": "/v1/videos/generations/ref2va",
}
JOBS = "/v1/videos/jobs"
TASKS = ("t2va", "fl2va", "ref2va")
FIXED_STEPS = 50
DEFAULT_API_KEY = "your-secret-key"


def resolve_api_key() -> str:
    """The same names, in the same order, as every other H3 test in this repo
    (``minimax_h3_client.API_KEY_ENV_VARS``)."""
    for name in API_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return DEFAULT_API_KEY


def http(method, url, body=None, ctype=None, timeout=120, raw=False, headers=None):
    """(status, parsed_or_bytes). Never raises: an HTTP error status is returned with
    its body, a transport failure is ``(0, {'error': ...})``."""
    req = urllib.request.Request(url, data=body, method=method)
    if ctype:
        req.add_header("Content-Type", ctype)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            code = resp.status
    except urllib.error.HTTPError as exc:
        data = exc.read()
        code = exc.code
    except Exception as exc:  # noqa: BLE001 - every transport failure is one class here
        return 0, {"error": f"{type(exc).__name__}: {exc}"}
    if raw:
        return code, data
    try:
        return code, json.loads(data)
    except ValueError:
        return code, {
            "error": "non-json",
            "body": data[:400].decode("utf-8", "replace"),
        }


class TenstorrentH3:
    """One deployment, one task family per base URL."""

    exposes_progress = True  # queued -> in_progress -> completed
    MAX_IMAGE_B64 = 10_000_000  # ImagePromptEntry.image maxLength default
    MAX_MEDIA_B64 = 80_000_000  # MediaSource.b64 maxLength default
    FIXED_STEPS = FIXED_STEPS  # what the deployment runs; never sent in a request

    def __init__(self, endpoints: dict, api_key: str | None = None, on_submit=None):
        """``endpoints``: task -> base URL (``all`` serves every task)."""
        self.endpoints = {k: v.rstrip("/") for k, v in endpoints.items()}
        self.api_key = api_key if api_key is not None else resolve_api_key()
        self.on_submit = on_submit
        self._caps_cache: dict = {}

    def base_url(self, task: str) -> str:
        if task in self.endpoints:
            return self.endpoints[task]
        if "all" in self.endpoints:
            return self.endpoints["all"]
        raise LookupError(
            f"no endpoint serves {task}: configured {sorted(self.endpoints)}"
        )

    def serves(self, task: str) -> bool:
        return task in self.endpoints or "all" in self.endpoints

    def auth(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # -- capability limits ----------------------------------------------------------
    def caps(self, task: str) -> dict:
        base = self.base_url(task)
        if base not in self._caps_cache:
            caps = {"image": self.MAX_IMAGE_B64, "media": self.MAX_MEDIA_B64}
            code, spec = http(
                "GET", f"{base}/openapi.json", timeout=20, headers=self.auth()
            )
            if code == 200 and isinstance(spec, dict):
                schemas = spec.get("components", {}).get("schemas", {})
                img = (
                    schemas.get("ImagePromptEntry", {})
                    .get("properties", {})
                    .get("image", {})
                    .get("maxLength")
                )
                b64 = [
                    a.get("maxLength")
                    for a in schemas.get("MediaSource", {})
                    .get("properties", {})
                    .get("b64", {})
                    .get("anyOf", [])
                    if isinstance(a, dict) and a.get("maxLength")
                ]
                if isinstance(img, int):
                    caps["image"] = img
                if b64:
                    caps["media"] = int(b64[0])
            self._caps_cache[base] = caps
        return self._caps_cache[base]

    def _b64(self, name: str, cap: int):
        with open(M.asset(name), "rb") as fh:
            enc = base64.b64encode(fh.read()).decode("ascii")
        if len(enc) > cap:
            return (
                None,
                f"{name}: base64 is {len(enc)} chars, over this endpoint's {cap}-char cap",
            )
        return enc, None

    def _references(self, case: dict):
        refs, oversize = {}, []
        cap = self.caps(case["task"])["media"]
        for key in ("images", "videos", "audios"):
            out = []
            for name in case.get(key) or []:
                enc, why = self._b64(name, cap)
                if why:
                    oversize.append(why)
                else:
                    out.append({"b64": enc})
            if out:
                refs[key] = out
        return refs, oversize

    def _keyframes(self, case: dict):
        prompts, oversize = [], []
        cap = self.caps(case["task"])["image"]
        for i, name in enumerate(case.get("images") or []):
            enc, why = self._b64(name, cap)
            if why:
                oversize.append(why)
            else:
                prompts.append({"image": enc, "frame_pos": 0 if i == 0 else -1})
        return prompts, oversize

    @staticmethod
    def reject(reasons) -> tuple:
        """400 (not 0): a payload this endpoint cannot accept is deterministic, never a
        transport failure, and must not be retried."""
        return 400, {"error": {"message": M.CLIENT_REJECT + "; ".join(reasons)}}

    def build(self, case: dict):
        """(path, payload, grace_s) or (None, (code, body), 0) when the case cannot be sent."""
        task = case["task"]
        payload = {
            "prompt": M.read_prompt(case["prompt"]),
            "duration_seconds": int(case["duration_s"]),
            "seed": int(case["seed"]),
            "aspect_ratio": case.get("aspect_ratio", "16:9"),
        }
        grace = 60
        if task == "t2va":
            pass
        elif task == "fl2va":
            prompts, oversize = self._keyframes(case)
            if oversize or not prompts:
                return (
                    None,
                    self.reject(oversize or ["fl2va requires at least one image"]),
                    0,
                )
            payload["image_prompts"] = prompts
            grace = 180
        elif task == "ref2va":
            refs, oversize = self._references(case)
            if oversize or not refs:
                return (
                    None,
                    self.reject(oversize or ["ref2va requires at least one reference"]),
                    0,
                )
            payload["references"] = refs
            grace = 600
        else:
            return None, self.reject([f"unsupported task {task}"]), 0
        return ROUTE[task], payload, grace

    # -- lifecycle -----------------------------------------------------------------
    def post(self, case: dict):
        """((code, body), task); the body carries ``id`` on 202."""
        task = case["task"]
        path, payload, grace = self.build(case)
        if path is None:
            return payload, task
        code, data = http(
            "POST", f"{self.base_url(task)}{path}", body=json.dumps(payload).encode(),
            ctype="application/json", timeout=60 + grace, headers=self.auth(), raw=True,
        )  # fmt: skip
        if not isinstance(data, (bytes, bytearray)):
            return (code, data), task
        if code == 200 and (b"ftyp" in data[:64] or b"moov" in data[:64]):
            return self.reject([
                f"synchronous deployment: POST {path} answered 200 with the finished clip "
                f"({len(data)} bytes of mp4) instead of 202 + a job id"
            ]), task  # fmt: skip
        try:
            body = json.loads(data)
        except ValueError:
            body = {"error": "non-json", "body": data[:400].decode("utf-8", "replace")}
        job = body.get("id") if isinstance(body, dict) else None
        if job and self.on_submit:
            self.on_submit(task, job)
        return (code, body), task

    def status(self, task: str, job_id: str):
        code, body = http(
            "GET",
            f"{self.base_url(task)}/v1/videos/generations/{job_id}",
            headers=self.auth(),
        )
        return (
            code,
            (body or {}).get("status") if isinstance(body, dict) else None,
            body,
        )

    def content(self, task: str, job_id: str, dest: str):
        code, data = http(
            "GET", f"{self.base_url(task)}/v1/videos/generations/{job_id}/download",
            timeout=600, raw=True, headers=self.auth(),
        )  # fmt: skip
        ok = code == 200 and isinstance(data, (bytes, bytearray)) and len(data) > 1000
        if ok:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(data)
        return ok, code

    def cancel(self, task: str, job_id: str) -> int:
        code, _ = http(
            "POST",
            f"{self.base_url(task)}/v1/videos/generations/{job_id}/cancel",
            headers=self.auth(),
            timeout=30,
        )
        M.log(f"  [cancel] {task} job {job_id} -> HTTP {code}")
        return code

    def get(self, task: str, path: str, key: bool = True, timeout: int = 30):
        return http(
            "GET",
            f"{self.base_url(task)}{path}",
            timeout=timeout,
            headers=self.auth() if key else None,
        )

    def post_json(
        self, task: str, path: str, payload, key: bool = True, timeout: int = 30
    ):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return http(
            "POST", f"{self.base_url(task)}{path}", body=body, ctype="application/json",
            timeout=timeout, headers=self.auth() if key else None,
        )  # fmt: skip
