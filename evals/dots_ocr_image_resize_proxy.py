# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Validated-geometry image-resize reverse proxy for dots.ocr evals.

The tt_symbiote dots.ocr vision tower is only tuned for the validated patch
grid (84, 132) -- box 1848x1176 px. Arbitrary-resolution images (as fed by
lmms-eval tasks such as textvqa_val / infovqa_val) select an untuned vision-SDPA
program config; the TTNN op throws and the torch fallback then dies on the
``rot_mats`` kwarg, killing EngineCore (see tt_symbiote core/run_config.py).

This proxy sits in front of the vLLM server and force-letterboxes every image in
``/v1/chat/completions`` requests to exactly 1848x1176 (preserve aspect ratio,
white pad), so requests always land on the validated grid. Every other path
(``/health``, ``/v1/models``, ...) is forwarded verbatim.

Because images are letterboxed to a fixed box, resulting eval scores are a
LOWER BOUND on supported inputs, not a clean published comparison. This is a
serving-side workaround; the real fix is arbitrary-resolution vision-SDPA
buckets in tt_symbiote (model owner).

Run with an interpreter that has ``requests`` + ``Pillow`` (e.g. the tt-metal
python_env):

    "$TT_METAL_HOME"/python_env/bin/python \
        evals/dots_ocr_image_resize_proxy.py

Env overrides:
    PROXY_PORT      (default 8001)   port this proxy listens on
    UPSTREAM_URL    (default http://127.0.0.1:8000)   the real vLLM server
    TARGET_WIDTH    (default 1848)
    TARGET_HEIGHT   (default 1176)
    FORWARD_TIMEOUT (default 1800)   seconds; first request captures a trace
"""

import base64
import binascii
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
from PIL import Image

PROXY_PORT = int(os.environ.get("PROXY_PORT", "8001"))
UPSTREAM_URL = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:8000").rstrip("/")
TARGET_WIDTH = int(os.environ.get("TARGET_WIDTH", "1848"))
TARGET_HEIGHT = int(os.environ.get("TARGET_HEIGHT", "1176"))
FORWARD_TIMEOUT = float(os.environ.get("FORWARD_TIMEOUT", "1800"))

CHAT_PATH = "/v1/chat/completions"

# Hop-by-hop headers must not be forwarded (RFC 7230 6.1).
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}

_counter_lock = threading.Lock()
_images_rewritten = 0
_unrecognized_parts = 0


def _log(msg: str) -> None:
    print(f"[resize-proxy] {msg}", file=sys.stderr, flush=True)


def _letterbox(img: Image.Image) -> Image.Image:
    """Resize ``img`` to fit TARGET_WIDTH x TARGET_HEIGHT, preserving aspect
    ratio, padding the remainder with white. Output is always exactly the
    target size so the server always sees the validated patch grid."""
    img = img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), (255, 255, 255))
    scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), (255, 255, 255))
    canvas.paste(resized, ((TARGET_WIDTH - new_w) // 2, (TARGET_HEIGHT - new_h) // 2))
    return canvas


def _rewrite_data_uri(url: str) -> str:
    """If ``url`` is a base64 image data URI, letterbox it to the validated
    geometry and return a new PNG data URI. Otherwise return ``url`` unchanged.
    """
    global _images_rewritten
    if not isinstance(url, str) or not url.startswith("data:image"):
        return url
    try:
        header, b64 = url.split(",", 1)
    except ValueError:
        return url
    if ";base64" not in header:
        # Non-base64 data URI (rare); leave untouched.
        return url
    try:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        boxed = _letterbox(img)
        out = io.BytesIO()
        boxed.save(out, format="PNG")
        new_b64 = base64.b64encode(out.getvalue()).decode("ascii")
    except (binascii.Error, OSError, ValueError) as e:
        _log(f"WARNING: could not decode/resize image ({e!r}); forwarding original")
        return url
    with _counter_lock:
        _images_rewritten += 1
    return f"data:image/png;base64,{new_b64}"


def _rewrite_chat_body(body: bytes) -> bytes:
    """Letterbox every image_url in an OpenAI chat-completions payload."""
    global _unrecognized_parts
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        _log(f"WARNING: chat body not JSON ({e!r}); forwarding unchanged")
        return body

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return body

    changed = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            # string content (text-only) -- nothing to resize
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict) and "url" in image_url:
                    new_url = _rewrite_data_uri(image_url["url"])
                    if new_url != image_url["url"]:
                        image_url["url"] = new_url
                        changed = True
                elif isinstance(image_url, str):
                    new_url = _rewrite_data_uri(image_url)
                    if new_url != image_url:
                        part["image_url"] = new_url
                        changed = True
                else:
                    with _counter_lock:
                        _unrecognized_parts += 1
                    _log(f"NOTE: unrecognized image_url shape: {type(image_url)!r}")

    if not changed:
        return body
    return json.dumps(payload).encode("utf-8")


class _ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence default per-request stderr spam
        pass

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _forward(self, method: str, body: bytes) -> None:
        url = f"{UPSTREAM_URL}{self.path}"
        fwd_headers = {
            k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP
        }
        try:
            resp = requests.request(
                method,
                url,
                data=body if body else None,
                headers=fwd_headers,
                timeout=FORWARD_TIMEOUT,
                stream=False,
            )
        except requests.RequestException as e:
            _log(f"ERROR forwarding {method} {self.path}: {e!r}")
            self.send_error(502, f"upstream request failed: {e}")
            return

        content = resp.content
        self.send_response(resp.status_code)
        for k, v in resp.headers.items():
            if k.lower() in _HOP_BY_HOP:
                continue
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if content:
            self.wfile.write(content)

    def do_GET(self):
        self._forward("GET", b"")

    def do_DELETE(self):
        self._forward("DELETE", self._read_body())

    def do_POST(self):
        body = self._read_body()
        if self.path == CHAT_PATH or self.path.endswith(CHAT_PATH):
            body = _rewrite_chat_body(body)
        self._forward("POST", body)


def main() -> None:
    _log(
        f"listening on :{PROXY_PORT} -> upstream {UPSTREAM_URL}; "
        f"letterboxing chat images to {TARGET_WIDTH}x{TARGET_HEIGHT}"
    )
    server = ThreadingHTTPServer(("0.0.0.0", PROXY_PORT), _ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _log(f"shutting down; images_rewritten={_images_rewritten} unrecognized_parts={_unrecognized_parts}")
        server.server_close()


if __name__ == "__main__":
    main()
