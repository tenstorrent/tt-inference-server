# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible serving prototype for the dispatch runtime (#16/#25).

Loads ANY HuggingFace model id (or local path) once onto the Tenstorrent device and
serves it over an OpenAI-compatible HTTP API, so any OpenAI client / curl can drive it:

    python -m tt_inference_server.dispatch.serve serve --unsafe \\
        mistral-community/Mistral-7B-v0.3 --port 8000

    curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
      "model": "mistral-community/Mistral-7B-v0.3",
      "messages": [{"role":"user","content":"The capital of France is"}]
    }'

Design notes
------------
* Built on the stdlib http.server (zero extra deps; runs in the tt-metal venv as-is).
* --unsafe is REQUIRED. It is an acknowledgement, not a per-model gate: anything served
  this way carries no correctness/SLA guarantee (validated or not). Without it, serve
  refuses with an explanation. This removes the "I didn't know it might not work"
  objection while we bring models up to snuff.
* The two-tier resolver (#3) decides config + capabilities: validated models resolve from
  model_matrix.toml; novel models auto-derive from the HF config and are flagged community.
* One device, one inference at a time: requests are serialized behind a lock. A failed
  request returns an OpenAI-style error (HTTP 422) and the server stays up for the next one.

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions, POST /v1/completions.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Module-global serving state (set in main()).
_HANDLE = None            # ModelHandle
_MODEL_ID = None          # str shown in the API
_INFER_LOCK = threading.Lock()


# ----------------------------------------------------------------------------
# Prompt rendering
# ----------------------------------------------------------------------------

def _render_chat(messages) -> str:
    """Render OpenAI chat messages to a single prompt string via the tokenizer's chat
    template (preserving multi-turn structure). Falls back to a simple role-tagged join
    when the tokenizer has no chat template."""
    tok = _HANDLE.tokenizer
    msgs = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    parts = [f"{m['role']}: {m['content']}" for m in msgs]
    parts.append("assistant:")
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Inference (serialized) — returns (text, usage_dict) or yields stream deltas
# ----------------------------------------------------------------------------

def _run(prompt: str, max_tokens: int, temperature: float):
    """Run a full (non-streaming) generation. Returns (text, finish_reason, usage)."""
    with _INFER_LOCK:
        text_parts = []
        meta = {"finish_reason": "length", "prompt_tokens": 0, "completion_tokens": 0}
        for piece in _HANDLE.generate_stream(prompt, max_new_tokens=max_tokens,
                                              temperature=temperature, chat=False):
            if isinstance(piece, dict):
                meta = piece
            else:
                text_parts.append(piece)
        usage = {
            "prompt_tokens": meta["prompt_tokens"],
            "completion_tokens": meta["completion_tokens"],
            "total_tokens": meta["prompt_tokens"] + meta["completion_tokens"],
        }
        return "".join(text_parts), meta["finish_reason"], usage


def _run_stream(prompt: str, max_tokens: int, temperature: float):
    """Yield (delta_text_or_None, meta_or_None). Holds the infer lock for the whole stream."""
    with _INFER_LOCK:
        for piece in _HANDLE.generate_stream(prompt, max_new_tokens=max_tokens,
                                             temperature=temperature, chat=False):
            if isinstance(piece, dict):
                yield None, piece
            else:
                yield piece, None


# ----------------------------------------------------------------------------
# HTTP handler
# ----------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("  [serve] %s - %s\n" % (self.address_string(), fmt % args))

    # -- response helpers --
    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, message, etype="invalid_request_error"):
        self._send_json(code, {"error": {"message": message, "type": etype}})

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # -- routes --
    def do_GET(self):
        if self.path == "/health":
            return self._send_json(200, {"status": "ok", "model": _MODEL_ID})
        if self.path == "/v1/models":
            return self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": _MODEL_ID, "object": "model", "created": int(time.time()),
                    "owned_by": "dispatch",
                    "dispatch": {"listed": _HANDLE.listed, "community": _HANDLE.community},
                }],
            })
        return self._error(404, f"unknown path {self.path}", "not_found")

    def do_POST(self):
        try:
            body = self._read_body()
        except Exception as exc:
            return self._error(400, f"invalid JSON body: {exc}")

        if self.path == "/v1/chat/completions":
            return self._handle_completion(body, chat=True)
        if self.path == "/v1/completions":
            return self._handle_completion(body, chat=False)
        return self._error(404, f"unknown path {self.path}", "not_found")

    def _handle_completion(self, body, chat: bool):
        max_tokens = int(body.get("max_tokens", 128))
        temperature = float(body.get("temperature", 1.0))
        stream = bool(body.get("stream", False))
        try:
            if chat:
                messages = body.get("messages")
                if not messages:
                    return self._error(400, "'messages' is required")
                prompt = _render_chat(messages)
            else:
                prompt = body.get("prompt")
                if prompt is None:
                    return self._error(400, "'prompt' is required")
                if isinstance(prompt, list):
                    prompt = prompt[0]
        except Exception as exc:
            return self._error(422, f"prompt construction failed: {exc}", "unsafe_model_failure")

        if stream:
            return self._stream_completion(prompt, max_tokens, temperature, chat)
        return self._full_completion(prompt, max_tokens, temperature, chat)

    def _full_completion(self, prompt, max_tokens, temperature, chat):
        try:
            text, finish_reason, usage = _run(prompt, max_tokens, temperature)
        except Exception as exc:
            # Graceful failure (#25): the model blew up mid-inference; report and stay up.
            return self._error(
                422,
                f"{type(exc).__name__}: {exc}",
                "unsafe_model_failure",
            )
        cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        if chat:
            choice = {"index": 0, "message": {"role": "assistant", "content": text},
                      "finish_reason": finish_reason}
            obj = "chat.completion"
        else:
            choice = {"index": 0, "text": text, "finish_reason": finish_reason}
            obj = "text_completion"
        self._send_json(200, {
            "id": cid, "object": obj, "created": created, "model": _MODEL_ID,
            "choices": [choice], "usage": usage,
        })

    def _stream_completion(self, prompt, max_tokens, temperature, chat):
        cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        obj = "chat.completion.chunk" if chat else "text_completion"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def sse(payload):
            self.wfile.write(b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n")
            self.wfile.flush()

        def chunk(delta, finish=None):
            ch = ({"index": 0, "delta": ({"content": delta} if delta is not None else {}),
                   "finish_reason": finish} if chat
                  else {"index": 0, "text": delta or "", "finish_reason": finish})
            return {"id": cid, "object": obj, "created": created, "model": _MODEL_ID,
                    "choices": [ch]}

        try:
            if chat:
                sse(chunk(None))  # role-priming delta (OpenAI sends an empty assistant delta first)
            finish_reason = "length"
            for delta, meta in _run_stream(prompt, max_tokens, temperature):
                if meta is not None:
                    finish_reason = meta["finish_reason"]
                    continue
                sse(chunk(delta))
            sse(chunk(None, finish=finish_reason))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception as exc:
            # Best-effort error frame; the stream may already be partially sent.
            try:
                sse({"error": {"message": f"{type(exc).__name__}: {exc}",
                               "type": "unsafe_model_failure"}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

_UNSAFE_NOTICE = (
    "Refusing to serve without --unsafe.\n"
    "  Models served by the dispatch prototype carry NO correctness or SLA guarantee\n"
    "  (validated or not). Pass --unsafe to acknowledge this and proceed:\n"
    "      python -m tt_inference_server.dispatch.serve serve --unsafe <model-id>\n"
)


def _ensure_tt_metal_root():
    """Point tt-metal at the correct source tree before ttnn is imported.

    tt-metal/ttnn locate their data (soc_descriptors, kernels, fabric mesh-graph descriptors,
    …) via a family of env vars — TT_METAL_RUNTIME_ROOT for the root, plus per-asset overrides
    like TT_MESH_GRAPH_DESC_PATH and TT_METAL_KERNEL_PATH. (Note: it does NOT read
    TT_METAL_HOME.) A shell carrying stale values from a previous checkout (e.g. ~/tt-metal)
    makes ttnn die with 'bad file: …' or 'mesh graph descriptor not found: …'.

    This venv lives at <tt-metal>/python_env, so sys.prefix.parent is the authoritative root.
    We (1) scrub every TT_* env var whose value is an absolute path that doesn't exist (the
    stale overrides — once unset, tt-metal recomputes the right default from the root), and
    (2) force TT_METAL_RUNTIME_ROOT (+ TT_METAL_HOME for other tooling) to the real root.
    Must run before ttnn is imported (i.e. before load_model)."""
    import os
    from pathlib import Path
    candidate = Path(sys.prefix).parent  # <tt-metal>/python_env -> <tt-metal>
    if not (candidate / "tt_metal" / "soc_descriptors").is_dir():
        cur = os.environ.get("TT_METAL_RUNTIME_ROOT") or os.environ.get("TT_METAL_HOME")
        print(f"[serve] WARNING: {candidate} has no tt_metal/soc_descriptors; relying on "
              f"current root ({cur!r}) — ttnn may fail", flush=True)
        return
    # (1) Scrub stale TT_* path overrides (absolute paths that no longer exist).
    scrubbed = []
    for k, v in list(os.environ.items()):
        if k.startswith("TT_") and v.startswith("/") and not Path(v).exists():
            del os.environ[k]
            scrubbed.append(f"{k}={v}")
    # (2) Force the authoritative root.
    for var in ("TT_METAL_RUNTIME_ROOT", "TT_METAL_HOME"):
        os.environ[var] = str(candidate)
    if scrubbed:
        print(f"[serve] scrubbed {len(scrubbed)} stale TT_* path var(s): "
              + "; ".join(scrubbed), flush=True)
    print(f"[serve] tt-metal root = {candidate}", flush=True)


def main(argv=None):
    _ensure_tt_metal_root()
    parser = argparse.ArgumentParser(prog="tt-inference-server",
                                     description="Dispatch OpenAI-compatible serving prototype")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("serve", help="Load a model and serve it over an OpenAI-compatible API")
    sp.add_argument("model", help="HuggingFace model id or local path")
    sp.add_argument("--unsafe", action="store_true",
                    help="Acknowledge no correctness/SLA guarantee (required)")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--max-seq", type=int, default=2048)
    sp.add_argument("--no-trace", action="store_true",
                    help="Don't reserve a device trace region (disables the traced fast path)")
    sp.add_argument("--runner", default=None, metavar="MODULE:CLASS",
                    help="Use a custom runner (e.g. 'pkg.mod:MyRunner') instead of the generic "
                         "TTModelRunner. Overrides auto-discovery. A runner self-declared by the "
                         "model repo is honored only with --unsafe.")

    args = parser.parse_args(argv)
    if args.command != "serve":
        parser.error("only 'serve' is supported")

    if not args.unsafe:
        sys.stderr.write(_UNSAFE_NOTICE)
        return 2

    global _HANDLE, _MODEL_ID
    _MODEL_ID = args.model

    from tt_inference_server.dispatch import load_model
    print(f"[serve] loading {args.model} (unsafe acknowledged) ...", flush=True)
    try:
        _HANDLE = load_model(
            args.model,
            max_seq=args.max_seq,
            unsafe=True,
            trace_region_size=0 if args.no_trace else 134217728,
            runner=args.runner,
        )
    except Exception as exc:
        # Clean load-time failure (unknown arch / weight mapping / download error).
        sys.stderr.write(
            f"[serve] FAILED to load '{args.model}': {type(exc).__name__}: {exc}\n"
            "  The architecture may be unsupported (introspection/weight mapping failed).\n"
        )
        return 1

    tag = "community/unverified" if _HANDLE.community else "validated"
    print(f"[serve] ready: {args.model} ({tag})", flush=True)

    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[serve] OpenAI-compatible API on {url}", flush=True)
    print(f"[serve]   GET  {url}/v1/models", flush=True)
    print(f"[serve]   POST {url}/v1/chat/completions", flush=True)
    print(f"[serve]   POST {url}/v1/completions", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] shutting down ...", flush=True)
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
