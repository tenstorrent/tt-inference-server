# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""dots.ocr local-endpoint customer demo.

Sends a batch of real document images to a locally-running, OpenAI-compatible
dots.ocr endpoint (the tt-inference-server vLLM server, optionally behind the
validated-geometry resize proxy) and collects the OCR transcriptions.

The demo is intentionally a *plain* OpenAI Chat Completions client: it shows a
customer exactly what an integration looks like (POST /v1/chat/completions with
an image_url data URI) and produces three artifacts:

  1. <out>/txt/<image>.txt   -- raw transcription per image
  2. <out>/results.json      -- structured results (latency, tokens, text)
  3. <out>/report.html       -- self-contained gallery (image + extracted text)

Run it with an interpreter that has ``requests`` (e.g. the tt-metal python_env):

    "$TT_METAL_HOME"/python_env/bin/python \
        demo/dots_ocr_endpoint_demo.py \
        --image-dir /path/to/sample_docs --limit 20

(``--image-dir`` may also be supplied via the ``SAMPLE_DOCS`` environment variable.)

Endpoint selection:
  --base-url defaults to the resize proxy (http://127.0.0.1:8001/v1) so that
  arbitrary-resolution images are letterboxed to dots.ocr's validated grid and
  never crash the vision tower. Point it at :8000/v1 to hit the server directly.

Authentication:
  The tt-inference-server vLLM server requires a JWT bearer token whose payload
  is EXACTLY {"team_id": "tenstorrent", "token_id": "debug-test"} signed with the
  server's JWT_SECRET (see utils/vllm_run_utils.py::get_encoded_api_key). Rather
  than hand-minting that token, this demo mints it automatically from $JWT_SECRET
  when no explicit --api-key/$DEMO_API_KEY is given. Set the same JWT_SECRET you
  launched the server with, and the demo just works. (If the server was started
  with --no-auth, no token is needed.)
"""

import argparse
import base64
import hashlib
import hmac
import json
import mimetypes
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from html import escape
from pathlib import Path

import requests

# The S2 decode path batches at most DP-size (8) concurrent sequences before the
# generator raises NotImplementedError, so cap demo concurrency at the DP batch.
MAX_CONCURRENCY = 8

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
DEFAULT_PROMPT = (
    "Extract all the text content from this image, preserving the reading order."
)
# Placeholder default for --api-key; treated as "not provided" so we fall back to
# minting a token from $JWT_SECRET.
PLACEHOLDER_API_KEY = "your-secret-key"
# The server compares the bearer token against jwt.encode(this_payload, JWT_SECRET)
# — it must match utils/vllm_run_utils.py::get_encoded_api_key EXACTLY.
CANONICAL_JWT_PAYLOAD = {"team_id": "tenstorrent", "token_id": "debug-test"}


def _b64url(raw: bytes) -> str:
    """Base64url without padding (JWT segment encoding)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def mint_bearer_from_secret(jwt_secret: str) -> str:
    """Mint the exact bearer token the server accepts, from its JWT_SECRET.

    The server (utils/vllm_run_utils.py::get_encoded_api_key) hands vLLM a fixed
    HS256 token string and vLLM authorizes by EXACT STRING MATCH — it does not
    decode the JWT. So the client token must be byte-identical to the server's.

    We build the JWT by hand (HMAC-SHA256) instead of using PyJWT, because PyJWT
    versions order the header fields differently ({"alg","typ"} in >=2.5 vs
    {"typ","alg"} in older releases), which changes the signature and breaks the
    string match (the cause of spurious 401s when the client's PyJWT differs from
    the server image's). Hardcoding the modern header order — what the server
    image emits — makes this work on any host, with or without PyJWT installed.
    """
    header_b64 = _b64url(b'{"alg":"HS256","typ":"JWT"}')
    payload_b64 = _b64url(
        json.dumps(CANONICAL_JWT_PAYLOAD, separators=(",", ":")).encode("utf-8")
    )
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(
        jwt_secret.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    return f"{header_b64}.{payload_b64}.{_b64url(signature)}"


def resolve_api_key(cli_api_key: str) -> str | None:
    """Decide the bearer token to send.

    Precedence: an explicit, non-placeholder --api-key/$DEMO_API_KEY wins. Else,
    if $JWT_SECRET is set, mint the canonical token from it. Else, send no token
    (works only if the server runs with --no-auth).
    """
    if cli_api_key and cli_api_key != PLACEHOLDER_API_KEY:
        print("Auth     : using explicitly-provided API key")
        return cli_api_key

    jwt_secret = os.getenv("JWT_SECRET")
    if jwt_secret:
        token = mint_bearer_from_secret(jwt_secret)
        print("Auth     : minted bearer token from $JWT_SECRET")
        return token

    print(
        "Auth     : no --api-key and no $JWT_SECRET set; sending unauthenticated "
        "requests (only works if the server was started with --no-auth)",
        file=sys.stderr,
    )
    return None


def discover_images(image_dir: Path, limit: int) -> list[Path]:
    images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    return images[:limit]


def to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# Each worker thread gets its own requests.Session; a Session is not guaranteed
# thread-safe when shared across concurrent requests, so we key one per thread.
_thread_local = threading.local()


def get_session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        _thread_local.session = sess
    return sess


def ocr_one(session, base_url, api_key, model, prompt, max_tokens, path, timeout):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                # Image MUST precede the text prompt. dots.ocr was trained/validated
                # with image-before-text ordering (see the validated
                # tt_symbiote .../models/dots_ocr inputs builder, which places
                # {image} before {text}). With text-first ordering the model can
                # emit <|endoftext|> as its very first greedy token on some pages
                # (observed on pages 1 & 8 of the t1 set), yielding empty output.
                # Image-first matches the trained format and produces full output.
                "content": [
                    {"type": "image_url", "image_url": {"url": to_data_uri(path)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.perf_counter()
    resp = session.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens")
    tps = (completion_tokens / elapsed) if completion_tokens and elapsed else None
    return {
        "image": path.name,
        "text": choice["message"]["content"],
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "latency_s": round(elapsed, 2),
        "tokens_per_s": round(tps, 2) if tps else None,
    }


def write_html(report_path: Path, results, meta, image_dir: Path):
    rows = []
    for r in results:
        img_path = image_dir / r["image"]
        if r.get("ok", True) and "text" in r:
            try:
                img_uri = to_data_uri(img_path)
                img_tag = f'<img class="doc" src="{img_uri}" alt="{escape(r["image"])}"/>'
            except OSError:
                img_tag = "<em>(image unavailable)</em>"
            badge = (
                f'<span class="ok">finish={escape(str(r.get("finish_reason")))}</span> '
                f'<span class="m">{r.get("completion_tokens")} tok</span> '
                f'<span class="m">{r.get("latency_s")} s</span> '
                f'<span class="m">{r.get("tokens_per_s")} tok/s</span>'
            )
            text_html = f'<pre class="ocr">{escape(r["text"])}</pre>'
        else:
            img_tag = "<em>(no image)</em>"
            badge = f'<span class="err">ERROR: {escape(str(r.get("error", "")))}</span>'
            text_html = ""
        rows.append(
            f'<div class="card"><div class="hd"><h3>{escape(r["image"])}</h3>'
            f"<div class=\"badges\">{badge}</div></div>"
            f'<div class="body"><div class="imgwrap">{img_tag}</div>'
            f'<div class="txtwrap">{text_html}</div></div></div>'
        )
    conc = meta.get("concurrency", 1)
    agg = meta.get("aggregate_tps")
    wall = meta.get("wall_clock_s")
    summary = (
        f'<div class="summary"><b>{meta["model"]}</b> via '
        f'<code>{escape(meta["base_url"])}</code><br>'
        f'{meta["ok"]}/{meta["total"]} succeeded &middot; '
        f'concurrency {conc} &middot; '
        f'wall-clock {wall} s &middot; '
        f'avg latency {meta["avg_latency"]} s &middot; '
        f'avg {meta["avg_tps"]} tok/s &middot; '
        f'aggregate {agg} tok/s &middot; '
        f'generated {datetime.now():%Y-%m-%d %H:%M}</div>'
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>dots.ocr endpoint demo</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}}
 header{{padding:20px 28px;background:#161a22;border-bottom:1px solid #262b36}}
 header h1{{margin:0 0 6px;font-size:20px}}
 .summary{{font-size:13px;color:#9aa4b2;line-height:1.6}}
 .summary code{{color:#7fd1ff}}
 .card{{margin:18px 28px;background:#161a22;border:1px solid #262b36;border-radius:10px;overflow:hidden}}
 .hd{{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;background:#1b2029;border-bottom:1px solid #262b36}}
 .hd h3{{margin:0;font-size:15px}}
 .badges span{{font-size:12px;margin-left:8px;padding:2px 8px;border-radius:10px;background:#262b36}}
 .ok{{color:#7CFC9B}} .err{{color:#ff8080}} .m{{color:#9aa4b2}}
 .body{{display:flex;gap:16px;padding:16px;align-items:flex-start}}
 .imgwrap{{flex:0 0 42%}}
 img.doc{{width:100%;border-radius:6px;border:1px solid #262b36;background:#fff}}
 .txtwrap{{flex:1;min-width:0}}
 pre.ocr{{white-space:pre-wrap;word-wrap:break-word;font-size:13px;line-height:1.45;
   max-height:560px;overflow:auto;margin:0;padding:12px;background:#0f1115;border-radius:6px;border:1px solid #262b36}}
</style></head><body>
<header><h1>dots.ocr &mdash; local endpoint demo</h1>{summary}</header>
{''.join(rows)}
</body></html>"""
    report_path.write_text(html, encoding="utf-8")


def main():
    default_out = str(Path(__file__).resolve().parent / "demo_outputs")
    ap = argparse.ArgumentParser(description="dots.ocr local endpoint OCR demo")
    ap.add_argument(
        "--image-dir",
        default=os.getenv("SAMPLE_DOCS"),
        help="Folder of images to OCR (defaults to the $SAMPLE_DOCS env var).",
    )
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--base-url", default=os.getenv("DEMO_BASE_URL", "http://127.0.0.1:8001/v1"))
    ap.add_argument("--model", default=os.getenv("DEMO_MODEL", "rednote-hilab/dots.ocr"))
    ap.add_argument("--api-key", default=os.getenv("DEMO_API_KEY", PLACEHOLDER_API_KEY))
    ap.add_argument("--prompt", default=os.getenv("DEMO_PROMPT", DEFAULT_PROMPT))
    ap.add_argument("--max-tokens", type=int, default=int(os.getenv("DEMO_MAX_TOKENS", "2048")))
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default=default_out)
    ap.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("DEMO_CONCURRENCY", "1")),
        help=(
            "Number of requests to send in parallel (default 1 = sequential). "
            f"Concurrency drives the server's continuous batching; capped at "
            f"{MAX_CONCURRENCY} (the DP batch size). Higher values raise "
            "aggregate throughput up to the batch limit."
        ),
    )
    args = ap.parse_args()

    if not args.image_dir:
        ap.error("--image-dir is required (or set the $SAMPLE_DOCS environment variable)")

    image_dir = Path(args.image_dir)
    images = discover_images(image_dir, args.limit)
    if not images:
        print(f"No images found in {image_dir}", file=sys.stderr)
        sys.exit(1)

    out = Path(args.out)
    (out / "txt").mkdir(parents=True, exist_ok=True)

    api_key = resolve_api_key(args.api_key)

    concurrency = max(1, args.concurrency)
    if concurrency > MAX_CONCURRENCY:
        print(
            f"Warning  : --concurrency {concurrency} exceeds the DP batch size; "
            f"capping at {MAX_CONCURRENCY} (higher would hit the S2 decode "
            "NotImplementedError on the server).",
            file=sys.stderr,
        )
        concurrency = MAX_CONCURRENCY

    print(f"Endpoint : {args.base_url}")
    print(f"Model    : {args.model}")
    print(f"Images   : {len(images)} from {image_dir}")
    print(f"Prompt   : {args.prompt!r}")
    print(f"Conc.    : {concurrency} ({'sequential' if concurrency == 1 else 'continuous batching'})")
    print(f"Output   : {out}\n")
    print(f"{'#':>3}  {'image':24}  {'status':8}  {'lat(s)':>7}  {'tok':>5}  {'tok/s':>6}  preview")
    print("-" * 100)

    results = [None] * len(images)
    print_lock = threading.Lock()

    def run_one(idx: int, path: Path):
        try:
            r = ocr_one(
                get_session(), args.base_url, api_key, args.model,
                args.prompt, args.max_tokens, path, args.timeout,
            )
            r["ok"] = True
            (out / "txt" / f"{path.stem}.txt").write_text(r["text"], encoding="utf-8")
            preview = " ".join(r["text"].split())[:48]
            status = r["finish_reason"] or "ok"
            line = (f"{idx + 1:>3}  {path.name:24}  {status:8}  {r['latency_s']:>7}  "
                    f"{str(r['completion_tokens']):>5}  {str(r['tokens_per_s']):>6}  {preview}")
        except Exception as e:  # noqa: BLE001 - demo: report and continue
            r = {"image": path.name, "ok": False, "error": str(e)}
            line = (f"{idx + 1:>3}  {path.name:24}  {'ERROR':8}  {'-':>7}  "
                    f"{'-':>5}  {'-':>6}  {e}")
        with print_lock:
            print(line)
        results[idx] = r

    wall_t0 = time.perf_counter()
    if concurrency == 1:
        for i, path in enumerate(images):
            run_one(i, path)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(run_one, i, path) for i, path in enumerate(images)]
            for f in as_completed(futures):
                f.result()
    wall_clock = time.perf_counter() - wall_t0

    ok = [r for r in results if r and r.get("ok")]
    lat = [r["latency_s"] for r in ok if r.get("latency_s")]
    tps = [r["tokens_per_s"] for r in ok if r.get("tokens_per_s")]
    avg_latency = round(sum(lat) / len(lat), 2) if lat else None
    avg_tps = round(sum(tps) / len(tps), 2) if tps else None
    total_completion = sum(
        r.get("completion_tokens") or 0 for r in ok
    )
    # Aggregate throughput across the whole batch: the metric that rises with
    # concurrency as the server overlaps requests via continuous batching.
    aggregate_tps = round(total_completion / wall_clock, 2) if wall_clock else None
    meta = {
        "model": args.model,
        "base_url": args.base_url,
        "prompt": args.prompt,
        "total": len(results),
        "ok": len(ok),
        "concurrency": concurrency,
        "wall_clock_s": round(wall_clock, 2),
        "avg_latency": avg_latency,
        "avg_tps": avg_tps,
        "aggregate_tps": aggregate_tps,
    }
    (out / "results.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8"
    )
    write_html(out / "report.html", results, meta, image_dir)

    print("-" * 100)
    print(f"\nDone: {len(ok)}/{len(results)} succeeded | concurrency {concurrency} | "
          f"wall-clock {round(wall_clock, 2)}s")
    print(f"  per-request : avg latency {avg_latency}s | avg {avg_tps} tok/s")
    print(f"  aggregate   : {total_completion} completion tokens in {round(wall_clock, 2)}s "
          f"= {aggregate_tps} tok/s")
    if concurrency > 1:
        print("  (aggregate tok/s should exceed the sequential run as the server "
              "batches requests; wall-clock should be well below N x sequential.)")
    print(f"  txt     : {out / 'txt'}")
    print(f"  json    : {out / 'results.json'}")
    print(f"  report  : {out / 'report.html'}")


if __name__ == "__main__":
    main()
