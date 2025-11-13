#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# A standard-library-only CLI that:
# - Downloads N audio samples from Hugging Face Datasets Server (LibriSpeech)
# - Base64-encodes each sample
# - Sends transcription requests to a local server
# - Supports --stream and non-streaming modes
#
# References:
# - Dataset: openslr/librispeech_asr (https://huggingface.co/datasets/openslr/librispeech_asr)
#
# Notes:
# - Uses Hugging Face Datasets Server HTTP APIs, no extra dependencies required.
# - Intended to work with tt-media-server audio endpoint: /audio/transcriptions

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed


DATASET_REPO = "openslr/librispeech_asr"
DATASET_REPO_STUB = "openslr___librispeech_asr"  # datasets-server path form
DATASETS_SERVER = "https://datasets-server.huggingface.co"

DEFAULT_CONFIG = "clean"
DEFAULT_SPLIT = "test"

DEFAULT_SAMPLES = 32
DEFAULT_TIMEOUT = 30
USER_AGENT = "tt-inference-server-example/1.0"
DEFAULT_RETRIES = 3
DEFAULT_RETRY_WAIT = 2.0
DEFAULT_CONCURRENCY = 1

# Mutable runtime configuration (can be overridden via CLI)
RETRIES = DEFAULT_RETRIES
RETRY_WAIT = DEFAULT_RETRY_WAIT

def get_hf_headers(accept: str) -> dict:
    headers = {
        "Accept": accept,
        "User-Agent": USER_AGENT,
    }
    # token = os.environ.get("HF_TOKEN")
    # if token:
    #     headers["Authorization"] = f"Bearer {token}"
    return headers


def eprint(message: str) -> None:
    sys.stderr.write(message + "\n")


def build_rows_url(config: str, split: str, offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET_REPO,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    return f"{DATASETS_SERVER}/rows?{query}"


def build_asset_audio_url(config: str, split: str, row_index: int) -> str:
    # Returns an endpoint that should yield the audio asset content (often WAV) or a redirect
    # Path shape: /assets/{repo_stub}/--/{config}/{split}/{row_index}/audio
    quoted_config = urllib.parse.quote(config, safe="")
    quoted_split = urllib.parse.quote(split, safe="")
    return f"{DATASETS_SERVER}/assets/{DATASET_REPO_STUB}/--/{quoted_config}/{quoted_split}/{row_index}/audio"

def build_asset_audio_json_url(config: str, split: str, row_index: int) -> str:
    # Query-param style endpoint returning JSON envelope with signed URL
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET_REPO,
            "config": config,
            "split": split,
            "row": row_index,
            "col": "audio",
        }
    )
    return f"{DATASETS_SERVER}/assets?{query}"

def http_get_json(url: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    req = urllib.request.Request(url, headers=get_hf_headers("application/json"))
    print(f"HF Download URL: {url}")
    last_err: Optional[Exception] = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                data = resp.read()
                if "application/json" not in content_type:
                    # Try to decode as JSON regardless; produce clearer error if not JSON
                    try:
                        return json.loads(data.decode("utf-8"))
                    except Exception:
                        raise RuntimeError(
                            f"Expected JSON from {url} but got Content-Type='{content_type}'"
                        )
                return json.loads(data.decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(
                f"HTTP {e.code} from {url}\nResponse: {body}"
            )
        except urllib.error.URLError as e:
            last_err = RuntimeError(f"Failed to reach {url}: {e}")
        if attempt < RETRIES - 1:
            time.sleep(RETRY_WAIT * (2 ** attempt))
    # Exhausted retries
    assert last_err is not None
    raise last_err


def http_get_bytes(url: str, timeout: int = DEFAULT_TIMEOUT) -> Tuple[bytes, str]:
    req = urllib.request.Request(url, headers=get_hf_headers("*/*"))
    last_err: Optional[Exception] = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                return resp.read(), content_type
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(
                f"HTTP {e.code} from {url}\nResponse: {body}"
            )
        except urllib.error.URLError as e:
            last_err = RuntimeError(f"Failed to reach {url}: {e}")
        if attempt < RETRIES - 1:
            time.sleep(RETRY_WAIT * (2 ** attempt))
    assert last_err is not None
    raise last_err


def fetch_rows_metadata(samples: int, config: str, split: str) -> dict:
    url = build_rows_url(config=config, split=split, offset=0, length=samples)
    # Perform a single, headerless GET exactly like a plain curl request
    req = urllib.request.Request(url)
    print(f"HF Download URL: {url}")
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
        data = resp.read()
        return json.loads(data.decode("utf-8"))


def fetch_audio_bytes_for_row(row_index: int, config: str, split: str) -> bytes:
    # Try query-parameter JSON envelope first (preferred)
    json_asset_url = build_asset_audio_json_url(config=config, split=split, row_index=row_index)
    try:
        envelope = http_get_json(json_asset_url)
        direct_url = envelope.get("url") or envelope.get("gated_url")
        if direct_url:
            data, _ = http_get_bytes(direct_url)
            return data
    except Exception as e:
        eprint(f"[debug] asset envelope fetch failed for row {row_index}: {e}")

    # Fallback: path-style endpoint, request JSON envelope if possible
    asset_url = build_asset_audio_url(config=config, split=split, row_index=row_index)
    try:
        envelope2 = http_get_json(asset_url)
        direct_url2 = envelope2.get("url") or envelope2.get("gated_url")
        if not direct_url2:
            raise RuntimeError(
                f"Asset API returned JSON without 'url' for row_index={row_index} ({asset_url})"
            )
        data2, _ = http_get_bytes(direct_url2)
        return data2
    except Exception:
        # Final attempt: direct bytes (may still fail with 403)
        data3, _ = http_get_bytes(asset_url)
        return data3


def base64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

def fetch_audio_from_rows_src(row: dict) -> bytes:
    audio_list = row.get("audio") or []
    if not audio_list:
        raise RuntimeError("Row does not contain 'audio' field")
    src = audio_list[0].get("src")
    if not src:
        raise RuntimeError("Row audio entry missing 'src' signed URL")
    # The 'src' contains Expires, Signature, and Key-Pair-Id query params. Use it as-is.
    data, _ = http_get_bytes(src)
    return data


def check_server_health(host: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[dict]:
    url = f"{host.rstrip('/')}/tt-liveness"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            body = resp.read().decode("utf-8")
            try:
                return json.loads(body)
            except Exception:
                return {}
    except Exception:
        return None


def post_json(host: str, path: str, payload: dict, timeout: int = DEFAULT_TIMEOUT) -> Tuple[int, bytes, str]:
    url = f"{host.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/json, application/x-ndjson",
        "Content-Type": "application/json",
        "Authorization": "Bearer your-secret-key",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            return resp.status, resp.read(), content_type
    except urllib.error.HTTPError as e:
        body = e.read()  # bytes
        content_type = e.headers.get("Content-Type", "") if e.headers else ""
        return e.code, body, content_type


def post_streaming(host: str, path: str, payload: dict, timeout: int = DEFAULT_TIMEOUT) -> int:
    url = f"{host.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/json, application/x-ndjson",
        "Content-Type": "application/json",
        "Authorization": "Bearer your-secret-key",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    start = time.perf_counter()
    first_content_time: Optional[float] = None
    received_chunks = 0
    final_duration = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                body = resp.read().decode("utf-8", errors="replace")
                eprint(f"[stream] HTTP {resp.status}: {body}")
                return resp.status
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Ignore malformed line; continue
                    continue
                received_chunks += 1
                text = obj.get("text", "")
                if "duration" in obj:
                    final_duration = obj.get("duration")
                if text and first_content_time is None:
                    first_content_time = time.perf_counter() - start
                # Minimal per-chunk log
                chunk_id = obj.get("chunk_id", received_chunks)
                print(f"[stream] chunk={chunk_id} text={text!r}")
        total = time.perf_counter() - start
        ttft = first_content_time if first_content_time is not None else 0.0
        dur_display = f"{final_duration:.2f}s" if isinstance(final_duration, (int, float)) else "N/A"
        print(f"[stream] done in {total:.2f}s | TTFT={ttft:.2f}s | duration={dur_display} | chunks={received_chunks}")
        return 200
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        eprint(f"[stream] HTTP {e.code}: {body}")
        return e.code
    except urllib.error.URLError as e:
        eprint(f"[stream] URL error: {e}")
        return 599


def run(args: argparse.Namespace) -> int:
    # Use default constants for datasets server, retries, retry wait, config, split, and timeout
    config = DEFAULT_CONFIG
    split = DEFAULT_SPLIT

    # Health check
    health = check_server_health(args.host, timeout=DEFAULT_TIMEOUT)
    if not health:
        eprint(f"Service not healthy at {args.host}/tt-liveness. Start the server and try again.")
        return 1

    # Ensure dataset rows are accessible (fail clearly if not)
    try:
        rows_meta = fetch_rows_metadata(samples=args.samples, config=config, split=split)
    except Exception as e:
        eprint(
            "Failed to access Hugging Face dataset rows.\n"
            f"Dataset: {DATASET_REPO}, config: {config}, split: {split}\n"
            f"Details: {e}\n"
            "See dataset page: https://huggingface.co/datasets/openslr/librispeech_asr"
        )
        return 2

    rows = rows_meta.get("rows", [])
    total = min(len(rows), args.samples)

    def download_audio_b64_for_index(i: int) -> Optional[str]:
        row_obj = rows[i].get("row", {})
        try:
            audio_bytes = fetch_audio_from_rows_src(row_obj)
        except Exception:
            try:
                audio_bytes = fetch_audio_bytes_for_row(i, config=config, split=split)
            except Exception as e:
                eprint(
                    f"[{i+1}/{total}] Failed to fetch audio for row {i} "
                    f"(config={config}, split={split}): {e}"
                )
                return None
        return base64_encode(audio_bytes)

    def run_inference_for_b64(audio_b64: str, request_id: int) -> Tuple[bool, Optional[float]]:
        payload = {
            "file": audio_b64,
            "stream": bool(args.stream),
            "is_preprocessing_enabled": bool(args.preprocess),
            "perform_diarization": bool(args.diarization),
        }
        if args.stream:
            # post_streaming internally logs chunk updates; enhance its final log to include RTR/E2EL
            start = time.perf_counter()
            # We need TTFT and audio duration; reuse post_streaming's behavior but compute E2EL/RTR here
            url = f"{args.host.rstrip('/')}/audio/transcriptions"
            data = json.dumps(payload).encode("utf-8")
            headers = {
                "Accept": "application/json, application/x-ndjson",
                "Content-Type": "application/json",
                "Authorization": "Bearer your-secret-key",
            }
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            first_content_time: Optional[float] = None
            received_chunks = 0
            final_duration = None
            try:
                with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
                    if resp.status != 200:
                        body = resp.read().decode("utf-8", errors="replace")
                        eprint(f"[stream] HTTP {resp.status}: {body}")
                        return False, None
                    while True:
                        raw_line = resp.readline()
                        if not raw_line:
                            break
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        received_chunks += 1
                        text = obj.get("text", "")
                        if "duration" in obj:
                            final_duration = obj.get("duration")
                        if text and first_content_time is None:
                            first_content_time = time.perf_counter() - start
                        chunk_id = obj.get("chunk_id", received_chunks)
                        print(f"[stream] req={request_id} chunk={chunk_id} text={text!r}")
                total = time.perf_counter() - start  # E2EL
                ttft = first_content_time if first_content_time is not None else 0.0
                # Compute RTR if duration is numeric and total > 0
                if isinstance(final_duration, (int, float)) and total > 0:
                    rtr = float(final_duration) / float(total)
                    rtr_display = f"{rtr:.2f}x"
                    dur_display = f"{final_duration:.2f}s"
                else:
                    rtr_display = "N/A"
                    dur_display = "N/A"
                print(f"[stream] E2EL={total:.2f}s | TTFT={ttft:.2f}s | audio_time={dur_display} | RTR={rtr_display} | chunks={received_chunks}")
                return True, (float(final_duration) if isinstance(final_duration, (int, float)) else None)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                eprint(f"[stream] HTTP {e.code}: {body}")
                return False, None
            except urllib.error.URLError as e:
                eprint(f"[stream] URL error: {e}")
                return False, None
        start = time.perf_counter()
        status, body, content_type = post_json(args.host, "/audio/transcriptions", payload, timeout=DEFAULT_TIMEOUT)
        elapsed = time.perf_counter() - start
        if status == 200 and body:
            text = None
            duration = None
            try:
                if "application/json" in content_type:
                    obj = json.loads(body.decode("utf-8"))
                    text = obj.get("text")
                    duration = obj.get("duration")
            except Exception:
                pass
            dur_display = f"{duration:.2f}s" if isinstance(duration, (int, float)) else "N/A"
            if isinstance(duration, (int, float)) and elapsed > 0:
                rtr = float(duration) / float(elapsed)
                rtr_display = f"{rtr:.2f}x"
            else:
                rtr_display = "N/A"
            print(f"[non-stream] E2EL={elapsed:.2f}s | audio_time={dur_display} | RTR={rtr_display} | text={(text or '')!r}")
            return True, (float(duration) if isinstance(duration, (int, float)) else None)
        body_text = body.decode("utf-8", errors="replace") if isinstance(body, (bytes, bytearray)) else str(body)
        eprint(f"[non-stream] HTTP {status}: {body_text}")
        return False, None

    successes = 0
    failures = 0

    if total <= 0:
        print(f"Completed: successes=0, failures=0, total=0")
        return 0

    # Phase 1: download all samples (may use concurrency), but do not start inference yet.
    max_workers = max(1, int(getattr(args, "concurrency", DEFAULT_CONCURRENCY)))
    downloaded_b64: List[Optional[str]] = [None] * total
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(download_audio_b64_for_index, i): i for i in range(total)}
        for fut in as_completed(future_to_index):
            i = future_to_index[fut]
            try:
                downloaded_b64[i] = fut.result()
            except Exception as e:
                eprint(f"[download-worker] unhandled exception for index {i}: {e}")
                downloaded_b64[i] = None
    downloaded_count = sum(1 for x in downloaded_b64 if x is not None)
    failures += (total - downloaded_count)
    print(f"Downloaded {downloaded_count} samples from dataset: {DATASET_REPO}")

    if downloaded_count == 0:
        print(f"Completed: successes=0, failures={failures}, total={total}")
        return 3

    # Phase 2: run inference only on successfully downloaded samples.
    b64_queue = [b64 for b64 in downloaded_b64 if b64 is not None]
    inference_phase_start = time.perf_counter()
    total_audio_time = 0.0
    counted_durations = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_inference_for_b64, b64, req_id) for req_id, b64 in enumerate(b64_queue, start=1)]
        for fut in as_completed(futures):
            try:
                ok, duration = fut.result()
            except Exception as e:
                eprint(f"[inference-worker] unhandled exception: {e}")
                ok = False
                duration = None
            if ok:
                successes += 1
                if isinstance(duration, (int, float)):
                    total_audio_time += float(duration)
                    counted_durations += 1
            else:
                failures += 1
    inference_wall_time = time.perf_counter() - inference_phase_start

    if inference_wall_time > 0 and counted_durations > 0:
        full_rtr = total_audio_time / inference_wall_time
        full_rtr_display = f"{full_rtr:.2f}x"
    else:
        full_rtr_display = "N/A"
    print(
        f"Completed: successes={successes}, failures={failures}, total={total} | "
        f"inference_time={inference_wall_time:.2f}s | audio_total={total_audio_time:.2f}s | FULL_RTR={full_rtr_display}"
    )
    return 0 if failures == 0 else 3


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download N LibriSpeech samples and send audio transcription requests."
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Number of samples to send (default 32)")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    parser.add_argument("--preprocess", action="store_true", help="Enable server-side audio preprocessing")
    parser.add_argument("--diarization", action="store_true", help="Enable diarization when preprocessing is enabled")
    parser.add_argument("--host", type=str, default="http://localhost:8000", help="Inference server host base URL")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Number of concurrent inference requests (default 1)")
    args = parser.parse_args(argv)
    if args.samples <= 0:
        args.samples = 1
    if args.concurrency <= 0:
        args.concurrency = 1
    # No automatic override; use exactly what the user provided/defaults.
    return args


if __name__ == "__main__":
    sys.exit(run(parse_args()))


