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
DEFAULT_AUTHORIZATION = "your-secret-key"
DATA_DIR = os.path.join(os.path.dirname(__file__), "example_data")

# Mutable runtime configuration (can be overridden via CLI)
RETRIES = DEFAULT_RETRIES
RETRY_WAIT = DEFAULT_RETRY_WAIT


def get_hf_headers(accept: str) -> dict:
    headers = {
        "Accept": accept,
        "User-Agent": USER_AGENT,
    }
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        data = resp.read()
        print(f"HF Download Content Type: {content_type}")
        if "application/json" not in content_type:
            # Try to decode as JSON regardless; produce clearer error if not JSON
            return json.loads(data.decode("utf-8"))
        return json.loads(data.decode("utf-8"))


def http_get_bytes(url: str, timeout: int = DEFAULT_TIMEOUT) -> Tuple[bytes, str]:
    req = urllib.request.Request(url, headers=get_hf_headers("*/*"))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        return resp.read(), content_type


def fetch_rows_metadata(samples: int, config: str, split: str) -> dict:
    url = build_rows_url(config=config, split=split, offset=0, length=samples)
    # Perform a single, headerless GET exactly like a plain curl request
    req = urllib.request.Request(url)
    print(f"Fetching HF dataset metadata from {url}")
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
        data = resp.read()
        print("Done fetching rows metadata.")
        return json.loads(data.decode("utf-8"))


def fetch_audio_bytes_for_row(row_index: int, config: str, split: str) -> bytes:
    # Try query-parameter JSON envelope first (preferred)
    json_asset_url = build_asset_audio_json_url(
        config=config, split=split, row_index=row_index
    )
    envelope = http_get_json(json_asset_url)
    direct_url = envelope.get("url") or envelope.get("gated_url")
    if direct_url:
        data, _ = http_get_bytes(direct_url)
        return data

    # Fallback: path-style endpoint, request JSON envelope if possible
    asset_url = build_asset_audio_url(config=config, split=split, row_index=row_index)
    envelope2 = http_get_json(asset_url)
    direct_url2 = envelope2.get("url") or envelope2.get("gated_url")
    if not direct_url2:
        raise RuntimeError(
            f"Asset API returned JSON without 'url' for row_index={row_index} ({asset_url})"
        )
    data2, _ = http_get_bytes(direct_url2)
    return data2


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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            return None
        body = resp.read().decode("utf-8")
        return json.loads(body)


def post_json(
    host: str,
    path: str,
    payload: dict,
    authorization: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[int, bytes, str]:
    url = f"{host.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/json, application/x-ndjson",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {authorization}",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        return resp.status, resp.read(), content_type


def post_streaming(
    host: str,
    path: str,
    payload: dict,
    authorization: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> int:
    url = f"{host.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/json, application/x-ndjson",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {authorization}",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    start = time.perf_counter()
    first_content_time: Optional[float] = None
    received_chunks = 0
    final_duration = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            obj = json.loads(line)
            received_chunks += 1
            text = obj.get("text", "")
            if "duration" in obj:
                final_duration = obj.get("duration")
            if text and first_content_time is None:
                first_content_time = time.perf_counter() - start
            chunk_id = obj.get("chunk_id", received_chunks)
            print(f"[stream] chunk={chunk_id} text={text!r}")
    total = time.perf_counter() - start
    ttft = first_content_time if first_content_time is not None else 0.0
    dur_display = (
        f"{final_duration:.2f}s" if isinstance(final_duration, (int, float)) else "N/A"
    )
    print(
        f"[stream] done in {total:.2f}s | TTFT={ttft:.2f}s | duration={dur_display} | chunks={received_chunks}"
    )
    return 200


def download_audio_b64_for_index(
    rows: list, index: int, config: str, split: str, total: int
) -> Tuple[Optional[str], bool]:
    row_obj = rows[index].get("row", {})
    filename = extract_audio_filename(row_obj=row_obj, fallback_index=index)
    audio_bytes, downloaded = load_or_download_audio(
        row_obj=row_obj,
        row_index=index,
        config=config,
        split=split,
        filename=filename,
    )
    return base64_encode(audio_bytes), downloaded


def ensure_data_dir() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


def extract_audio_filename(row_obj: dict, fallback_index: int) -> str:
    audio_entry = row_obj.get("audio")

    # HF datasets-server rows endpoint often returns a dict with "path"
    if isinstance(audio_entry, dict):
        candidate = audio_entry.get("path") or audio_entry.get("file")
        if candidate:
            base = os.path.basename(candidate)
            if base:
                return base

    # Older shape: list of dicts with path/file
    if isinstance(audio_entry, list) and audio_entry:
        candidate = audio_entry[0].get("path") or audio_entry[0].get("file")
        if candidate:
            base = os.path.basename(candidate)
            if base:
                return base

    # Fall back to a simple indexed name to keep files distinct
    return f"sample-{fallback_index}.flac"


def load_or_download_audio(
    row_obj: dict, row_index: int, config: str, split: str, filename: str
) -> Tuple[bytes, bool]:
    data_dir = ensure_data_dir()
    filepath = os.path.join(data_dir, filename)
    if os.path.isfile(filepath):
        print(f"[cache] using {filename}")
        with open(filepath, "rb") as f:
            return f.read(), False
    audio_bytes = fetch_audio_from_rows_src(row_obj)
    with open(filepath, "wb") as f:
        f.write(audio_bytes)
    print(f"[download] saved {filename}")
    return audio_bytes, True


def download_audio_samples(
    samples: int, config: str, split: str, concurrency: int
) -> Tuple[List[Optional[str]], int, int]:
    rows_meta = fetch_rows_metadata(samples=samples, config=config, split=split)
    rows = rows_meta.get("rows", [])
    total = min(len(rows), samples)

    if total <= 0:
        print("Completed: successes=0, failures=0, total=0")
        return [], 0, 0

    max_workers = max(1, int(concurrency))
    downloaded_b64: List[Optional[str]] = [None] * total
    downloaded_new = 0
    cache_hits = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                download_audio_b64_for_index, rows, i, config, split, total
            ): i
            for i in range(total)
        }
        for fut in as_completed(future_to_index):
            i = future_to_index[fut]
            b64_value, downloaded = fut.result()
            downloaded_b64[i] = b64_value
            if downloaded:
                downloaded_new += 1
            else:
                cache_hits += 1

    downloaded_count = sum(1 for x in downloaded_b64 if x is not None)
    print(
        f"Download phase complete: ready={downloaded_count} "
        f"(new={downloaded_new}, cached={cache_hits}) in {ensure_data_dir()}"
    )
    return downloaded_b64, total, downloaded_count


def stream_single_request(
    host: str,
    payload: dict,
    request_id: int,
    authorization: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[bool, Optional[float]]:
    start = time.perf_counter()
    url = f"{host.rstrip('/')}/audio/transcriptions"
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/json, application/x-ndjson",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {authorization}",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    first_content_time: Optional[float] = None
    received_chunks = 0
    final_duration = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            obj = json.loads(line)
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
    if isinstance(final_duration, (int, float)) and total > 0:
        rtr = float(final_duration) / float(total)
        rtr_display = f"{rtr:.2f}x"
        dur_display = f"{final_duration:.2f}s"
    else:
        rtr_display = "N/A"
        dur_display = "N/A"
    print(
        f"[stream] E2EL={total:.2f}s | TTFT={ttft:.2f}s | audio_time={dur_display} | RTR={rtr_display} | chunks={received_chunks}"
    )
    return True, (
        float(final_duration) if isinstance(final_duration, (int, float)) else None
    )


def nonstream_single_request(
    host: str,
    payload: dict,
    authorization: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[bool, Optional[float]]:
    start = time.perf_counter()
    status, body, content_type = post_json(
        host, "/audio/transcriptions", payload, authorization, timeout=timeout
    )
    elapsed = time.perf_counter() - start
    if status == 200 and body:
        text = None
        duration = None
        if "application/json" in content_type:
            obj = json.loads(body.decode("utf-8"))
            text = obj.get("text")
            duration = obj.get("duration")
        dur_display = (
            f"{duration:.2f}s" if isinstance(duration, (int, float)) else "N/A"
        )
        if isinstance(duration, (int, float)) and elapsed > 0:
            rtr = float(duration) / float(elapsed)
            rtr_display = f"{rtr:.2f}x"
        else:
            rtr_display = "N/A"
        print(
            f"[non-stream] E2EL={elapsed:.2f}s | audio_time={dur_display} | RTR={rtr_display} | text={(text or '')!r}"
        )
        return True, (float(duration) if isinstance(duration, (int, float)) else None)
    body_text = (
        body.decode("utf-8", errors="replace")
        if isinstance(body, (bytes, bytearray))
        else str(body)
    )
    eprint(f"[non-stream] HTTP {status}: {body_text}")
    return False, None


def make_streaming_inference_requests(
    b64_queue: List[str],
    host: str,
    preprocess: bool,
    diarization: bool,
    concurrency: int,
    authorization: str,
) -> Tuple[int, int, float, float, float]:
    successes = 0
    failures = 0
    total_audio_time = 0.0
    counted_durations = 0
    inference_phase_start = time.perf_counter()

    max_workers = max(1, int(concurrency))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for req_id, b64 in enumerate(b64_queue, start=1):
            payload = {
                "file": b64,
                "stream": True,
                "is_preprocessing_enabled": bool(preprocess),
                "perform_diarization": bool(diarization),
            }
            futures.append(
                executor.submit(
                    stream_single_request,
                    host,
                    payload,
                    req_id,
                    authorization,
                    DEFAULT_TIMEOUT,
                )
            )

        for fut in as_completed(futures):
            ok, duration = fut.result()
            if ok:
                successes += 1
                if isinstance(duration, (int, float)):
                    total_audio_time += float(duration)
                    counted_durations += 1
            else:
                failures += 1

    inference_wall_time = time.perf_counter() - inference_phase_start
    return successes, failures, inference_wall_time, total_audio_time, counted_durations


def make_non_streaming_inference_requests(
    b64_queue: List[str],
    host: str,
    preprocess: bool,
    diarization: bool,
    concurrency: int,
    authorization: str,
) -> Tuple[int, int, float, float, float]:
    successes = 0
    failures = 0
    total_audio_time = 0.0
    counted_durations = 0
    inference_phase_start = time.perf_counter()

    max_workers = max(1, int(concurrency))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for b64 in b64_queue:
            payload = {
                "file": b64,
                "stream": False,
                "is_preprocessing_enabled": bool(preprocess),
                "perform_diarization": bool(diarization),
            }
            futures.append(
                executor.submit(
                    nonstream_single_request,
                    host,
                    payload,
                    authorization,
                    DEFAULT_TIMEOUT,
                )
            )

        for fut in as_completed(futures):
            ok, duration = fut.result()
            if ok:
                successes += 1
                if isinstance(duration, (int, float)):
                    total_audio_time += float(duration)
                    counted_durations += 1
            else:
                failures += 1

    inference_wall_time = time.perf_counter() - inference_phase_start
    return successes, failures, inference_wall_time, total_audio_time, counted_durations


def make_inference_requests(
    b64_queue: List[str],
    host: str,
    stream: bool,
    preprocess: bool,
    diarization: bool,
    concurrency: int,
    authorization: str,
) -> Tuple[int, int, float, float, float]:
    if stream:
        return make_streaming_inference_requests(
            b64_queue, host, preprocess, diarization, concurrency, authorization
        )
    return make_non_streaming_inference_requests(
        b64_queue, host, preprocess, diarization, concurrency, authorization
    )


def run(args: argparse.Namespace) -> int:
    # Use default constants for datasets server, retries, retry wait, config, split, and timeout
    config = DEFAULT_CONFIG
    split = DEFAULT_SPLIT

    # Health check
    health = check_server_health(args.host, timeout=DEFAULT_TIMEOUT)
    if not health:
        eprint(
            f"Service not healthy at {args.host}/tt-liveness. Start the server and try again."
        )
        return 1

    # Step 1: Download audio samples
    downloaded_b64, total, downloaded_count = download_audio_samples(
        samples=args.samples,
        config=config,
        split=split,
        concurrency=args.concurrency,
    )

    successes = 0
    failures = 0

    if total <= 0:
        print("Completed: successes=0, failures=0, total=0")
        return 0

    failures += total - downloaded_count
    if downloaded_count == 0:
        print(f"Completed: successes=0, failures={failures}, total={total}")
        return 3

    # Step 2: Make inference requests
    b64_queue = [b64 for b64 in downloaded_b64 if b64 is not None]
    successes2, failures2, inference_wall_time, total_audio_time, counted_durations = (
        make_inference_requests(
            b64_queue=b64_queue,
            host=args.host,
            stream=bool(args.stream),
            preprocess=bool(args.preprocess),
            diarization=bool(args.diarization),
            concurrency=int(getattr(args, "concurrency", DEFAULT_CONCURRENCY)),
            authorization=args.authorization,
        )
    )
    successes += successes2
    failures += failures2

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
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Number of samples to send (default 32)",
    )
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Enable server-side audio preprocessing",
    )
    parser.add_argument(
        "--diarization",
        action="store_true",
        help="Enable diarization when preprocessing is enabled",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:8000",
        help="Inference server host base URL",
    )
    parser.add_argument(
        "--authorization",
        type=str,
        default=DEFAULT_AUTHORIZATION,
        help='Authorization token sent as "Bearer <token>"',
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Number of concurrent inference requests (default 1)",
    )
    args = parser.parse_args(argv)
    if args.samples <= 0:
        args.samples = 1
    if args.concurrency <= 0:
        args.concurrency = 1
    # No automatic override; use exactly what the user provided/defaults.
    return args


if __name__ == "__main__":
    sys.exit(run(parse_args()))
