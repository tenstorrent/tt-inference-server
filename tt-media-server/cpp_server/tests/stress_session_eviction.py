#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""
Stress test for SessionManager eviction races (issues #3001 and #2907).

Starts the server with a tiny session pool and hammers it with concurrent
requests that force eviction while sessions are being acquired.  The test
passes if the server stays alive throughout.

Usage (from cpp_server directory):
    python3 tests/stress_session_eviction.py [--port 8765] [--duration 30]
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import threading
import json
import urllib.request
import urllib.error
from collections import Counter

# --------------------------------------------------------------------------- #
API_KEY = "your-secret-key"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# --------------------------------------------------------------------------- #


def http_post(url: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=HEADERS, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {}
    except Exception:
        return -1, {}


def http_get(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except Exception:
        return -1, {}


def wait_for_server(base: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            code, _ = http_get(f"{base}/health")
            if code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# --------------------------------------------------------------------------- #


def create_session_with_slot(base: str, slot_id: int) -> str | None:
    """Create a session with a pre-assigned slot (no IPC/worker needed)."""
    code, body = http_post(f"{base}/v1/sessions", {"slot_id": slot_id})
    if code in (200, 201):
        return body.get("session_id")
    return None


def send_chat(base: str, session_id: str | None = None) -> int:
    """POST a minimal chat completion.  Returns HTTP status code."""
    body: dict = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
        "stream": False,
    }
    if session_id:
        body["session_id"] = session_id
    code, _ = http_post(f"{base}/v1/chat/completions", body)
    return code


def close_session(base: str, session_id: str) -> int:
    req = urllib.request.Request(
        f"{base}/v1/sessions/{session_id}",
        headers=HEADERS,
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:
        return -1


# --------------------------------------------------------------------------- #


def stress_loop(
    base: str,
    session_ids: list[str],
    results: Counter,
    stop_event: threading.Event,
    worker_id: int,
):
    """
    Each worker alternates between:
      a) Acquiring an existing session (triggers acquireSessionSlot).
      b) Creating a new session with the next slot id (triggers eviction).
    """
    slot_counter = 100 + worker_id * 1000  # start past the pre-seeded slots
    while not stop_event.is_set():
        if session_ids and (worker_id % 2 == 0):
            # Re-use an existing session → acquireSessionSlot (in-flight)
            sid = session_ids[worker_id % len(session_ids)]
            code = send_chat(base, sid)
            results[f"chat:{code}"] += 1
        else:
            # Create a new session → createSession → evictOldSessions
            slot_counter += 1
            sid = create_session_with_slot(base, slot_counter % 256)
            if sid:
                results["session:created"] += 1
                session_ids.append(sid)
                if len(session_ids) > 20:
                    session_ids.pop(0)
            else:
                results["session:fail"] += 1


# --------------------------------------------------------------------------- #


def run(args):
    binary = os.path.join(
        os.path.dirname(__file__), "..", "build", "tt_media_server_cpp"
    )
    binary = os.path.realpath(binary)
    if not os.path.exists(binary):
        print(f"ERROR: binary not found at {binary}", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    env.update(
        {
            "MAX_SESSIONS_COUNT": "6",
            "SESSION_EVICTION_RATE": "80",
            "SESSION_EVICTION_COUNT": "2",
            "DEVICE_IDS": "(0)",
            "LLM_DEVICE_BACKEND": "mock",
            "TT_LOG_LEVEL": "warn",
        }
    )

    base = f"http://127.0.0.1:{args.port}"
    log_path = "/tmp/stress_eviction_server.log"
    print(f"Starting server on port {args.port} (log → {log_path}) …")
    with open(log_path, "w") as log_fh:
        server = subprocess.Popen(
            [binary, "-p", str(args.port)],
            env=env,
            stdout=log_fh,
            stderr=log_fh,
            preexec_fn=os.setsid,
        )

    if not wait_for_server(base, timeout=30):
        print("ERROR: server did not become healthy within 30 s", file=sys.stderr)
        server.kill()
        sys.exit(1)

    print("Server is healthy.  Pre-seeding sessions with slot IDs 0-4 …")
    session_ids: list[str] = []
    for slot in range(5):
        sid = create_session_with_slot(base, slot)
        if sid:
            session_ids.append(sid)
            print(f"  slot {slot} → {sid}")
        else:
            print(f"  WARNING: could not create session for slot {slot}")

    print(
        f"\nRunning {args.workers} concurrent stress workers for {args.duration}s …"
    )
    results: Counter = Counter()
    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=stress_loop,
            args=(base, session_ids, results, stop_event, i),
            daemon=True,
        )
        for i in range(args.workers)
    ]
    for t in threads:
        t.start()

    deadline = time.time() + args.duration
    alive = True
    while time.time() < deadline:
        time.sleep(1)
        if server.poll() is not None:
            print(f"\nFAILURE: server process exited with code {server.returncode}!")
            alive = False
            break
        elapsed = int(time.time() - (deadline - args.duration))
        code, _ = http_get(f"{base}/health")
        print(f"  [{elapsed:3d}s] health={code}  results={dict(results)}")

    stop_event.set()
    for t in threads:
        t.join(timeout=3)

    if alive:
        print("\nSUCCESS: server stayed alive for the full stress duration.")

    print("\nShutting down server …")
    try:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    server.wait(timeout=5)
    return 0 if alive else 1


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SessionManager eviction stress test")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--duration", type=int, default=30, help="seconds to run")
    parser.add_argument("--workers", type=int, default=8, help="concurrent threads")
    sys.exit(run(parser.parse_args()))
