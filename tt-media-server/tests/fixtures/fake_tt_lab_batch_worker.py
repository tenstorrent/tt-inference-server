#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stand-in for `tt-lab gemma --serve-batch` (tests only): the same tagged pipe
protocol, with a deterministic token sequence per request. Request r with last
prompt token p emits p+1, p+2, ... (mod 1000, offset 1000) until its limit; a
prompt ending in token 7 stops after 3 tokens with the end-of-turn token 106.
Slots are enforced like the real worker: one more outstanding request than
FAKE_SLOTS is a protocol error. One token per active request per step."""
import os
import select
import struct
import sys

slots = int(os.environ.get("TT_GEMMA31_SLOTS", "4"))
out = sys.stdout.buffer


def emit(rid, value):
    out.write(struct.pack("<2i", rid, value))


emit(-3, slots)
out.flush()
buf = b""
active = {}  # id -> [next token, remaining, stop_after]
while True:
    readable = select.select([0], [], [], 0 if active else None)[0]
    if readable:
        chunk = os.read(0, 65536)
        if not chunk:
            sys.exit(0)
        buf += chunk
    while len(buf) >= 8:
        op, rid = struct.unpack_from("<2i", buf)
        if op == 2:
            buf = buf[8:]
            active.pop(rid, None)
            emit(rid, -4)
            continue
        if len(buf) < 16:
            break
        _, rid, n, limit = struct.unpack_from("<4i", buf)
        if len(buf) < 16 + 4 * n:
            break
        prompt = struct.unpack_from(f"<{n}i", buf, 16)
        buf = buf[16 + 4 * n:]
        if len(active) >= slots:
            sys.exit("more outstanding requests than slots")
        active[rid] = [1000 + prompt[-1] % 1000, limit, 3 if prompt[-1] == 7 else None]
    for rid in list(active):
        token, remaining, stop_after = active[rid]
        if stop_after == 0:
            emit(rid, 106)
            emit(rid, -1)
            del active[rid]
            continue
        emit(rid, token)
        remaining -= 1
        if remaining == 0:
            emit(rid, -2)
            del active[rid]
        else:
            active[rid] = [1000 + (token + 1) % 1000, remaining, None if stop_after is None else stop_after - 1]
    out.flush()
