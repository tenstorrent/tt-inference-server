#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stand-in for `tt-lab gemma --serve-batch` (tests only): the same tagged pipe
protocol as src/gemma.cpp gemma_serve_batch, with a deterministic token sequence
per request instead of a model.

Request with last prompt token p emits 1000 + p % 1000, then +1 each step
(mod 1000, offset 1000), until its limit. Special prompts (last token):
  7  -> three tokens, then the end-of-turn token 106 and STOPPED (-1); with
        ignore_eos (message 3, flag 1) 106 is just a token and decoding goes on.
  13 -> two tokens, then the worker exits (a crashed device worker).
More outstanding requests than slots, an unknown message or an invalid header
is fatal, as in the real worker. One token per active request per step; a
newly submitted request gets its first token in the step after submission.
Arguments (the real command line) are ignored; TT_GEMMA31_SLOTS sets slots and
FAKE_STEP_SECONDS slows each step."""
import os
import select
import struct
import sys
import time

slots = int(os.environ.get("TT_GEMMA31_SLOTS", "4"))
step_seconds = float(os.environ.get("FAKE_STEP_SECONDS", "0"))
out = sys.stdout.buffer
END_TOKENS = (1, 106, 50)


def emit(rid, value):
    out.write(struct.pack("<2i", rid, value))


emit(-3, slots)
out.flush()
buf = b""
active = {}  # id -> dict(token, remaining, special countdown, ignore_eos, crash countdown)
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
        if op not in (1, 3):
            sys.exit(f"invalid message {op}")
        header = 5 if op == 3 else 4
        if len(buf) < 4 * header:
            break
        _, rid, n, limit, *flags = struct.unpack_from(f"<{header}i", buf)
        flags = flags[0] if flags else 0
        if n < 1 or limit < 1 or flags & ~1:
            sys.exit("invalid header")
        if len(buf) < 4 * (header + n):
            break
        prompt = struct.unpack_from(f"<{n}i", buf, 4 * header)
        buf = buf[4 * (header + n):]
        if len(active) >= slots:
            sys.exit("more outstanding requests than slots")
        active[rid] = dict(token=1000 + prompt[-1] % 1000, remaining=limit,
                           end_after=3 if prompt[-1] == 7 else None, ignore_eos=bool(flags & 1),
                           crash_after=2 if prompt[-1] == 13 else None)
    for rid in list(active):
        state = active[rid]
        if state["crash_after"] == 0:
            out.flush()
            os._exit(3)
        if state["end_after"] == 0:
            token = 106
            state["end_after"] = None
        else:
            token = state["token"]
            state["token"] = 1000 + (token + 1) % 1000
            if state["end_after"] is not None:
                state["end_after"] -= 1
        if state["crash_after"] is not None:
            state["crash_after"] -= 1
        emit(rid, token)
        state["remaining"] -= 1
        if token in END_TOKENS and not state["ignore_eos"]:
            emit(rid, -1)
            del active[rid]
        elif state["remaining"] == 0:
            emit(rid, -2)
            del active[rid]
    out.flush()
    if active and step_seconds:
        time.sleep(step_seconds)
