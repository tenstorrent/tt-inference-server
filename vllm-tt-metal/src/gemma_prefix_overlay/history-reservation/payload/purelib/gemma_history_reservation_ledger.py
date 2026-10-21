# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Isolated CPU prototype. No scheduler/model installer calls this module yet."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Row:
    key: tuple[str, int]  # Stable request ID plus cleanup/reuse generation.
    start: int
    end: int
    retain: bool


def project_serial(live, rows, cleanup=(), retained_limit=2):
    """Simulate exact cap1 row order before submitting any model work.

    `live` maps current owner to saved continuation endpoint. Cleanup must be
    delivered before acquisition in this SAME worker submission. This computes
    host ownership, not device completion or scheduler KV allocation.
    """
    state = dict(live)
    if len(state) > retained_limit:
        raise RuntimeError(f"Existing retained owners exceed limit: {state}")
    for key in cleanup:
        state.pop(key, None)
    peak = len(state)
    for row in rows:
        if type(row.start) is not int or type(row.end) is not int or not 0 <= row.start < row.end:
            raise ValueError(f"Invalid advancing row: {row}")
        if type(row.retain) is not bool:
            raise ValueError(f"Missing exact finality: {row}")
        if row.start == 0:
            state.pop(row.key, None)
        elif row.key in state:
            if state[row.key] != row.start:
                raise RuntimeError(f"Continuation mismatch: {row}, saved={state[row.key]}")
        elif row.start % 128:
            raise RuntimeError(f"Absent history at non-page cached start: {row}")
        state[row.key] = row.end
        peak = max(peak, len(state))
        if peak > retained_limit + 1:
            raise RuntimeError(f"Temporary bank exhausted before row: {row}")
        if not row.retain:
            del state[row.key]
        if len(state) > retained_limit:
            raise RuntimeError(f"Retained bound exceeded after row: {row}")
    return state, peak


class ReservationLedger:
    """Tentative pre-allocation decisions plus explicit successful-submit commit.

    Future integration must attest UniProc/FIFO/cap1, obtain real final rows,
    and call submitted only AFTER synchronous execute_model returns. There is
    deliberately no token-output release API. Failure poisons further use.
    """
    def __init__(self, limit=2):
        if type(limit) is not int or limit < 1:
            raise ValueError(f"Invalid retained limit: {limit}")
        self.limit = limit
        self.live = {}
        self.tentative = None
        self.pending = None
        self.poisoned = False

    def begin(self):
        if self.poisoned or self.tentative is not None or self.pending is not None:
            raise RuntimeError("Overlapping, failed or uncommitted host submission")
        self.tentative = set(self.live)

    def consider(self, key, start, count, prompt_end):
        if self.tentative is None or self.pending is not None:
            raise RuntimeError("No open pre-allocation selection")
        if not isinstance(key, tuple) or len(key) != 2 or type(key[1]) is not int or key[1] < 0:
            raise ValueError(f"Generation-tagged identity required: {key}")
        if count <= 0:
            return count
        if start + count < prompt_end:
            if key not in self.tentative and len(self.tentative) >= self.limit:
                return 0
            self.tentative.add(key)
        # Never release a prospective final here: its allocation can still fail.
        return count

    def prepare(self, rows, cleanup=()):
        if self.tentative is None or self.pending is not None:
            raise RuntimeError("No open selection or duplicate prepare")
        rows = tuple(rows)
        for row in rows:
            if row.retain and row.key not in self.tentative:
                raise RuntimeError(f"Final scheduled partial has no reservation: {row}")
        state, peak = project_serial(self.live, rows, cleanup, self.limit)
        self.pending = state
        self.tentative = None
        return peak

    def submitted(self):
        if self.poisoned or self.pending is None:
            raise RuntimeError("No successful prepared submission")
        self.live = self.pending
        self.pending = None

    def failed(self):
        self.poisoned = True
