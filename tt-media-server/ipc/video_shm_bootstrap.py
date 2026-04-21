#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Operator CLI for the video SHM segments.

Under the current ownership model (see :mod:`ipc.video_shm`) the server and
runner processes both perform a *create-or-attach* open and never unlink the
segments themselves. That keeps ring-buffer state alive across restarts of
either side — but it also means the segments need an explicit operator action
to reset.

Subcommands
-----------
``up``      Ensure the input and output ring segments (and their ``<name>_state``
            siblings) exist. Idempotent — a no-op if they already exist.
``down``    Unlink all four segments and delete any leftover video result files
            from ``/dev/shm``. Destructive: only run when both the server and
            the runner are stopped, or when you explicitly want to drop
            in-flight ring state.
``status``  Print segment sizes, writer/reader indices and queue depth for the
            input and output rings. Read-only.

Environment variables
---------------------
``TT_VIDEO_SHM_INPUT``   Name of the input  ring segment (default ``tt_video_in``).
``TT_VIDEO_SHM_OUTPUT``  Name of the output ring segment (default ``tt_video_out``).

Examples
--------
    python -m ipc.video_shm_bootstrap up
    python -m ipc.video_shm_bootstrap status
    python -m ipc.video_shm_bootstrap down
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import shared_memory as _shm

from ipc.video_shm import VideoShm, cleanup_orphaned_video_files

DEFAULT_INPUT_NAME = "tt_video_in"
DEFAULT_OUTPUT_NAME = "tt_video_out"


def _names() -> tuple[str, str]:
    in_name = os.environ.get("TT_VIDEO_SHM_INPUT", DEFAULT_INPUT_NAME)
    out_name = os.environ.get("TT_VIDEO_SHM_OUTPUT", DEFAULT_OUTPUT_NAME)
    return in_name, out_name


def _exists(name: str) -> bool:
    return os.path.exists(f"/dev/shm/{name}")


def up(in_name: str, out_name: str) -> int:
    """Create both ring segments (idempotent)."""
    for shm_name, mode in ((in_name, "input"), (out_name, "output")):
        already = _exists(shm_name)
        shm = VideoShm(shm_name, mode=mode)
        shm.open()
        shm.close()
        status = "attached (already existed)" if already else "created"
        print(f"[bootstrap] {shm_name:20s}  {status}")
        state_name = f"{shm_name}{VideoShm._STATE_SUFFIX}"
        print(f"[bootstrap]   └─ {state_name:18s}  state segment ready")
    return 0


def down(in_name: str, out_name: str) -> int:
    """Unlink both ring segments + their state segments; delete result files."""
    segments_removed = 0
    for shm_name in (in_name, out_name):
        for name in (shm_name, f"{shm_name}{VideoShm._STATE_SUFFIX}"):
            if _unlink_segment(name):
                print(f"[bootstrap] unlinked  {name}")
                segments_removed += 1
            else:
                print(f"[bootstrap] missing   {name} (nothing to do)")

    files_removed = cleanup_orphaned_video_files()
    if files_removed:
        print(f"[bootstrap] removed {files_removed} orphaned video file(s)")
    print(
        f"[bootstrap] done — {segments_removed} SHM segment(s), "
        f"{files_removed} video file(s) cleaned up"
    )
    return 0


def status(in_name: str, out_name: str) -> int:
    """Print sizes, indices and queue depth for each ring."""
    print(
        f"{'segment':24s}  {'size':>10s}  {'writer':>10s}  {'reader':>10s}  {'depth':>6s}"
    )
    print("-" * 72)
    any_present = False
    for shm_name, mode in ((in_name, "input"), (out_name, "output")):
        if not _exists(shm_name):
            print(f"{shm_name:24s}  (not created)")
            continue
        any_present = True
        shm = VideoShm(shm_name, mode=mode)
        shm.open()
        try:
            size = shm._shm.size if shm._shm else 0
            widx = shm._get_writer_index()
            ridx = shm._get_reader_index()
            depth = shm.queue_depth()
            print(f"{shm_name:24s}  {size:>10d}  {widx:>10d}  {ridx:>10d}  {depth:>6d}")
        finally:
            shm.close()
    if not any_present:
        print("\nNo segments exist. Run `python -m ipc.video_shm_bootstrap up`.")
    return 0


def _unlink_segment(name: str) -> bool:
    """Unlink a single POSIX SHM segment by name. Returns True if it was removed.

    ``SharedMemory.unlink`` already unregisters from resource_tracker internally,
    so we must not double-unregister — doing so raises a KeyError inside the
    tracker daemon at process exit.
    """
    if not _exists(name):
        return False
    try:
        shm = _shm.SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return False
    try:
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ipc.video_shm_bootstrap",
        description="Create, destroy or inspect the video SHM segments.",
    )
    parser.add_argument(
        "action",
        choices=("up", "down", "status"),
        help="up: create segments; down: unlink segments; status: show indices",
    )
    parser.add_argument(
        "--input-name",
        default=None,
        help=f"Input ring name (env TT_VIDEO_SHM_INPUT, default {DEFAULT_INPUT_NAME})",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=f"Output ring name (env TT_VIDEO_SHM_OUTPUT, default {DEFAULT_OUTPUT_NAME})",
    )
    args = parser.parse_args(argv)

    env_in, env_out = _names()
    in_name = args.input_name or env_in
    out_name = args.output_name or env_out

    dispatch = {"up": up, "down": down, "status": status}
    return dispatch[args.action](in_name, out_name)


if __name__ == "__main__":
    sys.exit(main())
