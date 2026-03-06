#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Create a second demo video (test_video_b.mp4) from test_video.mp4 with a visible
"Source B" overlay. Used so devices 1 and 3 can show a different source than 0 and 2.
"""
import sys
from pathlib import Path

def main():
    # Use script directory so it works on host (demo/) and in container (/app/demo/)
    demo_dir = Path(__file__).resolve().parent
    src = demo_dir / "test_video.mp4"
    dst = demo_dir / "test_video_b.mp4"
    if not src.exists():
        print(f"[create_second_demo_video] Source not found: {src}", flush=True)
        return 1
    if dst.exists():
        print(f"[create_second_demo_video] Already exists: {dst}", flush=True)
        return 0

    try:
        import cv2
    except ImportError:
        print("[create_second_demo_video] opencv not available", flush=True)
        return 1

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[create_second_demo_video] Could not open: {src}", flush=True)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    if not out.isOpened():
        print("[create_second_demo_video] Could not create writer", flush=True)
        cap.release()
        return 1

    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Green banner at top + "Source B" text so devices 1 & 3 look different
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 200, 0), -1)
        cv2.putText(
            frame, "Source B (Device 1 & 3)",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )
        out.write(frame)
        n += 1
    cap.release()
    out.release()
    print(f"[create_second_demo_video] Wrote {n} frames to {dst}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
