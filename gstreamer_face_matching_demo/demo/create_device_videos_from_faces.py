#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Create two demo videos from face photos for the 4-device parallel demo:
  - video_device1.mp4: Mohamed, Jim, Jonathan (from registered_faces)
  - video_device2.mp4: attached photos (device2_photo1.png, device2_photo2.png) + Teja, Jim
No text overlays; output is 640x480 @ 30fps for pipeline compatibility.
"""
import sys
from pathlib import Path

FPS = 30
# Shorter per-image so video feels like original (smoother, more frames)
FRAMES_PER_IMAGE = 25   # ~0.8 sec per image at 30fps
OUTPUT_SIZE = (640, 480)  # match pipeline
# Match original video length: ~10 sec = 300 frames at 30fps
MIN_FRAMES = 300


def video_from_images(image_paths, out_path, min_frames=None):
    """Write a video from a list of image paths; loop until at least min_frames."""
    if min_frames is None:
        min_frames = MIN_FRAMES
    try:
        import cv2
    except ImportError:
        print("[create_device_videos] opencv not available", flush=True)
        return False
    if not image_paths:
        return False
    valid = []
    for p in image_paths:
        p = Path(p)
        if not p.exists():
            print(f"[create_device_videos] Skip missing: {p}", flush=True)
            continue
        valid.append(p)
    if not valid:
        return False
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, FPS, OUTPUT_SIZE)
    if not out.isOpened():
        return False
    frames_written = 0
    while frames_written < min_frames:
        for img_path in valid:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, OUTPUT_SIZE)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for _ in range(FRAMES_PER_IMAGE):
                out.write(frame_bgr)
                frames_written += 1
                if frames_written >= min_frames:
                    break
            if frames_written >= min_frames:
                break
    out.release()
    print(f"[create_device_videos] Wrote {frames_written} frames -> {out_path}", flush=True)
    return True


def main():
    demo_dir = Path(__file__).resolve().parent
    out1 = demo_dir / "video_device1.mp4"
    out2 = demo_dir / "video_device2.mp4"
    if out1.exists() and out2.exists():
        print("[create_device_videos] video_device1.mp4 and video_device2.mp4 already exist", flush=True)
        return 0

    # In container: /app/registered_faces; on host: same repo demo/../registered_faces
    if demo_dir.name == "demo":
        faces_dir = demo_dir.parent / "registered_faces"
    else:
        faces_dir = Path("/app/registered_faces")
    if not faces_dir.exists():
        faces_dir = demo_dir.parent / "registered_faces"

    # Device 1: Mohamed, Jim, Jonathan
    device1_images = []
    for name in ("Mohamed", "Jim", "Jonathan Su"):
        d = faces_dir / name
        if d.is_dir():
            for f in ("face.jpg", "face_2.jpg", "face_3.jpg"):
                if (d / f).exists():
                    device1_images.append(d / f)
                    break
    ok1 = video_from_images(device1_images, out1)

    # Device 2: attached photos + Teja, Jim
    device2_images = []
    for name in ("device2_photo1.png", "device2_photo2.png"):
        p = demo_dir / name
        if p.exists():
            device2_images.append(p)
    for name in ("Teja", "Jim"):
        d = faces_dir / name
        if d.is_dir() and (d / "face.jpg").exists():
            device2_images.append(d / "face.jpg")
    ok2 = video_from_images(device2_images, out2)

    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
