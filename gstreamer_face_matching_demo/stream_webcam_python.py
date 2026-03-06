#!/usr/bin/env python3
"""
USB webcam stream with YuNet+SFace face recognition (Python fallback when GStreamer plugin is unavailable).
Captures from /dev/video0, runs inference, streams MJPEG to TCP 8081 for the HTTP server.
"""
import os
import sys
import socket
import argparse
import threading
import time
import cv2
import numpy as np

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("LOGURU_LEVEL", "WARNING")

# Add /app so we can import websocket_server
if "/app" not in sys.path:
    sys.path.insert(0, "/app")

from websocket_server import (
    init_models,
    load_faces_from_disk,
    process_pending_faces,
    process_frame_with_timing,
)

TCP_HOST = "0.0.0.0"
TCP_PORT = 8081
BOUNDARY = b"frame"
JPEG_QUALITY = 85


def draw_results(frame_bgr, results, timing):
    """Draw face boxes, identities, and timing overlay on frame."""
    for r in results:
        x1, y1 = int(r["x1"]), int(r["y1"])
        x2, y2 = int(r["x2"]), int(r["y2"])
        identity = r["identity"]
        score = r["score"]
        is_recognized = identity != "Unknown"
        color = (0, 255, 0) if is_recognized else (0, 0, 255)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        if is_recognized:
            label = f"{identity}: {int(score*100)}%"
            cv2.putText(frame_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if r.get("keypoints"):
            for kp in r["keypoints"]:
                cv2.circle(frame_bgr, (int(kp[0]), int(kp[1])), 2, (0, 255, 255), -1)

    total_ms = timing.get("total", 0)
    fps = 1000 / total_ms if total_ms > 0 else 0
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (5, 5), (200, 55), (0, 0, 0), -1)
    frame_bgr = cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0)
    cv2.putText(frame_bgr, f"Inference: {total_ms:.1f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame_bgr, f"FPS: {fps:.0f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame_bgr


def main():
    parser = argparse.ArgumentParser(description="USB webcam face recognition stream (Python)")
    parser.add_argument("device", nargs="?", default="/dev/video0", help="Video device (default: /dev/video0)")
    parser.add_argument("--port", type=int, default=TCP_PORT, help="TCP port for MJPEG stream")
    args = parser.parse_args()

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    def make_placeholder_frame(msg, subtext=""):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        cv2.putText(img, msg, (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        if subtext:
            cv2.putText(img, subtext, (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        return img

    def send_jpeg(conn, img):
        _, jpeg = cv2.imencode(".jpg", img, encode_params)
        conn.sendall(b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

    # Listen FIRST so Stream tab never gets "connection refused" (models load later)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((TCP_HOST, args.port))
    server.listen(1)
    print(f"[StreamWebcam] Listening on {TCP_HOST}:{args.port} — open Stream tab now", flush=True)

    while True:
        conn, addr = server.accept()
        print(f"[StreamWebcam] Client connected from {addr}", flush=True)
        try:
            # Load models in background; send "Loading..." frames so connection stays alive
            init_done = threading.Event()
            init_error = [None]  # list so inner function can set

            def do_init():
                try:
                    init_models()
                    load_faces_from_disk()
                    process_pending_faces()
                except Exception as e:
                    init_error[0] = e
                finally:
                    init_done.set()

            print("[StreamWebcam] Loading models (one-time)...", flush=True)
            t = threading.Thread(target=do_init, daemon=True)
            t.start()
            while not init_done.wait(timeout=0.5):
                send_jpeg(conn, make_placeholder_frame("Loading models...", "YuNet + SFace (this may take 1–2 min)"))
            if init_error[0]:
                print(f"[StreamWebcam] Init failed: {init_error[0]}", flush=True)
                send_jpeg(conn, make_placeholder_frame("Model load failed", str(init_error[0])))
                conn.close()
                continue
            print("[StreamWebcam] Models ready. Opening camera...", flush=True)
        except (BrokenPipeError, ConnectionResetError):
            conn.close()
            continue

        cap = cv2.VideoCapture(args.device)
        if not cap.isOpened():
            print(f"[StreamWebcam] WARNING: Could not open {args.device} - sending placeholder", flush=True)
            try:
                err_img = make_placeholder_frame(f"Camera unavailable: {args.device}", "Use Webcam tab or run with --group-add $(getent group video | cut -d: -f3)")
                _, jpeg = cv2.imencode(".jpg", err_img, encode_params)
                msg = b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                for _ in range(60):
                    conn.sendall(msg)
                    time.sleep(0.5)
            except Exception:
                pass
            conn.close()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results, timing = process_frame_with_timing(frame_rgb)
                frame_bgr = draw_results(frame_bgr, results, timing)
                _, jpeg = cv2.imencode(".jpg", frame_bgr, encode_params)
                jpeg_bytes = jpeg.tobytes()
                msg = b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                try:
                    conn.sendall(msg)
                except (BrokenPipeError, ConnectionResetError):
                    break
        except Exception as e:
            print(f"[StreamWebcam] Error: {e}", flush=True)
        finally:
            cap.release()
            conn.close()
        print("[StreamWebcam] Client disconnected", flush=True)


if __name__ == "__main__":
    main()
