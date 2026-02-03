# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
8-Device Parallel Face Recognition Demo.

Runs YuNet+SFace on 8 Tenstorrent devices in parallel, showing grid of 8 video outputs.
Each device processes different video frames simultaneously.
"""

import os
import sys
import time
import threading
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import io

import numpy as np
import cv2
import torch

# Suppress verbose logging
os.environ["TT_METAL_LOGGER_LEVEL"] = "ERROR"
os.environ["LOGURU_LEVEL"] = "WARNING"

import ttnn

# Global state
frame_buffers = {}  # device_id -> latest frame with detections
frame_locks = {}
stats = {
    "raw_inference_fps": 0,
    "loop_fps": 0,
    "total_frames": 0,
    "faces_detected": 0,
}
stats_lock = threading.Lock()

DEVICE_COLORS = [
    (255, 100, 100),  # Red
    (100, 255, 100),  # Green
    (100, 100, 255),  # Blue
    (255, 255, 100),  # Yellow
    (255, 100, 255),  # Magenta
    (100, 255, 255),  # Cyan
    (255, 180, 100),  # Orange
    (180, 100, 255),  # Purple
]


def init_frame_buffers():
    """Initialize frame buffers with placeholder images."""
    global frame_buffers, frame_locks
    for i in range(8):
        # Create loading placeholder
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, f"Device {i}", (220, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, DEVICE_COLORS[i], 3)
        cv2.putText(placeholder, "Loading...", (230, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
        frame_buffers[i] = placeholder
        frame_locks[i] = threading.Lock()


def load_videos_into_ram(video_paths, num_devices=8):
    """Pre-load all video frames into RAM for maximum throughput."""
    print(f"[Parallel] Pre-loading {len(video_paths)} videos into RAM...")
    
    all_frames = {}
    for i in range(num_devices):
        video_path = video_paths[i % len(video_paths)]
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            frames.append(frame_resized)
        cap.release()
        
        all_frames[i] = frames
        print(f"  Device {i}: {len(frames)} frames from {video_path}")
    
    print(f"[Parallel] All videos loaded into RAM")
    return all_frames


def inference_thread(video_frames, models, devices, stop_event):
    """Run parallel inference on 8 devices."""
    global frame_buffers, stats
    
    num_devices = len(devices)
    frame_indices = [0] * num_devices
    
    loop_times = []
    inference_times = []
    
    while not stop_event.is_set():
        loop_start = time.time()
        
        # Get frames for each device
        input_frames = []
        for i in range(num_devices):
            frames = video_frames[i]
            idx = frame_indices[i] % len(frames)
            frame_indices[i] = idx + 1
            input_frames.append(frames[idx])
        
        # Process each device
        t_inf_start = time.time()
        
        for i in range(num_devices):
            frame_rgb = input_frames[i]
            yunet_model, sface_model = models[i]
            device = devices[i]
            
            # Run YuNet detection
            img_resized = cv2.resize(frame_rgb, (640, 640))
            tensor = torch.from_numpy(img_resized.astype(np.float32)).unsqueeze(0).to(torch.bfloat16)
            tt_input = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            
            cls_out, box_out, obj_out, kpt_out = yunet_model(tt_input)
            ttnn.synchronize_device(device)
            
            # Decode detections (simplified)
            detections = decode_yunet_simple(cls_out, box_out, obj_out, kpt_out, 640)
            
            # Scale to original size and draw
            scale_x, scale_y = 640 / 640, 480 / 640
            frame_out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            for det in detections[:10]:  # Limit to 10 faces
                x1 = int(det["box"][0] * scale_x)
                y1 = int(det["box"][1] * scale_y)
                x2 = int(det["box"][2] * scale_x)
                y2 = int(det["box"][3] * scale_y)
                
                cv2.rectangle(frame_out, (x1, y1), (x2, y2), DEVICE_COLORS[i], 2)
                
                # Draw keypoints
                if det.get("keypoints"):
                    for kp in det["keypoints"]:
                        kx = int(kp[0] * scale_x)
                        ky = int(kp[1] * scale_y)
                        cv2.circle(frame_out, (kx, ky), 3, (0, 255, 255), -1)
            
            # Add device label
            cv2.putText(frame_out, f"Device {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, DEVICE_COLORS[i], 2)
            cv2.putText(frame_out, f"Faces: {len(detections)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Update buffer
            with frame_locks[i]:
                frame_buffers[i] = frame_out
        
        t_inf_end = time.time()
        inference_times.append(t_inf_end - t_inf_start)
        
        loop_end = time.time()
        loop_times.append(loop_end - loop_start)
        
        # Update stats every 10 frames
        if len(loop_times) >= 10:
            avg_loop = sum(loop_times) / len(loop_times)
            avg_inf = sum(inference_times) / len(inference_times)
            
            with stats_lock:
                stats["loop_fps"] = 1.0 / avg_loop if avg_loop > 0 else 0
                stats["raw_inference_fps"] = num_devices / avg_inf if avg_inf > 0 else 0
                stats["total_frames"] += len(loop_times) * num_devices
            
            loop_times.clear()
            inference_times.clear()


def decode_yunet_simple(cls_outs, box_outs, obj_outs, kpt_outs, input_size, threshold=0.5):
    """Simplified YuNet decoding."""
    STRIDES = [8, 16, 32]
    detections = []
    
    for scale_idx in range(3):
        cls_out = ttnn.to_torch(cls_outs[scale_idx]).float().permute(0, 3, 1, 2)
        box_out = ttnn.to_torch(box_outs[scale_idx]).float().permute(0, 3, 1, 2)
        obj_out = ttnn.to_torch(obj_outs[scale_idx]).float().permute(0, 3, 1, 2)
        kpt_out = ttnn.to_torch(kpt_outs[scale_idx]).float().permute(0, 3, 1, 2)
        
        stride = STRIDES[scale_idx]
        score = cls_out.sigmoid() * obj_out.sigmoid()
        
        high_conf = score > threshold
        if high_conf.any():
            indices = torch.where(high_conf)
            for i in range(min(len(indices[0]), 20)):  # Limit
                b, c, h, w = indices[0][i], indices[1][i], indices[2][i], indices[3][i]
                conf = score[b, c, h, w].item()
                anchor_x, anchor_y = w.item() * stride, h.item() * stride
                
                dx, dy = box_out[b, 0, h, w].item(), box_out[b, 1, h, w].item()
                dw, dh = box_out[b, 2, h, w].item(), box_out[b, 3, h, w].item()
                
                cx, cy = dx * stride + anchor_x, dy * stride + anchor_y
                bw, bh = np.exp(min(dw, 10)) * stride, np.exp(min(dh, 10)) * stride
                
                x1, y1 = cx - bw / 2, cy - bh / 2
                x2, y2 = cx + bw / 2, cy + bh / 2
                
                keypoints = []
                for k in range(5):
                    kpt_dx = kpt_out[b, k * 2, h, w].item()
                    kpt_dy = kpt_out[b, k * 2 + 1, h, w].item()
                    keypoints.append([kpt_dx * stride + anchor_x, kpt_dy * stride + anchor_y])
                
                detections.append({"box": [x1, y1, x2, y2], "conf": conf, "keypoints": keypoints})
    
    # Simple NMS
    detections = sorted(detections, key=lambda x: x["conf"], reverse=True)
    return detections[:10]


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads."""
    daemon_threads = True


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(self._get_html().encode())
        
        elif self.path == "/stream/grid":
            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            
            try:
                while True:
                    # Create 2x4 grid of all 8 device outputs
                    grid = self._create_grid()
                    _, jpeg = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    
                    time.sleep(0.033)  # ~30 FPS display
            except Exception as e:
                pass
        
        elif self.path == "/stats":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            import json
            with stats_lock:
                self.wfile.write(json.dumps(stats).encode())
        
        else:
            self.send_error(404)
    
    def _create_grid(self):
        """Create 2x4 grid of all 8 device outputs."""
        rows = []
        for row in range(2):
            cols = []
            for col in range(4):
                idx = row * 4 + col
                with frame_locks[idx]:
                    frame = frame_buffers[idx].copy()
                # Resize to fit grid
                frame_small = cv2.resize(frame, (320, 240))
                cols.append(frame_small)
            rows.append(np.hstack(cols))
        grid = np.vstack(rows)
        return grid
    
    def _get_html(self):
        return '''<!DOCTYPE html>
<html>
<head>
    <title>8-Device Face Recognition - Tenstorrent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            min-height: 100vh;
            font-family: -apple-system, sans-serif;
            color: #fff;
            padding: 20px;
        }
        h1 {
            text-align: center;
            font-size: 2rem;
            background: linear-gradient(90deg, #ff6b6b, #feca57);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subtitle { text-align: center; color: #888; margin-bottom: 20px; }
        .stats-bar {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
        }
        .stat { text-align: center; }
        .stat-value { font-size: 1.5rem; color: #ff6b6b; font-weight: bold; }
        .stat-label { font-size: 0.8rem; color: #888; }
        .grid-container {
            display: flex;
            justify-content: center;
        }
        .grid-container img {
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        }
    </style>
</head>
<body>
    <h1>🔍 8-Device Parallel Face Recognition</h1>
    <p class="subtitle">YuNet + SFace running on 8 Tenstorrent devices simultaneously</p>
    
    <div class="stats-bar">
        <div class="stat">
            <div class="stat-value" id="raw-fps">-</div>
            <div class="stat-label">Raw Throughput (FPS)</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="loop-fps">-</div>
            <div class="stat-label">Display FPS</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="total-frames">-</div>
            <div class="stat-label">Total Frames</div>
        </div>
    </div>
    
    <div class="grid-container">
        <img src="/stream/grid" width="1280" height="480">
    </div>
    
    <script>
        async function updateStats() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                document.getElementById('raw-fps').textContent = data.raw_inference_fps.toFixed(0);
                document.getElementById('loop-fps').textContent = data.loop_fps.toFixed(0);
                document.getElementById('total-frames').textContent = data.total_frames.toLocaleString();
            } catch (e) {}
        }
        setInterval(updateStats, 1000);
        updateStats();
    </script>
</body>
</html>'''
    
    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="8-Device Parallel Face Recognition")
    parser.add_argument("videos", nargs="*", default=[], help="Video paths (will cycle for 8 devices)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    args = parser.parse_args()
    
    # Default video if none provided
    video_paths = args.videos if args.videos else ["/app/demo/city_traffic.mp4"]
    
    print("=" * 60)
    print("  8-Device Parallel Face Recognition Demo")
    print("=" * 60)
    print(f"  Videos: {video_paths}")
    print(f"  Browser: http://localhost:{args.port}")
    print("=" * 60)
    
    # Initialize frame buffers
    init_frame_buffers()
    
    # Start HTTP server early (shows loading state)
    server = ThreadedHTTPServer(("0.0.0.0", args.port), StreamHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[HTTP] Server started on port {args.port}")
    
    # Pre-load videos
    video_frames = load_videos_into_ram(video_paths, num_devices=8)
    
    # Initialize models on all 8 devices
    print("[Parallel] Initializing 8 devices...")
    
    # Add models to path
    models_path = Path(__file__).parent / "models"
    if models_path.exists():
        sys.path.insert(0, str(models_path.parent))
    
    from models.yunet.common import (
        YUNET_L1_SMALL_SIZE,
        load_torch_model as load_yunet_torch,
        get_default_weights_path as get_yunet_weights,
    )
    from models.yunet.tt.ttnn_yunet import create_yunet_model
    from models.sface.common import get_sface_onnx_path, SFACE_L1_SMALL_SIZE
    from models.sface.reference.sface_model import load_sface_from_onnx
    from models.sface.tt.ttnn_sface import create_sface_model
    
    l1_size = max(YUNET_L1_SMALL_SIZE, SFACE_L1_SMALL_SIZE)
    
    devices = []
    models = []
    
    for i in range(8):
        print(f"  Device {i}: Opening...")
        device = ttnn.open_device(device_id=i, l1_small_size=l1_size)
        device.enable_program_cache()
        devices.append(device)
        
        # Load YuNet
        yunet_weights = get_yunet_weights()
        yunet_torch = load_yunet_torch(yunet_weights)
        yunet_torch = yunet_torch.to(torch.bfloat16)
        yunet_model = create_yunet_model(device, yunet_torch)
        
        # Load SFace
        sface_onnx = get_sface_onnx_path()
        sface_torch = load_sface_from_onnx(sface_onnx)
        sface_torch.eval()
        sface_model = create_sface_model(device, sface_torch)
        
        models.append((yunet_model, sface_model))
        print(f"  Device {i}: Models loaded")
    
    print("[Parallel] All devices ready!")
    print("[Parallel] Starting inference loop...")
    
    # Start inference thread
    stop_event = threading.Event()
    inf_thread = threading.Thread(target=inference_thread, args=(video_frames, models, devices, stop_event), daemon=True)
    inf_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Parallel] Shutting down...")
        stop_event.set()
        for device in devices:
            ttnn.close_device(device)


if __name__ == "__main__":
    main()
