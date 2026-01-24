#!/usr/bin/env python3
"""
8-Device Parallel Video Stream Demo

Takes a single video input and runs inference on 8 TT devices simultaneously,
displaying 8 output windows in a grid UI.

Usage:
    python3 parallel_8device_stream.py [video_path]
"""
import sys
import time
import threading
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads for multiple streams."""
    daemon_threads = True
import cv2
import numpy as np
import torch

# TT imports
import ttnn
from models.demos.yolov8s.runner.performant_runner import YOLOv8sPerformantRunner
from models.demos.yolov8s.common import YOLOV8S_L1_SMALL_SIZE
from models.demos.utils.common_demo_utils import (
    get_mesh_mappers,
    load_coco_class_names,
    postprocess,
)

# Global state for streaming
frame_buffers = [None] * 8  # One buffer per device output
frame_locks = [threading.Lock() for _ in range(8)]
stats = {"fps": 0, "latency_ms": 0, "total_frames": 0, "throughput_fps": 0}
running = True

# Colors for each device (to visually distinguish outputs)
DEVICE_COLORS = [
    (66, 133, 244),   # Blue
    (52, 168, 83),    # Green
    (234, 67, 53),    # Red
    (251, 188, 5),    # Yellow
    (155, 89, 182),   # Purple
    (26, 188, 156),   # Teal
    (241, 90, 34),    # Orange
    (149, 165, 166),  # Gray
]

CLASS_COLORS = {}  # Will be populated with consistent colors per class


def init_frame_buffers():
    """Initialize frame buffers with placeholder images."""
    for i in range(8):
        placeholder = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, f"Device {i}", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, DEVICE_COLORS[i], 3)
        cv2.putText(placeholder, "Loading...", (220, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, jpeg = cv2.imencode('.jpg', placeholder)
        frame_buffers[i] = jpeg.tobytes()


def get_class_color(class_id):
    """Get consistent color for a class."""
    if class_id not in CLASS_COLORS:
        np.random.seed(class_id)
        CLASS_COLORS[class_id] = (
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)),
        )
    return CLASS_COLORS[class_id]


def draw_detections(image, result, names, device_id, conf_threshold=0.5):
    """Draw bounding boxes and device label on image."""
    boxes = result["boxes"]["xyxy"]
    scores = result["boxes"]["conf"]
    classes = result["boxes"]["cls"]
    
    # Draw detections
    for box, score, cls in zip(boxes, scores, classes):
        conf = float(score) if isinstance(score, (int, float)) else float(score.item()) if hasattr(score, 'item') else float(score)
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = [int(c.item()) if hasattr(c, 'item') else int(c) for c in box]
        class_id = int(cls.item()) if hasattr(cls, 'item') else int(cls)
        class_name = names[class_id] if class_id < len(names) else f"class_{class_id}"
        
        color = get_class_color(class_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw device label with colored border
    dev_color = DEVICE_COLORS[device_id % len(DEVICE_COLORS)]
    cv2.rectangle(image, (0, 0), (image.shape[1]-1, image.shape[0]-1), dev_color, 4)
    
    label = f"Device {device_id}"
    cv2.rectangle(image, (5, 5), (110, 30), dev_color, -1)
    cv2.putText(image, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image


def preprocess_batch(frame, batch_size):
    """Preprocess single frame into batch tensor."""
    # Resize to 640x640
    resized = cv2.resize(frame, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize and convert to tensor
    tensor = torch.from_numpy(rgb).float().div(255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
    
    # Replicate to batch
    batch = tensor.repeat(batch_size, 1, 1, 1)
    return batch, resized


def inference_thread(video_paths, runner, mesh_device, output_mesh_composer, names, num_devices):
    """Main inference loop - 8 videos on 8 devices at FULL SPEED."""
    global frame_buffers, stats, running
    
    # Pre-load frames from all videos into memory for max speed
    print(f"\n📥 Pre-loading frames from {num_devices} videos...")
    all_video_frames = []  # List of lists: [video_idx][frame_idx]
    
    for i, path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Could not open video {path}")
            running = False
            return
        
        frames = []
        while len(frames) < 300:  # Load up to 300 frames per video
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (640, 640))
            frames.append(resized)
        cap.release()
        
        if not frames:
            print(f"Error: No frames loaded from {path}")
            running = False
            return
            
        all_video_frames.append(frames)
        print(f"   Device {i}: {path} ({len(frames)} frames)", flush=True)
    
    print(f"\n🎬 Starting 8-device parallel inference...", flush=True)
    print(f"   Mode: Pre-loaded frames, showing RAW inference speed", flush=True)
    
    frame_indices = [0] * num_devices  # Current frame index for each video
    batch_count = 0
    fps_start = time.time()
    batches_in_window = 0
    inference_only_batches = 0
    inference_only_start = time.time()
    display_update_interval = 0.033  # Update display at ~30 FPS
    last_display_update = time.time()
    
    while running:
        try:
            t_start = time.time()
            
            # Get next frame from each video (cycling through pre-loaded frames)
            frames = []
            tensors = []
            for i in range(num_devices):
                video_frames = all_video_frames[i]
                frame = video_frames[frame_indices[i] % len(video_frames)]
                frame_indices[i] += 1
                frames.append(frame)
                
                # Preprocess to tensor
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).float().div(255.0)
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
                tensors.append(tensor)
            
            # Stack into batch [8, 3, 640, 640]
            batch_tensor = torch.cat(tensors, dim=0)
            
            # Run inference on all 8 devices in parallel - THE FAST PART
            t_inf_start = time.time()
            preds = runner.run(batch_tensor)
            ttnn.synchronize_device(mesh_device)
            t_inf_end = time.time()
            
            # Pure inference time (just runner.run + sync)
            pure_inference_ms = (t_inf_end - t_inf_start) * 1000
            inference_time_ms = (t_inf_end - t_start) * 1000  # Total loop time
            batch_count += 1
            batches_in_window += 1
            inference_only_batches += 1
            
            # Calculate RAW inference FPS from pure inference time only
            # Pure inference = just runner.run() + sync, no preprocessing
            if pure_inference_ms > 0:
                stats["raw_inference_fps"] = (num_devices * 1000) / pure_inference_ms
                stats["pure_inference_ms"] = pure_inference_ms
            
            # Only update display at ~30 FPS (not every inference)
            now = time.time()
            if now - last_display_update >= display_update_interval:
                last_display_update = now
                
                # Convert predictions for display
                preds_torch = ttnn.to_torch(preds[0], dtype=torch.float32, mesh_composer=output_mesh_composer)
                
                # Update each device's output
                for i in range(num_devices):
                    output_frame = frames[i].copy()
                    
                    # Get detections for this device
                    try:
                        single_pred = preds_torch[i:i+1]
                        single_tensor = batch_tensor[i:i+1]
                        result = postprocess(single_pred, single_tensor, [output_frame], [[str(i)]], names)
                        if result and len(result) > 0:
                            output_frame = draw_detections(output_frame, result[0], names, i)
                    except Exception:
                        pass
                    
                    # Add device label only (stats shown in header)
                    cv2.rectangle(output_frame, (5, 5), (100, 30), (0, 0, 0), -1)
                    cv2.putText(output_frame, f"Device {i}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DEVICE_COLORS[i], 2)
                    
                    # Encode to JPEG
                    _, jpeg = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    with frame_locks[i]:
                        frame_buffers[i] = jpeg.tobytes()
            
            # Update overall stats every second
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                throughput = (batches_in_window * num_devices) / elapsed
                stats["throughput_fps"] = throughput
                stats["fps"] = batches_in_window / elapsed
                stats["latency_ms"] = inference_time_ms
                stats["total_frames"] = batch_count * num_devices
                
                raw_fps = stats.get('raw_inference_fps', 0)
                pure_ms = stats.get('pure_inference_ms', 0)
                print(f"⚡ RAW: {raw_fps:.0f} FPS ({pure_ms:.1f}ms inference) | Loop: {inference_time_ms:.1f}ms | Total: {stats['total_frames']}", flush=True)
                
                batches_in_window = 0
                fps_start = time.time()
        
        except Exception as e:
            print(f"❌ ERROR in inference loop: {e}", flush=True)
            import traceback
            traceback.print_exc()
            break
    
    print("\n🛑 Inference thread stopped", flush=True)


class StreamHandler(BaseHTTPRequestHandler):
    """HTTP handler for MJPEG streams."""
    
    def log_message(self, format, *args):
        pass  # Suppress logs
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(self.get_html().encode())
        
        elif self.path == '/stream/grid':
            # Combined grid stream - single connection for all 8 devices
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            while running:
                try:
                    # Decode all 8 frames
                    frames = []
                    for i in range(8):
                        with frame_locks[i]:
                            data = frame_buffers[i]
                        if data:
                            nparr = np.frombuffer(data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                # Resize each to 320x320 for grid
                                frame = cv2.resize(frame, (320, 320))
                                frames.append(frame)
                            else:
                                frames.append(np.zeros((320, 320, 3), dtype=np.uint8))
                        else:
                            frames.append(np.zeros((320, 320, 3), dtype=np.uint8))
                    
                    # Create 4x2 grid (1280x640)
                    row1 = np.hstack(frames[0:4])
                    row2 = np.hstack(frames[4:8])
                    grid = np.vstack([row1, row2])
                    
                    # Encode and send
                    _, jpeg = cv2.imencode('.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
                except:
                    break
                
                time.sleep(0.033)  # ~30fps
        
        elif self.path == '/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            import json
            self.wfile.write(json.dumps(stats).encode())
        
        else:
            self.send_error(404)
    
    def get_html(self):
        return '''<!DOCTYPE html>
<html>
<head>
    <title>8-Device Parallel YOLOv8s Demo</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 100%);
            font-family: 'Segoe UI', system-ui, sans-serif;
            color: #fff;
            min-height: 100vh;
        }
        .header {
            background: rgba(0,0,0,0.3);
            padding: 15px 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .header h1 {
            font-size: 1.5rem;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stats {
            display: flex;
            gap: 30px;
            font-size: 1rem;
        }
        .stat { display: flex; align-items: center; gap: 8px; }
        .stat-label { color: #888; }
        .stat-value { color: #00ff88; font-weight: bold; font-size: 1.2rem; }
        .stat-value.highlight { color: #ff0; font-size: 1.8rem; }
        .grid-container {
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            height: calc(100vh - 80px);
        }
        .grid-container img {
            max-width: 100%;
            max-height: 100%;
            border-radius: 12px;
            border: 3px solid rgba(255,255,255,0.2);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚡ 8-Device Data Parallel YOLOv8s</h1>
        <div class="stats">
            <div class="stat">
                <span class="stat-label">🚀 Raw Inference:</span>
                <span class="stat-value highlight" id="raw-fps">-</span>
                <span class="stat-label">FPS</span>
            </div>
            <div class="stat">
                <span class="stat-label">Batch Latency:</span>
                <span class="stat-value" id="latency">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Total:</span>
                <span class="stat-value" id="frames">-</span>
            </div>
        </div>
    </div>
    <div class="grid-container">
        <img src="/stream/grid" alt="8-Device Grid">
    </div>
    <script>
        setInterval(async () => {
            try {
                const res = await fetch('/stats');
                const data = await res.json();
                document.getElementById('raw-fps').textContent = Math.round(data.raw_inference_fps || 0);
                document.getElementById('latency').textContent = data.latency_ms.toFixed(1) + ' ms';
                document.getElementById('frames').textContent = data.total_frames.toLocaleString();
            } catch(e) {}
        }, 500);
    </script>
</body>
</html>'''


def main():
    global running
    
    # Initialize placeholder frames for all devices
    init_frame_buffers()
    
    parser = argparse.ArgumentParser(description="8-Device Parallel Video Stream")
    parser.add_argument("videos", nargs='*', default=["/app/demo/city_traffic.mp4"], 
                       help="Video file paths (1-8 videos). If less than 8, videos are repeated.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  8-Device Parallel YOLOv8s Streaming Demo")
    print("=" * 70)
    
    # Get devices
    device_ids = ttnn.get_device_ids()
    num_devices = len(device_ids)
    print(f"\n📟 Found {num_devices} TT devices")
    
    if num_devices < 8:
        print(f"⚠️  Only {num_devices} devices found. Using all available devices.")
    
    # Prepare video paths - repeat if necessary to fill all devices
    input_videos = args.videos if args.videos else ["/app/demo/city_traffic.mp4"]
    video_paths = []
    for i in range(num_devices):
        video_paths.append(input_videos[i % len(input_videos)])
    
    print(f"\n🎥 Video assignments:")
    for i, path in enumerate(video_paths):
        print(f"   Device {i}: {path}")
    
    # Create mesh device
    mesh_shape = ttnn.MeshShape(1, num_devices)
    device_params = {
        "l1_small_size": YOLOV8S_L1_SMALL_SIZE,
        "trace_region_size": 6434816,
        "num_command_queues": 2,
    }
    
    print(f"\n🔧 Creating mesh device...")
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
    mesh_device.enable_program_cache()
    
    # Get mesh mappers
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    
    # Load model
    print(f"\n📦 Loading YOLOv8s model (batch_size={num_devices})...")
    print("   ⏳ First run includes trace compilation (~2 min)...")
    
    runner = YOLOv8sPerformantRunner(
        mesh_device,
        device_batch_size=num_devices,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
    )
    
    # Load class names
    names = load_coco_class_names()
    
    # Warmup
    print("\n🔥 Warming up...")
    dummy = torch.randn(num_devices, 3, 640, 640)
    for i in range(3):
        _ = runner.run(dummy)
        print(f"   Warmup {i+1}/3")
    ttnn.synchronize_device(mesh_device)
    print("✅ Model ready!")
    
    # Initialize frame buffers with placeholders
    init_frame_buffers()
    print("📺 Frame buffers initialized")
    
    # Start inference thread
    inf_thread = threading.Thread(
        target=inference_thread,
        args=(video_paths, runner, mesh_device, output_mesh_composer, names, num_devices)
    )
    inf_thread.daemon = True
    inf_thread.start()
    
    # Start HTTP server
    print(f"\n🌐 Starting server on http://localhost:{args.port}")
    print("   Open in browser to see 8 parallel video streams!\n")
    
    server = ThreadedHTTPServer(('0.0.0.0', args.port), StreamHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        running = False
    
    # Cleanup
    runner.release()
    ttnn.close_mesh_device(mesh_device)
    print("✅ Done!")


if __name__ == "__main__":
    main()
