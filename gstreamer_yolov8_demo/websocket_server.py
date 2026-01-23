# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
WebSocket Server - Receives frames, runs YOLOv8s on TT, returns bbox JSON only.

Usage (on T3K inside Docker):
    python websocket_server.py

Client connects via: ws://T3K_IP:8765
"""

import asyncio
import base64
import json
import time
import io
import os

import numpy as np
from PIL import Image
import websockets
import torch
import ttnn

from models.demos.yolov8s.runner.performant_runner import YOLOv8sPerformantRunner
from models.demos.utils.common_demo_utils import load_coco_class_names, postprocess as obj_postprocess

# Thresholds
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45

# Global model (initialized once)
model = None
device = None
names = None


def initialize_model():
    """Initialize TT device and load model."""
    global model, device, names
    
    print("[Server] Initializing TT device...", flush=True)
    device = ttnn.CreateDevice(
        device_id=0,
        l1_small_size=24576,
        trace_region_size=3211264,
        num_command_queues=2,
    )
    device.enable_program_cache()
    
    print("[Server] Loading YOLOv8s model (trace compile ~2 min)...", flush=True)
    model = YOLOv8sPerformantRunner(device, device_batch_size=1)
    names = load_coco_class_names()
    print("[Server] Model ready!", flush=True)


def run_inference(frame_bytes):
    """Run inference on a single frame, return detections."""
    global model, device, names
    
    # Decode JPEG and get original size
    img = Image.open(io.BytesIO(frame_bytes)).convert('RGB')
    orig_w, orig_h = img.size  # e.g., 640x480
    img = img.resize((640, 640))
    
    # Scale factors to convert back to original coordinates
    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0
    
    # Convert to tensor [1, 3, 640, 640]
    img_np = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    
    # Run inference
    start = time.perf_counter()
    preds = model.run(tensor)
    preds = ttnn.to_torch(preds[0], dtype=torch.float32)
    inference_ms = (time.perf_counter() - start) * 1000
    
    # Post-process (returns dict with boxes info)
    frame_bgr = np.array(img)[:, :, ::-1].copy()  # RGB to BGR
    result = obj_postprocess(preds, tensor, [frame_bgr], [["1"]], names)[0]
    
    # Extract detections as JSON with confidence filtering
    # Scale coordinates back to original frame size
    detections = []
    if "boxes" in result and len(result["boxes"]["xyxy"]) > 0:
        boxes = result["boxes"]["xyxy"]
        scores = result["boxes"]["conf"]
        classes = result["boxes"]["cls"]
        
        for box, score, cls in zip(boxes, scores, classes):
            conf = float(score.item() if hasattr(score, 'item') else score)
            if conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = [float(v.item() if hasattr(v, 'item') else v) for v in box]
            # Scale back to original frame coordinates
            detections.append({
                'x1': x1 * scale_x, 'y1': y1 * scale_y,
                'x2': x2 * scale_x, 'y2': y2 * scale_y,
                'class_id': int(cls.item() if hasattr(cls, 'item') else cls),
                'confidence': conf
            })
    
    return detections, inference_ms


async def handle_client(websocket):
    """Handle incoming WebSocket connection."""
    print(f"[Server] Client connected from {websocket.remote_address}", flush=True)
    
    try:
        async for message in websocket:
            print(f"[Server] Got message ({len(message)} bytes)", flush=True)
            data = json.loads(message)
            
            if 'frame' in data:
                # Decode base64 frame
                frame_bytes = base64.b64decode(data['frame'])
                print(f"[Server] Frame decoded ({len(frame_bytes)} bytes), running inference...", flush=True)
                
                # Run inference
                detections, inference_ms = run_inference(frame_bytes)
                print(f"[Server] Inference done: {len(detections)} detections in {inference_ms:.1f}ms", flush=True)
                
                # Send back JSON only (no image!)
                response = {
                    'detections': detections,
                    'inference_ms': inference_ms,
                    'count': len(detections)
                }
                await websocket.send(json.dumps(response))
                
    except websockets.exceptions.ConnectionClosed:
        print(f"[Server] Client disconnected", flush=True)
    except Exception as e:
        print(f"[Server] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()


async def main():
    """Start WebSocket server."""
    initialize_model()
    
    port = int(os.environ.get('WS_PORT', 8765))
    print(f"[Server] Starting WebSocket server on port {port}...", flush=True)
    
    async with websockets.serve(handle_client, "0.0.0.0", port):
        print(f"[Server] Ready! Connect via ws://localhost:{port}", flush=True)
        await asyncio.Future()  # Run forever


if __name__ == '__main__':
    asyncio.run(main())
