#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
GStreamer YOLOv8s Server - Receives frames via UDP, runs inference, sends bbox via UDP.

Flow:
  [UDP frames in] → YOLOv8s inference → [UDP bbox out]

Usage (on T3K inside Docker):
    python gst_server.py --client-ip CLIENT_IP

The server does NOT send video back - only bounding box JSON.
This is bandwidth efficient: ~1KB/frame instead of ~50KB/frame.
"""

import argparse
import socket
import json
import time
import io
import numpy as np
from PIL import Image
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import torch
import ttnn

from models.demos.yolov8s.runner.performant_runner import YOLOv8sPerformantRunner
from models.demos.utils.common_demo_utils import load_coco_class_names, postprocess as obj_postprocess

# Initialize GStreamer
Gst.init(None)

# Thresholds
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45

# Global model
model = None
device = None
names = None
bbox_socket = None
client_addr = None


def initialize_model():
    """Initialize TT device and load model."""
    global model, device, names
    
    print("[Server] Initializing TT device...")
    device = ttnn.CreateDevice(
        device_id=0,
        l1_small_size=24576,
        trace_region_size=3211264,
        num_command_queues=2,
    )
    device.enable_program_cache()
    
    print("[Server] Loading YOLOv8s model (trace compile ~2 min)...")
    model = YOLOv8sPerformantRunner(device, device_batch_size=1)
    names = load_coco_class_names()
    print("[Server] ✅ Model ready!")


def run_inference(jpeg_bytes):
    """Run inference on JPEG bytes, return detections list."""
    global model, device, names
    
    # Decode JPEG
    img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
    orig_w, orig_h = img.size
    img = img.resize((640, 640))
    
    # Convert to tensor [1, 3, 640, 640]
    img_np = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    
    # Run inference
    start = time.perf_counter()
    preds = model.run(tensor)
    preds = ttnn.to_torch(preds[0], dtype=torch.float32)
    inference_ms = (time.perf_counter() - start) * 1000
    
    # Post-process (returns dict with boxes info)
    frame_bgr = np.array(img)[:, :, ::-1].copy()
    result = obj_postprocess(preds, tensor, [frame_bgr], [["1"]], names)[0]
    
    # Scale detections back to original frame size
    scale_x = orig_w / 640
    scale_y = orig_h / 640
    
    # Result format: {"boxes": {"xyxy": [...], "conf": [...], "cls": [...]}}
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
            detections.append({
                'x1': x1 * scale_x,
                'y1': y1 * scale_y,
                'x2': x2 * scale_x,
                'y2': y2 * scale_y,
                'class_id': int(cls.item() if hasattr(cls, 'item') else cls),
                'confidence': conf
            })
    
    return detections, inference_ms


class FrameProcessor:
    """Processes incoming JPEG frames from GStreamer."""
    
    def __init__(self, client_ip, client_port):
        self.bbox_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_addr = (client_ip, client_port)
        self.frame_count = 0
        self.jpeg_buffer = bytearray()
        
    def on_new_sample(self, sink):
        """Called when a new frame arrives."""
        sample = sink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR
        
        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        
        try:
            jpeg_bytes = bytes(map_info.data)
            
            # Run inference
            detections, inference_ms = run_inference(jpeg_bytes)
            
            # Send bbox JSON to client
            response = {
                'detections': detections,
                'inference_ms': inference_ms,
                'frame': self.frame_count
            }
            self.bbox_socket.sendto(json.dumps(response).encode('utf-8'), self.client_addr)
            
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                fps = 1000 / inference_ms if inference_ms > 0 else 0
                print(f"[Server] Frame {self.frame_count}: {inference_ms:.1f}ms ({fps:.0f} FPS), {len(detections)} detections")
                
        finally:
            buf.unmap(map_info)
        
        return Gst.FlowReturn.OK


def main():
    parser = argparse.ArgumentParser(description='GStreamer YOLOv8s Server')
    parser.add_argument('--client-ip', required=True, help='Client IP to send bboxes to')
    parser.add_argument('--recv-port', type=int, default=5000, help='UDP port to receive frames')
    parser.add_argument('--send-port', type=int, default=5001, help='UDP port to send bboxes')
    args = parser.parse_args()
    
    # Initialize model
    initialize_model()
    
    # Create frame processor
    processor = FrameProcessor(args.client_ip, args.send_port)
    
    # Build GStreamer pipeline to receive JPEG over RTP/UDP
    pipeline_str = f'''
        udpsrc port={args.recv_port} caps="application/x-rtp,media=video,encoding-name=JPEG,payload=26" !
        rtpjpegdepay !
        appsink name=sink emit-signals=true sync=false
    '''
    
    print(f"[Server] Listening for frames on UDP port {args.recv_port}")
    print(f"[Server] Sending bboxes to {args.client_ip}:{args.send_port}")
    
    pipeline = Gst.parse_launch(pipeline_str)
    
    # Connect appsink callback
    sink = pipeline.get_by_name('sink')
    sink.connect('new-sample', processor.on_new_sample)
    
    # Start pipeline
    pipeline.set_state(Gst.State.PLAYING)
    
    # Run main loop
    loop = GLib.MainLoop()
    try:
        print("[Server] Running... Press Ctrl+C to stop")
        loop.run()
    except KeyboardInterrupt:
        print("\n[Server] Stopping...")
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    main()
