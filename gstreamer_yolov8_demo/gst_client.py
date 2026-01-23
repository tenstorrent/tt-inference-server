#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
GStreamer Webcam Client - Pure GStreamer pipeline with UDP transport.

Flow:
  [Webcam] → GStreamer → UDP → [T3K Server] → YOLOv8s → UDP bbox → [Client overlay] → Display

Usage (on laptop):
    python gst_client.py --server-ip T3K_IP

Requires: GStreamer, PyGObject
    Mac: brew install gstreamer gst-plugins-base gst-plugins-good pygobject3
    Linux: apt install gstreamer1.0-tools python3-gst-1.0
"""

import argparse
import socket
import threading
import json
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GLib, GstVideo
import cairo

# Initialize GStreamer
Gst.init(None)

# Global state
current_detections = []
detection_lock = threading.Lock()

# COCO class names
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

COLORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]


def bbox_receiver_thread(port):
    """Thread to receive bounding boxes via UDP."""
    global current_detections
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', port))
    sock.settimeout(1.0)
    
    print(f"[Client] Listening for bboxes on UDP port {port}")
    
    while True:
        try:
            data, addr = sock.recvfrom(65535)
            detections = json.loads(data.decode('utf-8'))
            with detection_lock:
                current_detections = detections.get('detections', [])
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[Client] Bbox receive error: {e}")


def draw_overlay(overlay, cr, timestamp, duration):
    """Cairo overlay callback to draw bounding boxes."""
    global current_detections
    
    cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    cr.set_font_size(14)
    
    with detection_lock:
        detections = current_detections.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        class_id = det.get('class_id', 0)
        conf = det.get('confidence', 0)
        
        color = COLORS[class_id % len(COLORS)]
        cr.set_source_rgb(*color)
        cr.set_line_width(2)
        
        # Draw rectangle
        cr.rectangle(x1, y1, x2 - x1, y2 - y1)
        cr.stroke()
        
        # Draw label
        label = f"{COCO_NAMES[class_id]}: {conf:.2f}"
        cr.move_to(x1, y1 - 5)
        cr.show_text(label)
    
    # Draw FPS/status
    cr.set_source_rgb(0, 1, 0)
    cr.move_to(10, 25)
    cr.show_text(f"Detections: {len(detections)} | TT YOLOv8s")


def main():
    parser = argparse.ArgumentParser(description='GStreamer Webcam Client')
    parser.add_argument('--server-ip', required=True, help='T3K server IP address')
    parser.add_argument('--send-port', type=int, default=5000, help='UDP port to send frames')
    parser.add_argument('--recv-port', type=int, default=5001, help='UDP port to receive bboxes')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    args = parser.parse_args()
    
    # Start bbox receiver thread
    recv_thread = threading.Thread(target=bbox_receiver_thread, args=(args.recv_port,), daemon=True)
    recv_thread.start()
    
    # Build GStreamer pipeline
    # Capture → Tee → [Send to server via UDP] + [Local display with overlay]
    pipeline_str = f'''
        autovideosrc ! 
        videoconvert ! 
        videoscale ! 
        video/x-raw,width={args.width},height={args.height},format=I420 !
        tee name=t
        
        t. ! queue ! 
            jpegenc quality=70 ! 
            rtpjpegpay ! 
            udpsink host={args.server_ip} port={args.send_port}
        
        t. ! queue ! 
            videoconvert ! 
            cairooverlay name=overlay ! 
            videoconvert ! 
            autovideosink sync=false
    '''
    
    print(f"[Client] Starting webcam capture...")
    print(f"[Client] Sending frames to {args.server_ip}:{args.send_port}")
    print(f"[Client] Receiving bboxes on port {args.recv_port}")
    
    pipeline = Gst.parse_launch(pipeline_str)
    
    # Connect overlay callback
    overlay = pipeline.get_by_name('overlay')
    if overlay:
        overlay.connect('draw', draw_overlay)
    
    # Start pipeline
    pipeline.set_state(Gst.State.PLAYING)
    
    # Run main loop
    loop = GLib.MainLoop()
    try:
        print("[Client] Running... Press Ctrl+C to stop")
        loop.run()
    except KeyboardInterrupt:
        print("\n[Client] Stopping...")
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    main()
