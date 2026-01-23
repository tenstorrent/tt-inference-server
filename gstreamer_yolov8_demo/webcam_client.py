# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Webcam Client - Captures webcam, sends to T3K server, receives bbox, draws locally.

Usage (on laptop):
    python webcam_client.py --server ws://T3K_IP:8765

Requires: pip install opencv-python websockets
"""

import argparse
import asyncio
import base64
import json
import cv2
import websockets

# COCO class names for drawing
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

# Colors for different classes
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]


def draw_detections(frame, detections):
    """Draw bounding boxes on frame."""
    for det in detections:
        x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
        class_id = det['class_id']
        conf = det['confidence']
        label = f"{COCO_NAMES[class_id]}: {conf:.2f}"
        color = COLORS[class_id % len(COLORS)]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame


async def run_client(server_url):
    """Main client loop."""
    print(f"Connecting to {server_url}...")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        async with websockets.connect(server_url) as ws:
            print("Connected! Press 'q' to quit.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize for inference
                frame_resized = cv2.resize(frame, (640, 640))
                
                # Encode frame as JPEG
                _, jpeg = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                
                # Send frame
                await ws.send(json.dumps({'frame': frame_b64}))
                
                # Receive detections
                response = await ws.recv()
                data = json.loads(response)
                
                # Draw detections on original frame (scaled back)
                if 'detections' in data:
                    # Scale detections from 640x640 to original frame size
                    h, w = frame.shape[:2]
                    scale_x, scale_y = w / 640, h / 640
                    for det in data['detections']:
                        det['x1'] *= scale_x
                        det['y1'] *= scale_y
                        det['x2'] *= scale_x
                        det['y2'] *= scale_y
                    
                    frame = draw_detections(frame, data['detections'])
                
                # Show FPS from server
                if 'inference_ms' in data:
                    fps = 1000 / data['inference_ms'] if data['inference_ms'] > 0 else 0
                    cv2.putText(frame, f"TT Inference: {data['inference_ms']:.1f}ms ({fps:.0f} FPS)", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display
                cv2.imshow('YOLOv8s - Tenstorrent', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Webcam client for TT YOLOv8s')
    parser.add_argument('--server', default='ws://localhost:8765', help='WebSocket server URL')
    args = parser.parse_args()
    
    asyncio.run(run_client(args.server))


if __name__ == '__main__':
    main()
