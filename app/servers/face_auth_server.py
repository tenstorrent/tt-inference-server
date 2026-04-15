#!/usr/bin/env python3
"""
Face Auth Server - Persistent YuNet + SFace model server.

Keeps face recognition models loaded on TT device 0, accepts images via Unix socket,
returns face detection and recognition results.

Based on tt-inference-server/gstreamer_face_matching_demo/websocket_server.py

Usage (inside container):
  TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  TT_VISIBLE_DEVICES='0' \
  python face_auth_server.py

Protocol (Unix socket at /tmp/face_auth_server.sock):
  REQ: {"image_base64": "..."} or {"cmd": "ping"}
  REP: {"status": "ok", "faces": [...], "time_ms": 123.4}
"""

import os
import sys
import json
import time
import socket
import base64
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch

# Suppress verbose logging
os.environ["TT_METAL_LOGGER_LEVEL"] = "ERROR"
os.environ["LOGURU_LEVEL"] = "WARNING"

# Add tt-metal to path
sys.path.insert(0, "/home/container_app_user/tt-metal")

import ttnn
import ttnn.distributed as dist

# Global models
yunet_model = None
sface_model = None
device = None
face_database = {}

FACES_DIR = Path("/home/container_app_user/voice-assistant/registered_faces")


def load_faces_from_disk():
    """Load saved faces from disk."""
    global face_database
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    
    for person_dir in FACES_DIR.iterdir():
        if person_dir.is_dir():
            embedding_path = person_dir / "embedding.npy"
            if embedding_path.exists():
                name = person_dir.name
                face_database[name] = np.load(embedding_path)
                print(f"[FaceAuth] Loaded face: {name}", flush=True)
    
    print(f"[FaceAuth] Loaded {len(face_database)} faces", flush=True)


def save_face_to_disk(name, embedding, face_image):
    """Save face embedding and image to disk."""
    person_dir = FACES_DIR / name
    person_dir.mkdir(exist_ok=True)
    np.save(person_dir / "embedding.npy", embedding)
    cv2.imwrite(str(person_dir / "face.jpg"), cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
    print(f"[FaceAuth] Saved face: {name}", flush=True)


def init_models(physical_device_id: int = 0):
    """Initialize TT device and load models."""
    global yunet_model, sface_model, device
    
    from models.experimental.yunet.common import (
        YUNET_L1_SMALL_SIZE,
        load_torch_model as load_yunet_torch,
        get_default_weights_path as get_yunet_weights,
    )
    from models.experimental.yunet.tt.ttnn_yunet import create_yunet_model
    from models.experimental.sface.common import get_sface_onnx_path, SFACE_L1_SMALL_SIZE
    from models.experimental.sface.reference.sface_model import load_sface_from_onnx
    from models.experimental.sface.tt.ttnn_sface import create_sface_model
    
    print("[FaceAuth] Initializing Tenstorrent device...", flush=True)
    
    l1_size = max(YUNET_L1_SMALL_SIZE, SFACE_L1_SMALL_SIZE)
    
    # Use 1x1 mesh with specific physical device
    mesh = dist.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[physical_device_id],
        l1_small_size=l1_size,
        trace_region_size=0
    )
    device = mesh
    device.enable_program_cache()
    print(f"[FaceAuth] Device opened (physical_device_id={physical_device_id})", flush=True)
    
    # Load YuNet
    print("[FaceAuth] Loading YuNet model...", flush=True)
    yunet_weights = get_yunet_weights()
    yunet_torch = load_yunet_torch(yunet_weights)
    yunet_torch = yunet_torch.to(torch.bfloat16)
    yunet_model = create_yunet_model(device, yunet_torch)
    print("[FaceAuth] YuNet loaded!", flush=True)
    
    # Load SFace
    print("[FaceAuth] Loading SFace model...", flush=True)
    sface_onnx = get_sface_onnx_path()
    sface_torch = load_sface_from_onnx(sface_onnx)
    sface_torch.eval()
    sface_model = create_sface_model(device, sface_torch)
    print("[FaceAuth] SFace loaded!", flush=True)
    
    # Warmup YuNet (2 runs)
    print("[FaceAuth] Warming up YuNet...", flush=True)
    for _ in range(2):
        warmup = torch.randint(0, 256, (1, 640, 640, 3), dtype=torch.float32).to(torch.bfloat16)
        warmup_tt = ttnn.from_torch(warmup, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        _ = yunet_model(warmup_tt)
        ttnn.synchronize_device(device)
    print("[FaceAuth] YuNet warmed up!", flush=True)
    
    # Warmup SFace (2 runs)
    print("[FaceAuth] Warming up SFace...", flush=True)
    for _ in range(2):
        warmup = torch.randint(0, 256, (1, 112, 112, 3), dtype=torch.float32)
        warmup_tt = ttnn.from_torch(warmup, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        _ = sface_model(warmup_tt)
        ttnn.synchronize_device(device)
    print("[FaceAuth] SFace warmed up!", flush=True)
    
    # Load saved faces
    load_faces_from_disk()
    
    print("[FaceAuth] Ready!", flush=True)


def decode_yunet_output(cls_outs, box_outs, obj_outs, kpt_outs, input_size=640, threshold=0.35):
    """Decode YuNet outputs to face detections."""
    from models.experimental.yunet.common import STRIDES, DEFAULT_NMS_IOU_THRESHOLD
    
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
            for i in range(len(indices[0])):
                b, c, h, w = indices[0][i], indices[1][i], indices[2][i], indices[3][i]
                conf = score[b, c, h, w].item()
                anchor_x, anchor_y = w.item() * stride, h.item() * stride
                
                dx, dy = box_out[b, 0, h, w].item(), box_out[b, 1, h, w].item()
                dw, dh = box_out[b, 2, h, w].item(), box_out[b, 3, h, w].item()
                
                cx, cy = dx * stride + anchor_x, dy * stride + anchor_y
                bw, bh = np.exp(dw) * stride, np.exp(dh) * stride
                
                x1, y1 = cx - bw / 2, cy - bh / 2
                x2, y2 = cx + bw / 2, cy + bh / 2
                
                keypoints = []
                for k in range(5):
                    kpt_dx = kpt_out[b, k * 2, h, w].item()
                    kpt_dy = kpt_out[b, k * 2 + 1, h, w].item()
                    kx = kpt_dx * stride + anchor_x
                    ky = kpt_dy * stride + anchor_y
                    keypoints.append([kx, ky])
                
                detections.append({"box": [x1, y1, x2, y2], "conf": conf, "keypoints": keypoints})
    
    # NMS
    detections = sorted(detections, key=lambda x: x["conf"], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        remaining = []
        for det in detections:
            x1 = max(best["box"][0], det["box"][0])
            y1 = max(best["box"][1], det["box"][1])
            x2 = min(best["box"][2], det["box"][2])
            y2 = min(best["box"][3], det["box"][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (best["box"][2] - best["box"][0]) * (best["box"][3] - best["box"][1])
            area2 = (det["box"][2] - det["box"][0]) * (det["box"][3] - det["box"][1])
            if inter / max(area1 + area2 - inter, 1e-6) < 0.5:
                remaining.append(det)
        detections = remaining
    
    return keep


def detect_faces(frame_rgb):
    """Run YuNet face detection."""
    global yunet_model, device
    
    h, w = frame_rgb.shape[:2]
    
    # Resize to 640x640 for YuNet
    img_resized = cv2.resize(frame_rgb, (640, 640))
    tensor = torch.from_numpy(img_resized.astype(np.float32)).unsqueeze(0).to(torch.bfloat16)
    
    tt_input = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    cls_out, box_out, obj_out, kpt_out = yunet_model(tt_input)
    ttnn.synchronize_device(device)
    
    detections = decode_yunet_output(cls_out, box_out, obj_out, kpt_out, 640, 0.35)
    
    # Scale boxes back to original frame size
    scale_x, scale_y = w / 640, h / 640
    for det in detections:
        det["box"] = [
            det["box"][0] * scale_x,
            det["box"][1] * scale_y,
            det["box"][2] * scale_x,
            det["box"][3] * scale_y,
        ]
        if det.get("keypoints"):
            det["keypoints"] = [[kp[0] * scale_x, kp[1] * scale_y] for kp in det["keypoints"]]
    
    return detections


def get_embedding(face_image):
    """Extract face embedding using SFace."""
    global sface_model, device
    
    face_tensor = torch.from_numpy(face_image.astype(np.float32)).unsqueeze(0)
    tt_input = ttnn.from_torch(face_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = sface_model(tt_input)
    embedding = ttnn.to_torch(tt_output).float().numpy().flatten()
    return embedding


def align_face_keypoints(image, keypoints, target_size=112):
    """Align face using 5 keypoints."""
    left_eye = np.array(keypoints[0], dtype=np.float32)
    right_eye = np.array(keypoints[1], dtype=np.float32)
    nose = np.array(keypoints[2], dtype=np.float32)
    
    src_pts = np.float32([left_eye, right_eye, nose])
    dst_pts = np.float32([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
    ])
    
    M = cv2.getAffineTransform(src_pts, dst_pts)
    aligned = cv2.warpAffine(image, M, (target_size, target_size), borderMode=cv2.BORDER_REPLICATE)
    return aligned


def match_face(embedding, threshold=0.5):
    """Match embedding against face database."""
    if not face_database:
        return "Unknown", 0.0
    
    best_match, best_score = None, 0.0
    for name, db_embedding in face_database.items():
        score = np.dot(embedding, db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding) + 1e-8)
        if score > best_score:
            best_score = score
            best_match = name
    
    if best_score >= threshold:
        return best_match, best_score
    return "Unknown", best_score


def process_frame(frame_rgb):
    """Process frame: detect faces and recognize."""
    results = []
    
    detections = detect_faces(frame_rgb)
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        face_w, face_h = x2 - x1, y2 - y1
        
        if face_w < 30 or face_h < 30:
            continue
        
        keypoints = det.get("keypoints")
        if keypoints and len(keypoints) >= 5:
            face_aligned = align_face_keypoints(frame_rgb, keypoints, target_size=112)
        else:
            margin = int(max(face_w, face_h) * 0.1)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame_rgb.shape[1], x2 + margin)
            y2 = min(frame_rgb.shape[0], y2 + margin)
            face_crop = frame_rgb[y1:y2, x1:x2]
            face_aligned = cv2.resize(face_crop, (112, 112))
        
        embedding = get_embedding(face_aligned)
        identity, score = match_face(embedding)
        
        results.append({
            "x1": det["box"][0], "y1": det["box"][1],
            "x2": det["box"][2], "y2": det["box"][3],
            "identity": identity,
            "score": float(score),
            "conf": det["conf"],
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Face Auth Server")
    parser.add_argument("--socket", type=str, default="/tmp/face_auth_server.sock", help="Unix socket path")
    parser.add_argument("--device-id", type=int, default=0, help="Physical device ID")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("Face Auth Server - YuNet + SFace (Persistent)", flush=True)
    print("=" * 60, flush=True)
    print(f"Socket: {args.socket}", flush=True)
    print(f"Device ID: {args.device_id}", flush=True)
    print(f"TT_VISIBLE_DEVICES: {os.environ.get('TT_VISIBLE_DEVICES', 'not set')}", flush=True)

    # Initialize models
    t0 = time.time()
    init_models(args.device_id)
    print(f"\n[FaceAuth] Total init time: {time.time() - t0:.1f}s", flush=True)

    # Start server
    print("\n" + "=" * 60, flush=True)
    print("READY - Starting Unix socket server...", flush=True)
    print("=" * 60, flush=True)

    if os.path.exists(args.socket):
        os.unlink(args.socket)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    server.listen(1)
    os.chmod(args.socket, 0o777)

    print(f"Listening on {args.socket}", flush=True)

    try:
        while True:
            conn, addr = server.accept()
            try:
                data = conn.recv(10 * 1024 * 1024)  # 10MB max for images
                if not data:
                    continue

                request = json.loads(data.decode('utf-8'))

                if request.get("cmd") == "ping":
                    conn.sendall(json.dumps({"status": "ok", "model": "face_auth", "ready": True}).encode('utf-8'))
                    continue

                if request.get("cmd") == "shutdown":
                    conn.sendall(json.dumps({"status": "shutting_down"}).encode('utf-8'))
                    break

                if request.get("cmd") == "register":
                    # Register a new face
                    name = request.get("name", "Unknown")
                    image_b64 = request.get("image_base64", "")
                    if image_b64:
                        img_bytes = base64.b64decode(image_b64)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        detections = detect_faces(frame_rgb)
                        if detections:
                            det = max(detections, key=lambda d: d["conf"])
                            keypoints = det.get("keypoints")
                            if keypoints and len(keypoints) >= 5:
                                face_aligned = align_face_keypoints(frame_rgb, keypoints)
                            else:
                                x1, y1, x2, y2 = map(int, det["box"])
                                face_crop = frame_rgb[y1:y2, x1:x2]
                                face_aligned = cv2.resize(face_crop, (112, 112))
                            
                            embedding = get_embedding(face_aligned)
                            face_database[name] = embedding
                            save_face_to_disk(name, embedding, face_aligned)
                            conn.sendall(json.dumps({"status": "ok", "registered": name}).encode('utf-8'))
                        else:
                            conn.sendall(json.dumps({"status": "error", "error": "No face detected"}).encode('utf-8'))
                    continue

                # Process image for face recognition
                image_b64 = request.get("image_base64", "")
                if not image_b64:
                    conn.sendall(json.dumps({"status": "error", "error": "missing image_base64"}).encode('utf-8'))
                    continue

                t0 = time.time()
                img_bytes = base64.b64decode(image_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                faces = process_frame(frame_rgb)
                elapsed_ms = (time.time() - t0) * 1000

                print(f"[FaceAuth] Processed frame: {len(faces)} faces in {elapsed_ms:.1f}ms", flush=True)

                conn.sendall(json.dumps({
                    "status": "ok",
                    "faces": faces,
                    "time_ms": elapsed_ms,
                }).encode('utf-8'))

            except Exception as e:
                print(f"[Error] {e}", flush=True)
                try:
                    conn.sendall(json.dumps({"status": "error", "error": str(e)}).encode('utf-8'))
                except:
                    pass
            finally:
                conn.close()

    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
    finally:
        server.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)
        dist.close_mesh_device(device)
        print("Device closed.", flush=True)


if __name__ == "__main__":
    main()
