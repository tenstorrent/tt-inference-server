# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
WebSocket server for browser webcam face recognition on Tenstorrent.

Receives JPEG frames from browser, runs YuNet+SFace inference, returns face data.
"""

import asyncio
import base64
import io
import json
import time
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

import websockets

# Suppress verbose logging
os.environ["TT_METAL_LOGGER_LEVEL"] = "ERROR"
os.environ["LOGURU_LEVEL"] = "WARNING"

import ttnn

# Face database
FACES_DIR = Path(__file__).parent / "registered_faces"
FACES_DIR.mkdir(exist_ok=True)
face_database = {}  # name -> embedding

# Global models
yunet_model = None
sface_model = None
device = None


def load_faces_from_disk():
    """Load saved faces from disk."""
    global face_database
    if not FACES_DIR.exists():
        return
    for person_dir in FACES_DIR.iterdir():
        if person_dir.is_dir():
            embedding_path = person_dir / "embedding.npy"
            if embedding_path.exists():
                name = person_dir.name
                face_database[name] = np.load(embedding_path)
                print(f"[FaceRecognition] Loaded face: {name}", flush=True)
    print(f"[FaceRecognition] Loaded {len(face_database)} faces", flush=True)


def process_pending_faces():
    """Process faces that were registered without embeddings (pending_embedding flag)."""
    global face_database, sface_model, device
    
    if not FACES_DIR.exists() or sface_model is None:
        return
    
    pending_count = 0
    for person_dir in FACES_DIR.iterdir():
        if person_dir.is_dir():
            pending_flag = person_dir / "pending_embedding"
            if pending_flag.exists():
                name = person_dir.name
                face_path = person_dir / "face.jpg"
                
                if face_path.exists():
                    try:
                        # Load face image (already aligned 112x112 from registration)
                        face_img = cv2.imread(str(face_path))
                        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        
                        # If it's not 112x112, resize (legacy faces)
                        if face_rgb.shape[:2] != (112, 112):
                            face_rgb = cv2.resize(face_rgb, (112, 112))
                        
                        # Get embedding
                        embedding = get_embedding(face_rgb)
                        
                        # Save embedding
                        np.save(person_dir / "embedding.npy", embedding)
                        face_database[name] = embedding
                        
                        # Remove pending flag
                        pending_flag.unlink()
                        pending_count += 1
                        print(f"[FaceRecognition] Processed pending face: {name}", flush=True)
                    except Exception as e:
                        print(f"[FaceRecognition] Failed to process {name}: {e}", flush=True)
    
    if pending_count > 0:
        print(f"[FaceRecognition] Processed {pending_count} pending faces", flush=True)


def save_face_to_disk(name, embedding, face_image):
    """Save face embedding and image to disk."""
    person_dir = FACES_DIR / name
    person_dir.mkdir(exist_ok=True)
    np.save(person_dir / "embedding.npy", embedding)
    cv2.imwrite(str(person_dir / "face.jpg"), cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
    print(f"[FaceRecognition] Saved face: {name}", flush=True)


def delete_face_from_disk(name):
    """Delete face from disk."""
    import shutil
    person_dir = FACES_DIR / name
    if person_dir.exists():
        shutil.rmtree(person_dir)


def init_models():
    """Initialize TT device and load models."""
    global yunet_model, sface_model, device
    
    # Add models to path
    models_path = Path(__file__).parent / "models"
    if models_path.exists():
        sys.path.insert(0, str(models_path.parent))
    
    from models.experimental.yunet.common import (
        YUNET_L1_SMALL_SIZE,
        load_torch_model as load_yunet_torch,
        get_default_weights_path as get_yunet_weights,
    )
    from models.experimental.yunet.tt.ttnn_yunet import create_yunet_model
    from models.experimental.sface.common import get_sface_onnx_path, SFACE_L1_SMALL_SIZE
    from models.experimental.sface.reference.sface_model import load_sface_from_onnx
    from models.experimental.sface.tt.ttnn_sface import create_sface_model
    
    print("[FaceRecognition] Initializing Tenstorrent device...", flush=True)
    
    import ttnn.distributed as dist
    
    l1_size = max(YUNET_L1_SMALL_SIZE, SFACE_L1_SMALL_SIZE)
    
    # Use 1x1 mesh with specific physical device to avoid T3K L1 clash issues
    mesh = dist.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[0],
        l1_small_size=l1_size,
        trace_region_size=0
    )
    device = mesh  # MeshDevice can be used like a device
    device.enable_program_cache()
    
    # Load YuNet
    print("[FaceRecognition] Loading YuNet model...", flush=True)
    yunet_weights = get_yunet_weights()
    yunet_torch = load_yunet_torch(yunet_weights)
    yunet_torch = yunet_torch.to(torch.bfloat16)
    yunet_model = create_yunet_model(device, yunet_torch)
    print("[FaceRecognition] YuNet loaded!", flush=True)
    
    # Load SFace
    print("[FaceRecognition] Loading SFace model...", flush=True)
    sface_onnx = get_sface_onnx_path()
    sface_torch = load_sface_from_onnx(sface_onnx)
    sface_torch.eval()
    sface_model = create_sface_model(device, sface_torch)
    print("[FaceRecognition] SFace loaded!", flush=True)
    
    # Warmup (single run each)
    print("[FaceRecognition] Warming up YuNet...", flush=True)
    warmup = torch.randint(0, 256, (1, 640, 640, 3), dtype=torch.float32).to(torch.bfloat16)
    warmup_tt = ttnn.from_torch(warmup, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    _ = yunet_model(warmup_tt)
    ttnn.synchronize_device(device)
    
    print("[FaceRecognition] Warming up SFace...", flush=True)
    warmup = torch.randint(0, 256, (1, 112, 112, 3), dtype=torch.float32)
    warmup_tt = ttnn.from_torch(warmup, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    _ = sface_model(warmup_tt)
    ttnn.synchronize_device(device)
    
    # Load saved faces
    load_faces_from_disk()
    
    # Process any pending faces that were registered without embeddings
    process_pending_faces()
    
    print("[FaceRecognition] Ready!", flush=True)


def decode_yunet_output(cls_outs, box_outs, obj_outs, kpt_outs, input_size, threshold=0.35):
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
                
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2
                
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
            if inter / max(area1 + area2 - inter, 1e-6) < DEFAULT_NMS_IOU_THRESHOLD:
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


def align_face_keypoints(image: np.ndarray, keypoints: list, target_size: int = 112) -> np.ndarray:
    """Align face using 5 keypoints (eyes, nose, mouth corners).
    
    Uses affine transformation to:
    1. Make eyes horizontal
    2. Center the face
    3. Scale to target size
    
    Args:
        image: numpy array (H, W, 3) - full frame
        keypoints: List of 5 [x, y] pairs in PIXEL coordinates:
                   [[left_eye], [right_eye], [nose], [left_mouth], [right_mouth]]
        target_size: output size (default 112x112 for SFace)
    
    Returns:
        Aligned face image (target_size x target_size x 3)
    """
    # Source points (detected keypoints) - already in pixel coords
    left_eye = np.array(keypoints[0], dtype=np.float32)
    right_eye = np.array(keypoints[1], dtype=np.float32)
    nose = np.array(keypoints[2], dtype=np.float32)
    
    src_pts = np.float32([left_eye, right_eye, nose])
    
    # Reference destination points for 112x112 aligned face
    # Standard positions for SFace/ArcFace alignment
    dst_pts = np.float32([
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
    ])
    
    # Compute affine transform
    M = cv2.getAffineTransform(src_pts, dst_pts)
    
    # Apply transform
    aligned = cv2.warpAffine(image, M, (target_size, target_size), borderMode=cv2.BORDER_REPLICATE)
    
    return aligned


def match_face(embedding, threshold=0.5):
    """Match embedding against face database."""
    if not face_database:
        return "Unknown", 0.0
    
    best_match = None
    best_score = 0.0
    
    for name, db_embedding in face_database.items():
        score = np.dot(embedding, db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding) + 1e-8)
        if score > best_score:
            best_score = score
            best_match = name
    
    if best_score >= threshold:
        return best_match, best_score
    return "Unknown", best_score


# Frame counter for timing prints
_frame_count = 0

def process_frame(frame_rgb):
    """Process frame: detect faces and recognize (legacy, no timing return)."""
    faces, _ = process_frame_with_timing(frame_rgb, 0)
    return faces

def process_frame_with_timing(frame_rgb, decode_ms=0):
    """Process frame: detect faces and recognize, with full timing."""
    global _frame_count
    _frame_count += 1
    
    results = []
    timing = {'decode': decode_ms}
    
    # Detect faces (YuNet)
    t0 = time.time()
    detections = detect_faces(frame_rgb)
    timing['yunet'] = (time.time() - t0) * 1000
    
    # Recognize each face (SFace)
    t1 = time.time()
    sface_times = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        
        # Skip small faces
        face_w, face_h = x2 - x1, y2 - y1
        if face_w < 30 or face_h < 30:
            continue
        
        t_face = time.time()
        
        # Use keypoint alignment if available (proper SFace alignment)
        keypoints = det.get("keypoints")
        if keypoints and len(keypoints) >= 5:
            # Align face using 5 keypoints - this is critical for accuracy!
            face_aligned = align_face_keypoints(frame_rgb, keypoints, target_size=112)
        else:
            # Fallback: simple crop and resize (lower accuracy)
            margin = int(max(face_w, face_h) * 0.1)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame_rgb.shape[1], x2 + margin)
            y2 = min(frame_rgb.shape[0], y2 + margin)
            face_crop = frame_rgb[y1:y2, x1:x2]
            face_aligned = cv2.resize(face_crop, (112, 112))
        
        embedding = get_embedding(face_aligned)
        identity, score = match_face(embedding)
        sface_times.append((time.time() - t_face) * 1000)
        
        results.append({
            "x1": det["box"][0],
            "y1": det["box"][1],
            "x2": det["box"][2],
            "y2": det["box"][3],
            "keypoints": det.get("keypoints"),
            "identity": identity,
            "score": float(score),
            "conf": det["conf"],
        })
    
    timing['sface'] = (time.time() - t1) * 1000
    timing['total'] = timing['decode'] + timing['yunet'] + timing['sface']
    timing['num_faces'] = len(results)
    
    # Print timing for first 10 frames and every 10th frame
    if _frame_count <= 10 or _frame_count % 10 == 0:
        if _frame_count == 1:
            print("\n" + "="*70, flush=True)
            print("  WEBCAM PIPELINE - TIMING BREAKDOWN (TTNN)", flush=True)
            print("  Decode=JPEG decode | YuNet=detection | SFace=recognition", flush=True)
            print("="*70, flush=True)
        
        total = timing['total'] if timing['total'] > 0 else 1
        print(f"Frame {_frame_count}: Decode={timing['decode']:.1f}ms  YuNet={timing['yunet']:.1f}ms  SFace={timing['sface']:.1f}ms({timing['num_faces']})  Total={total:.1f}ms  FPS={1000/total:.0f}", flush=True)
    
    return results, timing


def register_face_from_frame(frame_rgb, name):
    """Register a face from a single frame."""
    global face_database
    
    detections = detect_faces(frame_rgb)
    
    if not detections:
        return False, "No face detected", None
    
    # Use the most confident detection
    det = max(detections, key=lambda d: d["conf"])
    
    x1, y1, x2, y2 = map(int, det["box"])
    
    # Use keypoint alignment if available
    keypoints = det.get("keypoints")
    if keypoints and len(keypoints) >= 5:
        face_aligned = align_face_keypoints(frame_rgb, keypoints, target_size=112)
    else:
        # Fallback: simple crop
        margin = int(max(x2 - x1, y2 - y1) * 0.1)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame_rgb.shape[1], x2 + margin)
        y2 = min(frame_rgb.shape[0], y2 + margin)
        face_crop = frame_rgb[y1:y2, x1:x2]
        face_aligned = cv2.resize(face_crop, (112, 112))
    
    embedding = get_embedding(face_aligned)
    return True, "OK", (embedding, face_aligned)


def register_face_from_multiple_frames(frames_rgb_list, name):
    """Register a face from multiple frames (up to 3) and average embeddings."""
    global face_database
    
    embeddings = []
    best_face_image = None
    best_conf = 0
    
    for i, frame_rgb in enumerate(frames_rgb_list):
        detections = detect_faces(frame_rgb)
        
        if not detections:
            print(f"[Register] No face detected in photo {i+1}", flush=True)
            continue
        
        # Use the most confident detection
        det = max(detections, key=lambda d: d["conf"])
        
        x1, y1, x2, y2 = map(int, det["box"])
        
        # Use keypoint alignment if available
        keypoints = det.get("keypoints")
        if keypoints and len(keypoints) >= 5:
            face_aligned = align_face_keypoints(frame_rgb, keypoints, target_size=112)
        else:
            # Fallback: simple crop
            margin = int(max(x2 - x1, y2 - y1) * 0.1)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame_rgb.shape[1], x2 + margin)
            y2 = min(frame_rgb.shape[0], y2 + margin)
            face_crop = frame_rgb[y1:y2, x1:x2]
            face_aligned = cv2.resize(face_crop, (112, 112))
        
        embedding = get_embedding(face_aligned)
        embeddings.append(embedding)
        
        if det["conf"] > best_conf:
            best_conf = det["conf"]
            best_face_image = face_aligned
        
        print(f"[Register] Photo {i+1}: face detected with conf {det['conf']:.2f}", flush=True)
    
    if not embeddings:
        return False, "No faces detected in any photo"
    
    # Average the embeddings
    avg_embedding = np.mean(embeddings, axis=0)
    # Re-normalize after averaging
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    
    face_database[name] = avg_embedding
    save_face_to_disk(name, avg_embedding, best_face_image)
    
    return True, f"Registered {name} from {len(embeddings)} photo(s)"


async def handle_websocket(websocket):
    """Handle WebSocket connection."""
    print(f"[WebSocket] Client connected from {websocket.remote_address}", flush=True)
    
    try:
        async for message in websocket:
            print(f"[WebSocket] Received message type: {list(json.loads(message).keys())}", flush=True)
            data = json.loads(message)
            
            if "frame" in data:
                # Process frame
                t_start = time.time()
                
                # Decode JPEG (this is the "decoder" in webcam flow)
                t_decode = time.time()
                img_data = base64.b64decode(data["frame"])
                img_np = np.frombuffer(img_data, dtype=np.uint8)
                frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                decode_ms = (time.time() - t_decode) * 1000
                
                # Run inference (returns timing dict)
                faces, timing = process_frame_with_timing(frame_rgb, decode_ms)
                
                t_end = time.time()
                inference_ms = (t_end - t_start) * 1000
                
                response = {
                    "faces": faces,
                    "count": len(faces),
                    "inference_ms": inference_ms,
                }
                await websocket.send(json.dumps(response))
            
            elif "register" in data:
                # Register face - supports multiple photos
                name = data["register"]["name"]
                
                # Check if multiple frames or single frame
                if "frames" in data["register"]:
                    # Multiple photos (new format)
                    frames_rgb = []
                    for base64_frame in data["register"]["frames"]:
                        img_data = base64.b64decode(base64_frame)
                        img_np = np.frombuffer(img_data, dtype=np.uint8)
                        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames_rgb.append(frame_rgb)
                    
                    success, message = register_face_from_multiple_frames(frames_rgb, name)
                else:
                    # Single photo (legacy format)
                    img_data = base64.b64decode(data["register"]["frame"])
                    img_np = np.frombuffer(img_data, dtype=np.uint8)
                    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    success, msg, result = register_face_from_frame(frame_rgb, name)
                    if success and result:
                        embedding, face_img = result
                        face_database[name] = embedding
                        save_face_to_disk(name, embedding, face_img)
                        message = f"Registered face for {name}"
                    else:
                        message = msg
                
                response = {
                    "register_result": {
                        "success": success,
                        "message": message,
                    }
                }
                await websocket.send(json.dumps(response))
            
            elif "list_faces" in data:
                response = {
                    "faces": list(face_database.keys()),
                }
                await websocket.send(json.dumps(response))
            
            elif "delete_face" in data:
                name = data["delete_face"]
                if name in face_database:
                    del face_database[name]
                    delete_face_from_disk(name)
                    response = {"deleted": True, "name": name}
                else:
                    response = {"deleted": False, "error": f"Face '{name}' not found"}
                await websocket.send(json.dumps(response))
    
    except websockets.exceptions.ConnectionClosed:
        print(f"[WebSocket] Client disconnected", flush=True)
    except Exception as e:
        print(f"[WebSocket] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()


async def main():
    port = int(os.environ.get("WS_PORT", 8765))
    
    print("=" * 50, flush=True)
    print("  Face Recognition - WebSocket Server", flush=True)
    print("=" * 50, flush=True)
    print(f"  WebSocket: ws://localhost:{port}", flush=True)
    print("  Initializing models...", flush=True)
    print("=" * 50, flush=True)
    
    init_models()
    
    print(f"\n[WebSocket] Starting server on port {port}...", flush=True)
    async with websockets.serve(handle_websocket, "0.0.0.0", port):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
