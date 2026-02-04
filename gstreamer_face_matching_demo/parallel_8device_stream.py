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

# Enable ethernet dispatch for multi-device mesh (required for 8-device T3K)
os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

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

# Face database for recognition
face_database = {}  # name -> embedding
FACES_DIR = Path("/app/registered_faces")
RECOGNITION_THRESHOLD = 0.4


def align_face_keypoints(image: np.ndarray, keypoints: list, target_size: int = 112) -> np.ndarray:
    """Align face using 5 keypoints (eyes, nose, mouth corners).
    
    Uses affine transformation to make eyes horizontal, center the face, and scale.
    
    Args:
        image: numpy array (H, W, 3) - full frame
        keypoints: List of 5 [x, y] pairs: [left_eye, right_eye, nose, left_mouth, right_mouth]
        target_size: output size (default 112x112 for SFace)
    
    Returns:
        Aligned face image (target_size x target_size x 3)
    """
    # Source points (detected keypoints)
    left_eye = np.array(keypoints[0], dtype=np.float32)
    right_eye = np.array(keypoints[1], dtype=np.float32)
    nose = np.array(keypoints[2], dtype=np.float32)
    
    src_pts = np.float32([left_eye, right_eye, nose])
    
    # Reference destination points for 112x112 aligned face (SFace/ArcFace standard)
    dst_pts = np.float32([
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
    ])
    
    # Compute and apply affine transform
    M = cv2.getAffineTransform(src_pts, dst_pts)
    aligned = cv2.warpAffine(image, M, (target_size, target_size), borderMode=cv2.BORDER_REPLICATE)
    
    return aligned


def load_faces_from_disk():
    """Load registered face embeddings from disk."""
    global face_database
    if not FACES_DIR.exists():
        print(f"[FaceDB] No faces directory at {FACES_DIR}", flush=True)
        return
    
    for person_dir in FACES_DIR.iterdir():
        if person_dir.is_dir():
            name = person_dir.name
            embedding_path = person_dir / "embedding.npy"
            if embedding_path.exists():
                face_database[name] = np.load(embedding_path)
    
    print(f"[FaceDB] Loaded {len(face_database)} registered faces", flush=True)

def match_face(embedding):
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
    
    if best_score >= RECOGNITION_THRESHOLD:
        return best_match, best_score
    return "Unknown", best_score

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


def init_frame_buffers(num_devices=4):
    """Initialize frame buffers with placeholder images."""
    global frame_buffers, frame_locks
    for i in range(num_devices):
        # Create loading placeholder
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, f"Device {i}", (220, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, DEVICE_COLORS[i % len(DEVICE_COLORS)], 3)
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
    """Run parallel inference on 8 individual devices."""
    global frame_buffers, stats
    
    num_devices = len(devices)
    frame_indices = [0] * num_devices
    
    loop_times = []
    inference_times = []
    
    print(f"[Inference] Starting inference on {num_devices} devices...")
    
    while not stop_event.is_set():
        loop_start = time.time()
        t_inf_start = time.time()
        
        # Process each device
        for i in range(num_devices):
            frames = video_frames[i]
            idx = frame_indices[i] % len(frames)
            frame_indices[i] = idx + 1
            frame_rgb = frames[idx]
            
            yunet_model, sface_model = models[i]
            device = devices[i]
            
            # Prepare input
            img_resized = cv2.resize(frame_rgb, (640, 640))
            tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).unsqueeze(0).to(torch.bfloat16)
            tt_input = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            
            # Run YuNet
            cls_out, box_out, obj_out, kpt_out = yunet_model(tt_input)
            ttnn.synchronize_device(device)
            
            # Decode
            detections = decode_yunet_simple(cls_out, box_out, obj_out, kpt_out, 640)
            
            # Draw
            scale_x, scale_y = 640 / 640, 480 / 640
            frame_out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            for det in detections[:10]:
                x1 = int(det["box"][0] * scale_x)
                y1 = int(det["box"][1] * scale_y)
                x2 = int(det["box"][2] * scale_x)
                y2 = int(det["box"][3] * scale_y)
                cv2.rectangle(frame_out, (x1, y1), (x2, y2), DEVICE_COLORS[i], 2)
            
            cv2.putText(frame_out, f"Device {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, DEVICE_COLORS[i], 2)
            cv2.putText(frame_out, f"Faces: {len(detections)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            with frame_locks[i]:
                frame_buffers[i] = frame_out
        
        t_inf_end = time.time()
        inference_times.append(t_inf_end - t_inf_start)
        
        loop_end = time.time()
        loop_times.append(loop_end - loop_start)
        
        # Update stats
        if len(loop_times) >= 10:
            avg_loop = sum(loop_times) / len(loop_times)
            avg_inf = sum(inference_times) / len(inference_times)
            
            with stats_lock:
                stats["loop_fps"] = 1.0 / avg_loop if avg_loop > 0 else 0
                stats["raw_inference_fps"] = num_devices / avg_inf if avg_inf > 0 else 0
                stats["total_frames"] += len(loop_times) * num_devices
            
            print(f"[8-Device] {stats['raw_inference_fps']:.0f} FPS | Inference: {1000*avg_inf:.1f}ms | Frames: {stats['total_frames']}", flush=True)
            
            loop_times.clear()
            inference_times.clear()
        
        # Debug: print detection count every 50 iterations
        if iteration % 50 == 1:
            print(f"[Debug] Iter {iteration}: total_detections={total_detections}", flush=True)


def inference_thread_mesh(video_frames, yunet_model, sface_model, mesh_device, inputs_mesh_mapper, output_mesh_composer, stop_event, max_frames=10000):
    """Run parallel inference using mesh device with batched input.
    
    YuNet runs on mesh (all 8 devices in parallel) - batch [8, 640, 640, 3].
    SFace runs on mesh (batch faces to 8) - batch [8, 112, 112, 3].
    """
    global frame_buffers, stats
    
    num_devices = mesh_device.get_num_devices()
    frame_indices = [0] * num_devices
    
    loop_times = []
    inference_times = []
    total_frames_processed = 0
    
    print(f"[Mesh Inference] Starting batched inference on {num_devices} device mesh...", flush=True)
    print(f"[Mesh Inference] Will auto-stop after {max_frames} frames", flush=True)
    
    iteration = 0
    while not stop_event.is_set() and total_frames_processed < max_frames:
        iteration += 1
        loop_start = time.time()
        t_inf_start = time.time()
        
        # Prepare batch of frames (one per device)
        batch_frames = []
        batch_frames_rgb = []
        for i in range(num_devices):
            frames = video_frames[i]
            idx = frame_indices[i] % len(frames)
            frame_indices[i] = idx + 1
            frame_rgb = frames[idx]
            batch_frames_rgb.append(frame_rgb)
            
            # Resize and normalize
            img_resized = cv2.resize(frame_rgb, (640, 640))
            # YuNet expects 0-255 range, NOT normalized 0-1
            tensor = torch.from_numpy(img_resized.astype(np.float32)).to(torch.bfloat16)
            batch_frames.append(tensor)
        
        # Stack into batch [8, H, W, C]
        batch_tensor = torch.stack(batch_frames, dim=0)
        
        # Send to mesh device (sharded across devices)
        tt_batch = ttnn.from_torch(
            batch_tensor, 
            dtype=ttnn.bfloat16, 
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=inputs_mesh_mapper
        )
        
        # Run YuNet on mesh (parallel on all devices)
        cls_out, box_out, obj_out, kpt_out = yunet_model(tt_batch)
        ttnn.synchronize_device(mesh_device)
        
        # Get outputs back to CPU (gathered from all devices)
        cls_cpu = [ttnn.to_torch(c, mesh_composer=output_mesh_composer) for c in cls_out]
        box_cpu = [ttnn.to_torch(b, mesh_composer=output_mesh_composer) for b in box_out]
        obj_cpu = [ttnn.to_torch(o, mesh_composer=output_mesh_composer) for o in obj_out]
        kpt_cpu = [ttnn.to_torch(k, mesh_composer=output_mesh_composer) for k in kpt_out]
        
        # Debug shapes on first few iterations
        if iteration <= 3:
            print(f"[Debug] Output shapes: cls={[c.shape for c in cls_cpu]}", flush=True)
            # Check scores
            for si, c in enumerate(cls_cpu):
                max_score = (c.sigmoid() * obj_cpu[si].sigmoid()).max().item()
                print(f"[Debug] Scale {si}: max_score={max_score:.4f}", flush=True)
        
        t_inf_end = time.time()
        inference_times.append(t_inf_end - t_inf_start)
        
        # Step 1: Decode detections and collect all faces from all 8 frames
        total_detections = 0
        all_frame_data = []  # [(frame_idx, detections, frame_rgb, frame_out), ...]
        all_faces = []  # [(frame_idx, det_idx, face_tensor, box_coords), ...]
        
        for i in range(num_devices):
            detections = decode_yunet_batched(cls_cpu, box_cpu, obj_cpu, kpt_cpu, 640, batch_idx=i)
            total_detections += len(detections)
            
            frame_rgb = batch_frames_rgb[i]
            frame_out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            h_orig, w_orig = frame_rgb.shape[:2]
            scale_x, scale_y = w_orig / 640, h_orig / 640
            
            all_frame_data.append((i, detections, frame_rgb, frame_out, h_orig, w_orig, scale_x, scale_y))
            
            # Collect face crops for SFace batching
            for det_idx, det in enumerate(detections[:10]):  # Limit to 10 faces per frame
                x1 = int(det["box"][0] * scale_x)
                y1 = int(det["box"][1] * scale_y)
                x2 = int(det["box"][2] * scale_x)
                y2 = int(det["box"][3] * scale_y)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_orig, x2), min(h_orig, y2)
                
                if x2 - x1 >= 50 and y2 - y1 >= 50:
                    # Try keypoint alignment first (more accurate)
                    keypoints = det.get("keypoints")
                    if keypoints and len(keypoints) >= 5:
                        # Scale keypoints to original image coordinates
                        scaled_kpts = [[kp[0] * scale_x, kp[1] * scale_y] for kp in keypoints]
                        face_aligned = align_face_keypoints(frame_rgb, scaled_kpts, target_size=112)
                    else:
                        # Fallback: simple crop + resize
                        w, h = x2 - x1, y2 - y1
                        margin = int(max(w, h) * 0.1)
                        fx1, fy1 = max(0, x1 - margin), max(0, y1 - margin)
                        fx2, fy2 = min(w_orig, x2 + margin), min(h_orig, y2 + margin)
                        face_crop = frame_rgb[fy1:fy2, fx1:fx2]
                        face_aligned = cv2.resize(face_crop, (112, 112))
                    
                    face_tensor = torch.from_numpy(face_aligned.astype(np.float32))
                    all_faces.append((i, det_idx, face_tensor, (x1, y1, x2, y2)))
        
        # Step 2: Batch faces into groups of 8 and run SFace on mesh
        embeddings_map = {}  # (frame_idx, det_idx) -> embedding
        
        # Debug face collection on first few iters
        if iteration <= 3:
            print(f"[Debug SFace] Collected {len(all_faces)} faces from {num_devices} frames", flush=True)
        
        if all_faces:
            # Process faces in batches of 8 on TTNN mesh
            for batch_start in range(0, len(all_faces), num_devices):
                batch_faces = all_faces[batch_start:batch_start + num_devices]
                
                # Pad batch to 8 if needed
                while len(batch_faces) < num_devices:
                    dummy = torch.zeros(112, 112, 3, dtype=torch.float32)
                    batch_faces.append((-1, -1, dummy, None))
                
                # Stack into batch [8, 112, 112, 3] NHWC for TTNN
                face_batch = torch.stack([f[2] for f in batch_faces], dim=0)
                
                # Run SFace on mesh - distribute across 8 devices
                tt_faces = ttnn.from_torch(face_batch, dtype=ttnn.bfloat16, 
                                           layout=ttnn.ROW_MAJOR_LAYOUT,
                                           mesh_mapper=inputs_mesh_mapper,
                                           device=mesh_device)
                tt_emb = sface_model(tt_faces)
                embeddings_batch = ttnn.to_torch(tt_emb, mesh_composer=output_mesh_composer).float().numpy()
                
                # Map embeddings back to faces
                for idx, (frame_idx, det_idx, _, _) in enumerate(batch_faces):
                    if frame_idx >= 0:  # Skip padding
                        emb = embeddings_batch[idx].flatten()
                        embeddings_map[(frame_idx, det_idx)] = emb
                        
                        # Debug on first batch
                        if iteration == 1 and batch_start == 0 and idx == 0:
                            print(f"[Debug SFace TTNN] Embedding norm: {np.linalg.norm(emb):.4f}", flush=True)
                            print(f"[Debug SFace TTNN] All matches:", flush=True)
                            for name, db_emb in face_database.items():
                                score = np.dot(emb, db_emb) / (np.linalg.norm(emb) * np.linalg.norm(db_emb) + 1e-8)
                                print(f"  - {name}: {score:.4f}", flush=True)
        
        # Step 3: Draw results on frames
        for i, detections, frame_rgb, frame_out, h_orig, w_orig, scale_x, scale_y in all_frame_data:
            for det_idx, det in enumerate(detections[:10]):
                x1 = int(det["box"][0] * scale_x)
                y1 = int(det["box"][1] * scale_y)
                x2 = int(det["box"][2] * scale_x)
                y2 = int(det["box"][3] * scale_y)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_orig, x2), min(h_orig, y2)
                
                identity = "Unknown"
                score = 0.0
                
                # Get embedding if available
                if (i, det_idx) in embeddings_map:
                    embedding = embeddings_map[(i, det_idx)]
                    identity, score = match_face(embedding)
                
                # Draw box: green for recognized, red for unknown
                is_recognized = identity != "Unknown"
                color = (0, 255, 0) if is_recognized else (0, 0, 255)
                cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)
                
                if is_recognized:
                    label = f"{identity}: {int(score*100)}%"
                    cv2.putText(frame_out, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Device label
            cv2.putText(frame_out, f"Device {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, DEVICE_COLORS[i], 2)
            cv2.putText(frame_out, f"Faces: {len(detections)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            with frame_locks[i]:
                frame_buffers[i] = frame_out
        
        # Debug: print detection count periodically
        if iteration % 30 == 1:
            with stats_lock:
                stats["faces_detected"] = total_detections
            print(f"[Debug MESH] Iter {iteration}: detections={total_detections} across {num_devices} devices", flush=True)
        
        loop_end = time.time()
        loop_times.append(loop_end - loop_start)
        
        # Update stats
        if len(loop_times) >= 10:
            avg_loop = sum(loop_times) / len(loop_times)
            avg_inf = sum(inference_times) / len(inference_times)
            
            with stats_lock:
                stats["loop_fps"] = 1.0 / avg_loop if avg_loop > 0 else 0
                stats["raw_inference_fps"] = num_devices / avg_inf if avg_inf > 0 else 0
                stats["total_frames"] += len(loop_times) * num_devices
            
            print(f"[8-Device MESH] {stats['raw_inference_fps']:.0f} FPS | Inference: {1000*avg_inf:.1f}ms | Frames: {stats['total_frames']}", flush=True)
            
            loop_times.clear()
            inference_times.clear()
        
        total_frames_processed += num_devices
    
    # Update final stats (flush remaining)
    with stats_lock:
        stats["total_frames"] = total_frames_processed
    
    # Done - print final stats
    print(f"\n[Mesh Inference] COMPLETE - Processed {total_frames_processed} frames", flush=True)
    print(f"[Mesh Inference] Final FPS: {stats['raw_inference_fps']:.0f}", flush=True)
    stop_event.set()  # Signal main thread to exit


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def nms_detections(detections, iou_threshold=0.4):
    """Apply Non-Maximum Suppression to detections."""
    if not detections:
        return []
    
    # Sort by score descending
    detections = sorted(detections, key=lambda x: x["score"], reverse=True)
    
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        
        # Remove overlapping detections
        detections = [d for d in detections if compute_iou(best["box"], d["box"]) < iou_threshold]
    
    return keep


def decode_yunet_batched(cls_outs, box_outs, obj_outs, kpt_outs, input_size, batch_idx=0, threshold=0.35):
    """Decode YuNet outputs for a specific batch index (from mesh output)."""
    STRIDES = [8, 16, 32]
    detections = []
    
    for scale_idx in range(3):
        # cls_outs are already torch tensors from mesh_composer
        cls_out = cls_outs[scale_idx].float()
        box_out = box_outs[scale_idx].float()
        obj_out = obj_outs[scale_idx].float()
        kpt_out = kpt_outs[scale_idx].float()
        
        # Handle NHWC format - permute to NCHW
        if cls_out.dim() == 4 and cls_out.shape[-1] < cls_out.shape[1]:
            cls_out = cls_out.permute(0, 3, 1, 2)
            box_out = box_out.permute(0, 3, 1, 2)
            obj_out = obj_out.permute(0, 3, 1, 2)
            kpt_out = kpt_out.permute(0, 3, 1, 2)
        
        stride = STRIDES[scale_idx]
        score = cls_out[batch_idx:batch_idx+1].sigmoid() * obj_out[batch_idx:batch_idx+1].sigmoid()
        
        high_conf = score > threshold
        if high_conf.any():
            indices = torch.where(high_conf)
            for i in range(min(len(indices[0]), 20)):
                b, c, h, w = indices[0][i], indices[1][i], indices[2][i], indices[3][i]
                conf = score[0, c, h, w].item()
                anchor_x, anchor_y = w.item() * stride, h.item() * stride
                
                dx = box_out[batch_idx, 0, h, w].item()
                dy = box_out[batch_idx, 1, h, w].item()
                dw = box_out[batch_idx, 2, h, w].item()
                dh = box_out[batch_idx, 3, h, w].item()
                
                cx, cy = dx * stride + anchor_x, dy * stride + anchor_y
                bw, bh = np.exp(min(dw, 10)) * stride, np.exp(min(dh, 10)) * stride
                
                x1, y1 = cx - bw / 2, cy - bh / 2
                x2, y2 = cx + bw / 2, cy + bh / 2
                
                # Extract 5 keypoints (10 channels: x0,y0, x1,y1, ...)
                keypoints = []
                for k in range(5):
                    kpt_dx = kpt_out[batch_idx, k * 2, h, w].item()
                    kpt_dy = kpt_out[batch_idx, k * 2 + 1, h, w].item()
                    kx = kpt_dx * stride + anchor_x
                    ky = kpt_dy * stride + anchor_y
                    keypoints.append([kx, ky])
                
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "score": conf,
                    "keypoints": keypoints,
                })
    
    # Apply NMS to remove duplicate detections across scales
    return nms_detections(detections, iou_threshold=0.4)


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
    <h1>8-Device Parallel Face Recognition</h1>
    <p class="subtitle">YuNet + SFace running on 8 Tenstorrent devices in parallel</p>
    
    <div class="stats-bar">
        <div class="stat">
            <div class="stat-value" id="raw-fps">-</div>
            <div class="stat-label">Throughput (FPS)</div>
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
    parser.add_argument("--max-frames", type=int, default=1024, help="Auto-stop after N frames (default: 1024, divisible by 8)")
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
    
    # Get available devices
    device_ids = ttnn.get_device_ids()
    num_devices = len(device_ids)
    print(f"[Parallel] Found {num_devices} TT devices: {device_ids}")
    
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
    
    # Mesh mappers for data parallelism (same as yolov8)
    def get_mesh_mappers(device):
        if device.get_num_devices() > 1:
            inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
            weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device)
            output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        else:
            inputs_mesh_mapper = None
            weights_mesh_mapper = None
            output_mesh_composer = None
        return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer
    
    l1_size = max(YUNET_L1_SMALL_SIZE, SFACE_L1_SMALL_SIZE)
    
    # Pre-load videos
    video_frames = load_videos_into_ram(video_paths, num_devices=num_devices)
    init_frame_buffers(num_devices)
    
    # Pre-load torch models once
    print("[Parallel] Loading YuNet torch model...", flush=True)
    yunet_weights = get_yunet_weights()
    yunet_torch = load_yunet_torch(yunet_weights).to(torch.bfloat16)
    print("[Parallel] YuNet torch loaded!", flush=True)
    
    print("[Parallel] Loading SFace torch model...", flush=True)
    sface_onnx = get_sface_onnx_path()
    sface_torch = load_sface_from_onnx(sface_onnx)
    sface_torch.eval()
    print("[Parallel] SFace torch loaded!", flush=True)
    
    # Open mesh device for YuNet (8 devices for parallel detection)
    print(f"[Parallel] Opening mesh device with shape (1, {num_devices})...", flush=True)
    device_params = {
        "l1_small_size": l1_size,
        "trace_region_size": 6434816,
        "num_command_queues": 2,
    }
    mesh_shape = ttnn.MeshShape(1, num_devices)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
    mesh_device.enable_program_cache()
    print(f"[Parallel] Mesh device opened with {mesh_device.get_num_devices()} devices", flush=True)
    
    # Get mesh mappers for data parallel YuNet inference
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    
    # Create YuNet on mesh device (weights replicated to all devices)
    print(f"[Parallel] Creating YuNet model on mesh...", flush=True)
    yunet_model = create_yunet_model(mesh_device, yunet_torch, weights_mesh_mapper=weights_mesh_mapper)
    
    # Create SFace on mesh device (will batch faces to 8 and run in parallel)
    print(f"[Parallel] Creating SFace model on mesh...", flush=True)
    sface_model = create_sface_model(mesh_device, sface_torch, weights_mesh_mapper=weights_mesh_mapper)
    print(f"[Parallel] SFace model created!", flush=True)
    
    # Warmup YuNet on mesh
    print(f"[Parallel] Running YuNet warmup inference...", flush=True)
    dummy_yunet = torch.zeros(num_devices, 640, 640, 3, dtype=torch.bfloat16)
    tt_dummy = ttnn.from_torch(dummy_yunet, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                device=mesh_device, mesh_mapper=inputs_mesh_mapper)
    _ = yunet_model(tt_dummy)
    ttnn.synchronize_device(mesh_device)
    print(f"[Parallel] YuNet warmup done!", flush=True)
    
    # Warmup SFace on mesh (batch of 8 faces)
    print(f"[Parallel] Running SFace warmup inference...", flush=True)
    dummy_sface = torch.zeros(num_devices, 112, 112, 3, dtype=torch.float32)
    tt_sface_dummy = ttnn.from_torch(dummy_sface, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                      device=mesh_device, mesh_mapper=inputs_mesh_mapper)
    _ = sface_model(tt_sface_dummy)
    ttnn.synchronize_device(mesh_device)
    print(f"[Parallel] SFace warmup done!", flush=True)
    
    # Load registered faces for recognition
    load_faces_from_disk()
    
    print(f"[Parallel] All {num_devices} devices ready!", flush=True)
    
    # Start inference thread with mesh device (YuNet + SFace on TTNN mesh)
    stop_event = threading.Event()
    inf_thread = threading.Thread(
        target=inference_thread_mesh, 
        args=(video_frames, yunet_model, sface_model, mesh_device, inputs_mesh_mapper, output_mesh_composer, stop_event, args.max_frames),
    )
    inf_thread.start()
    
    try:
        inf_thread.join()  # Wait for thread to complete (auto-stops after max_frames)
    except KeyboardInterrupt:
        print("\n[Parallel] Interrupted by user...", flush=True)
        stop_event.set()
        inf_thread.join(timeout=2)
    
    print("[Parallel] Shutting down cleanly...", flush=True)
    
    # Stop HTTP server first (before closing device to avoid crashes)
    try:
        http_server.shutdown()
    except Exception:
        pass
    
    # Small delay to let HTTP threads finish
    time.sleep(0.5)
    
    # Now safe to close device
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    print("[Parallel] Done!", flush=True)
    
    # Force clean exit
    os._exit(0)


if __name__ == "__main__":
    main()
