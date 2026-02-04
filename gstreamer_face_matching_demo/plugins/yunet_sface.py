#!/usr/bin/env python3
"""
GStreamer YuNet+SFace Face Recognition Plugin for Tenstorrent Devices.

Pipeline:
1. YuNet - Face detection (detects faces and keypoints)
2. SFace - Face embedding extraction (for recognition/matching)
"""
import os
import sys
import gi
import time
import numpy as np
import cv2
import torch
from pathlib import Path

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject

sys.stdout = sys.stderr

import ttnn

# Initialize GStreamer
Gst.init(None)

# Face database for recognition
face_database = {}  # name -> embedding
FACES_DIR = Path("/app/registered_faces")


def load_faces_from_disk():
    """Load saved faces from disk."""
    global face_database
    if not FACES_DIR.exists():
        print(f"[FaceRecognition] No faces directory at {FACES_DIR}", flush=True)
        return
    
    for person_dir in FACES_DIR.iterdir():
        if person_dir.is_dir():
            name = person_dir.name
            embedding_path = person_dir / "embedding.npy"
            if embedding_path.exists():
                face_database[name] = np.load(embedding_path)
                print(f"[FaceRecognition] Loaded face: {name}", flush=True)
    print(f"[FaceRecognition] Loaded {len(face_database)} faces from disk", flush=True)


def process_pending_faces(sface_model, device):
    """Process faces that were registered without embeddings (pending_embedding flag)."""
    global face_database
    import cv2
    
    if not FACES_DIR.exists():
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
                        # Load face image
                        face_img = cv2.imread(str(face_path))
                        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_resized = cv2.resize(face_rgb, (112, 112))
                        
                        # Run SFace to get embedding
                        input_tensor = torch.tensor(face_resized, dtype=torch.float32).unsqueeze(0)
                        input_tt = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                        output_tt = sface_model(input_tt)
                        ttnn.synchronize_device(device)
                        embedding = ttnn.to_torch(output_tt).squeeze().numpy()
                        
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


class FaceRecognition(Gst.Element):
    """GStreamer element for Face Recognition on Tenstorrent devices."""

    __gtype_name__ = "GstFaceRecognitionTT"

    __gstmetadata__ = (
        "Face Recognition Tenstorrent",
        "Filter/Effect/Video",
        "YuNet+SFace face recognition on Tenstorrent hardware",
        "Tenstorrent",
    )

    _sink_template = Gst.PadTemplate.new(
        "sink",
        Gst.PadDirection.SINK,
        Gst.PadPresence.ALWAYS,
        Gst.Caps.from_string("video/x-raw,format=BGRx,width=640,height=640"),
    )
    _src_template = Gst.PadTemplate.new(
        "src",
        Gst.PadDirection.SRC,
        Gst.PadPresence.ALWAYS,
        Gst.Caps.from_string("video/x-raw,format=BGRx,width=640,height=640"),
    )
    __gsttemplates__ = (_sink_template, _src_template)

    __gproperties__ = {
        "device-id": (int, "Device ID", "TT Device ID", 0, 7, 0, GObject.ParamFlags.READWRITE),
        "detection-threshold": (float, "Detection Threshold", "Face detection confidence threshold", 0.0, 1.0, 0.35, GObject.ParamFlags.READWRITE),
        "recognition-threshold": (float, "Recognition Threshold", "Face recognition similarity threshold", 0.0, 1.0, 0.5, GObject.ParamFlags.READWRITE),
    }

    def __init__(self):
        super().__init__()
        print("[FaceRecognition] __init__ called", flush=True)
        
        self.device_id = 0
        self.detection_threshold = 0.35  # Match YuNet default (detects more small faces)
        self.recognition_threshold = 0.5
        
        self.yunet_model = None
        self.sface_model = None
        self.device = None
        self.frame_count = 0
        self.total_inference_time = 0
        
        # Temporal smoothing to reduce flickering
        self.prev_results = []  # Previous frame's results for smoothing
        
        # Create pads
        self.sinkpad = Gst.Pad.new_from_template(self._sink_template, "sink")
        self.sinkpad.set_chain_function(self._chain)
        self.sinkpad.set_event_function(self._sink_event)
        self.add_pad(self.sinkpad)
        
        self.srcpad = Gst.Pad.new_from_template(self._src_template, "src")
        self.add_pad(self.srcpad)
        
        print("[FaceRecognition] Pads created", flush=True)

    def _initialize_device(self):
        """Initialize TT device and load YuNet + SFace models."""
        # Models are at tt-metal/models/experimental/yunet and tt-metal/models/experimental/sface
        from models.experimental.yunet.common import (
            YUNET_L1_SMALL_SIZE,
            load_torch_model as load_yunet_torch,
            get_default_weights_path as get_yunet_weights,
        )
        from models.experimental.yunet.tt.ttnn_yunet import create_yunet_model
        from models.experimental.sface.common import get_sface_onnx_path, SFACE_L1_SMALL_SIZE
        from models.experimental.sface.reference.sface_model import load_sface_from_onnx
        from models.experimental.sface.tt.ttnn_sface import create_sface_model
        import ttnn.distributed as dist
        
        print(f"[FaceRecognition] Initializing device {self.device_id}...", flush=True)
        
        # Use larger L1 size for both models
        l1_size = max(YUNET_L1_SMALL_SIZE, SFACE_L1_SMALL_SIZE)
        
        # Use 1x1 mesh with specific physical device to avoid T3K L1 clash issues
        # This makes each device work like a standalone n150
        self.mesh = dist.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 1),
            physical_device_ids=[self.device_id],
            l1_small_size=l1_size,
            trace_region_size=0
        )
        self.device = self.mesh  # MeshDevice can be used like a device
        self.device.enable_program_cache()
        
        # Load YuNet
        print("[FaceRecognition] Loading YuNet model...", flush=True)
        yunet_weights = get_yunet_weights()
        yunet_torch = load_yunet_torch(yunet_weights)
        yunet_torch = yunet_torch.to(torch.bfloat16)
        self.yunet_model = create_yunet_model(self.device, yunet_torch)
        print("[FaceRecognition] YuNet loaded!", flush=True)
        
        # Load SFace
        print("[FaceRecognition] Loading SFace model...", flush=True)
        sface_onnx = get_sface_onnx_path()
        sface_torch = load_sface_from_onnx(sface_onnx)
        sface_torch.eval()
        self.sface_model = create_sface_model(self.device, sface_torch)
        print("[FaceRecognition] SFace loaded!", flush=True)
        
        # Warmup
        print("[FaceRecognition] Warming up models...", flush=True)
        self._warmup()
        print("[FaceRecognition] Ready!", flush=True)

    def _warmup(self):
        """Warmup both models."""
        # Warmup YuNet - use 640x640 to match inference shape
        for _ in range(2):
            warmup = torch.randint(0, 256, (1, 640, 640, 3), dtype=torch.float32).to(torch.bfloat16)
            warmup_tt = ttnn.from_torch(warmup, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
            _ = self.yunet_model(warmup_tt)
            ttnn.synchronize_device(self.device)
        
        # Warmup SFace
        for _ in range(2):
            warmup = torch.randint(0, 256, (1, 112, 112, 3), dtype=torch.float32)
            warmup_tt = ttnn.from_torch(warmup, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
            _ = self.sface_model(warmup_tt)
            ttnn.synchronize_device(self.device)
        
        # Load registered faces from disk
        load_faces_from_disk()
        
        # Process any pending faces that were registered without embeddings
        process_pending_faces(self.sface_model, self.device)
        
        print("[FaceRecognition] Ready!", flush=True)

    def _sink_event(self, pad, parent, event):
        """Handle sink pad events."""
        if event.type == Gst.EventType.CAPS:
            caps = event.parse_caps()
            print(f"[FaceRecognition] Got caps: {caps.to_string()}", flush=True)
            return self.srcpad.push_event(event)
        return self.srcpad.push_event(event)

    def _chain(self, pad, parent, buf):
        """Process each frame."""
        # Lazy init on first frame
        if self.yunet_model is None:
            self._initialize_device()

        try:
            t_total_start = time.time()
            
            # 1. Read input frame (preprocessing)
            t_preprocess_start = time.time()
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR

            frame_bgrx = np.frombuffer(map_info.data, dtype=np.uint8).reshape(640, 640, 4)
            frame_bgr = frame_bgrx[:, :, :3].copy()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            buf.unmap(map_info)
            t_preprocess_end = time.time()

            # 2. Face Detection (YuNet)
            t_yunet_start = time.time()
            detections = self._detect_faces(frame_rgb)
            t_yunet_end = time.time()

            # 3. Face Recognition (SFace) for each detected face
            t_sface_start = time.time()
            results = []
            sface_times = []
            for det in detections:
                t_face_start = time.time()
                face_crop = self._crop_and_align_face(frame_rgb, det)
                if face_crop is not None:
                    embedding = self._get_embedding(face_crop)
                    identity, score = self._match_face(embedding)
                    
                    # Temporal smoothing: check if similar face was recognized in previous frame
                    # This reduces flickering when recognition is unstable
                    for prev in self.prev_results:
                        if self._box_iou(det["box"], prev["box"]) > 0.5:
                            # Same face location - use previous identity if current is Unknown
                            if identity == "Unknown" and prev["identity"] != "Unknown":
                                identity = prev["identity"]
                                score = prev["score"] * 0.9  # Decay score slightly
                            break
                    
                    results.append({
                        "box": det["box"],
                        "keypoints": det.get("keypoints"),
                        "identity": identity,
                        "score": score,
                        "conf": det["conf"],
                    })
                    sface_times.append((time.time() - t_face_start) * 1000)
            t_sface_end = time.time()
            
            # Store for next frame's temporal smoothing
            self.prev_results = results

            # 4. Draw results (postprocessing)
            t_draw_start = time.time()
            
            # Calculate timing breakdown
            timing = {
                "preprocess_ms": (t_preprocess_end - t_preprocess_start) * 1000,
                "yunet_ms": (t_yunet_end - t_yunet_start) * 1000,
                "sface_ms": (t_sface_end - t_sface_start) * 1000,
                "sface_per_face_ms": sum(sface_times) / len(sface_times) if sface_times else 0,
                "num_faces": len(results),
            }
            
            out_image = self._draw_results(results, frame_bgr, timing)
            t_draw_end = time.time()
            
            timing["draw_ms"] = (t_draw_end - t_draw_start) * 1000
            timing["total_ms"] = (t_draw_end - t_total_start) * 1000

            self.frame_count += 1
            self.total_inference_time += timing["total_ms"]

            # Print timing every frame (first 10) then every 10th frame
            if self.frame_count <= 10 or self.frame_count % 10 == 0:
                avg_ms = self.total_inference_time / self.frame_count
                fps = 1000 / avg_ms if avg_ms > 0 else 0
                
                # Print header once
                if self.frame_count == 1:
                    print("\n" + "="*80, flush=True)
                    print("  GSTREAMER PIPELINE - REAL-TIME LATENCY (per frame)", flush=True)
                    print("="*80, flush=True)
                    print(f"{'Frame':<8} {'Decode':<10} {'YuNet':<10} {'SFace':<12} {'Encode':<10} {'TOTAL':<10} {'FPS':<6}", flush=True)
                    print(f"{'':8} {'(est.)':10} {'(TTNN)':10} {'(TTNN)':12} {'(est.)':10} {'':10} {'':<6}", flush=True)
                    print("-"*80, flush=True)
                
                # Estimated decode/encode times (measured separately)
                decode_est = 2.0  # ~2ms for MPEG4/H264 decode
                encode_est = 3.0  # ~3ms for JPEG encode
                total_pipeline = decode_est + timing['total_ms'] + encode_est
                
                # Single line per frame - easy to read
                sface_str = f"{timing['sface_ms']:.1f}({timing['num_faces']})"
                print(f"{self.frame_count:<8} {decode_est:<10.1f} {timing['yunet_ms']:<10.1f} {sface_str:<12} {encode_est:<10.1f} {total_pipeline:<10.1f} {fps:<6.0f}", flush=True)

            # 5. Convert BGR back to BGRx (add alpha channel)
            out_bgrx = cv2.cvtColor(out_image, cv2.COLOR_BGR2BGRA)
            
            # 6. Create output buffer
            out_buf = Gst.Buffer.new_allocate(None, out_bgrx.nbytes, None)
            out_buf.fill(0, out_bgrx.tobytes())
            out_buf.pts = buf.pts
            out_buf.dts = buf.dts
            out_buf.duration = buf.duration

            return self.srcpad.push(out_buf)

        except Exception as e:
            print(f"[FaceRecognition] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return Gst.FlowReturn.ERROR

    def _detect_faces(self, frame_rgb):
        """Run YuNet face detection."""
        from models.experimental.yunet.common import STRIDES, DEFAULT_NMS_IOU_THRESHOLD
        
        h, w = frame_rgb.shape[:2]
        
        # Prepare input - resize to 640x640 for YuNet
        img_resized = cv2.resize(frame_rgb, (640, 640))
        tensor = torch.from_numpy(img_resized).float().unsqueeze(0).to(torch.bfloat16)
        
        tt_input = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        cls_out, box_out, obj_out, kpt_out = self.yunet_model(tt_input)
        ttnn.synchronize_device(self.device)
        
        # Decode detections
        detections = self._decode_yunet_output(cls_out, box_out, obj_out, kpt_out, 640, self.detection_threshold)
        
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

    def _decode_yunet_output(self, cls_outs, box_outs, obj_outs, kpt_outs, input_size, threshold):
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

    def _box_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / max(union, 1e-6)

    def _crop_and_align_face(self, image, detection, target_size=112):
        """Crop and align face for SFace using keypoint alignment."""
        x1, y1, x2, y2 = map(int, detection["box"])
        
        # Skip very small faces (< 50px for reliable recognition)
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            return None
        
        # Try keypoint alignment first (matches registration in GUI)
        keypoints = detection.get("keypoints")
        if keypoints and len(keypoints) >= 5:
            # Use affine transform based on eye and nose positions
            left_eye = np.array(keypoints[0], dtype=np.float32)
            right_eye = np.array(keypoints[1], dtype=np.float32)
            nose = np.array(keypoints[2], dtype=np.float32)
            
            src_pts = np.float32([left_eye, right_eye, nose])
            
            # Standard destination points for 112x112 aligned face (SFace/ArcFace)
            dst_pts = np.float32([
                [38.2946, 51.6963],  # left eye
                [73.5318, 51.5014],  # right eye
                [56.0252, 71.7366],  # nose
            ])
            
            M = cv2.getAffineTransform(src_pts, dst_pts)
            face_aligned = cv2.warpAffine(image, M, (target_size, target_size), borderMode=cv2.BORDER_REPLICATE)
            return face_aligned
        
        # Fallback: simple crop + resize (if no keypoints)
        w, h = x2 - x1, y2 - y1
        margin = int(max(w, h) * 0.1)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.shape[1], x2 + margin)
        y2 = min(image.shape[0], y2 + margin)
        
        face_crop = image[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (target_size, target_size))
        return face_resized

    def _get_embedding(self, face_image):
        """Extract face embedding using SFace."""
        face_tensor = torch.from_numpy(face_image.astype(np.float32)).unsqueeze(0)
        tt_input = ttnn.from_torch(face_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        tt_output = self.sface_model(tt_input)
        embedding = ttnn.to_torch(tt_output).float().numpy().flatten()
        return embedding

    def _match_face(self, embedding):
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
        
        if best_score >= self.recognition_threshold:
            return best_match, best_score
        return "Unknown", best_score

    def _draw_results(self, results, image, timing):
        """Draw face boxes, identities, and simple timing overlay."""
        for r in results:
            x1, y1, x2, y2 = map(int, r["box"])
            identity = r["identity"]
            score = r["score"]
            
            # Color: green for known, red for unknown
            is_recognized = identity != "Unknown"
            color = (0, 255, 0) if is_recognized else (0, 0, 255)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Only show label for recognized faces (green)
            if is_recognized:
                label = f"{identity}: {int(score*100)}%"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Unknown faces: just red box, no text
            
            # Draw keypoints
            if r.get("keypoints"):
                for kp in r["keypoints"]:
                    cv2.circle(image, (int(kp[0]), int(kp[1])), 2, (0, 255, 255), -1)
        
        # Simple timing overlay - only inference and FPS
        inference_ms = timing['yunet_ms'] + timing['sface_ms']
        avg_ms = self.total_inference_time / max(self.frame_count, 1)
        fps = 1000 / avg_ms if avg_ms > 0 else 0
        
        # Small overlay at top-left
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (200, 55), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        
        # Only show inference time and FPS
        cv2.putText(image, f"Inference: {inference_ms:.1f} ms", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f"FPS: {fps:.0f}", (10, 48), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    def do_get_property(self, prop):
        if prop.name == "device-id":
            return self.device_id
        elif prop.name == "detection-threshold":
            return self.detection_threshold
        elif prop.name == "recognition-threshold":
            return self.recognition_threshold
        raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop, value):
        if prop.name == "device-id":
            self.device_id = value
        elif prop.name == "detection-threshold":
            self.detection_threshold = value
        elif prop.name == "recognition-threshold":
            self.recognition_threshold = value
        else:
            raise AttributeError(f"Unknown property {prop.name}")


# Register plugin
GObject.type_register(FaceRecognition)
__gstelementfactory__ = ("face_recognition", Gst.Rank.NONE, FaceRecognition)
