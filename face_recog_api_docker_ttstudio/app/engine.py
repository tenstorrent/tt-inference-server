# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
YuNet + SFace face pipeline (same logic as gstreamer_face_matching_demo/websocket_server.py).
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("LOGURU_LEVEL", "WARNING")

import ttnn


class FaceEngine:
    """Loads YuNet/SFace, maintains in-memory + on-disk face gallery."""

    def __init__(self, faces_dir: Path, device_id: int = 0, match_threshold: float = 0.5):
        self.faces_dir = Path(faces_dir)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.device_id = device_id
        self.match_threshold = match_threshold
        self._lock = threading.Lock()
        self.yunet_model = None
        self.sface_model = None
        self.device = None
        self.face_database: Dict[str, np.ndarray] = {}
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def initialize(self) -> None:
        with self._lock:
            if self._ready:
                return
            self._init_models_locked()
            self._load_faces_from_disk()
            self._process_pending_faces()
            self._ready = True

    def _init_models_locked(self) -> None:
        import ttnn.distributed as dist

        from models.experimental.yunet.common import (
            YUNET_L1_SMALL_SIZE,
            get_default_weights_path as get_yunet_weights,
            load_torch_model as load_yunet_torch,
        )
        from models.experimental.yunet.tt.ttnn_yunet import create_yunet_model
        from models.experimental.sface.common import SFACE_L1_SMALL_SIZE, get_sface_onnx_path
        from models.experimental.sface.reference.sface_model import load_sface_from_onnx
        from models.experimental.sface.tt.ttnn_sface import create_sface_model

        print(f"[FaceRecognition] Opening device {self.device_id}...", flush=True)
        l1_size = max(YUNET_L1_SMALL_SIZE, SFACE_L1_SMALL_SIZE)
        mesh = dist.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 1),
            physical_device_ids=[self.device_id],
            l1_small_size=l1_size,
            trace_region_size=0,
        )
        self.device = mesh
        self.device.enable_program_cache()
        print("[FaceRecognition] Device opened!", flush=True)

        print("[FaceRecognition] Loading YuNet weights...", flush=True)
        yunet_torch = load_yunet_torch(get_yunet_weights()).to(torch.bfloat16)
        print("[FaceRecognition] Creating YuNet TTNN model...", flush=True)
        self.yunet_model = create_yunet_model(self.device, yunet_torch)
        print("[FaceRecognition] YuNet loaded!", flush=True)

        print("[FaceRecognition] Loading SFace weights...", flush=True)
        sface_torch = load_sface_from_onnx(get_sface_onnx_path())
        sface_torch.eval()
        print("[FaceRecognition] Creating SFace TTNN model...", flush=True)
        self.sface_model = create_sface_model(self.device, sface_torch)
        print("[FaceRecognition] SFace loaded!", flush=True)

        print("[FaceRecognition] Warming up YuNet...", flush=True)
        w = torch.randint(0, 256, (1, 640, 640, 3), dtype=torch.float32).to(torch.bfloat16)
        tt_w = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        _ = self.yunet_model(tt_w)
        ttnn.synchronize_device(self.device)

        print("[FaceRecognition] Warming up SFace...", flush=True)
        w2 = torch.randint(0, 256, (1, 112, 112, 3), dtype=torch.float32)
        tt_w2 = ttnn.from_torch(w2, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        _ = self.sface_model(tt_w2)
        ttnn.synchronize_device(self.device)

        print("[FaceRecognition] Ready!", flush=True)

    def shutdown(self) -> None:
        with self._lock:
            if self.device is not None:
                try:
                    ttnn.close_device(self.device)
                except Exception:
                    pass
            self.device = None
            self.yunet_model = None
            self.sface_model = None
            self._ready = False

    def _load_faces_from_disk(self) -> None:
        if not self.faces_dir.exists():
            return
        for person_dir in self.faces_dir.iterdir():
            if person_dir.is_dir():
                emb = person_dir / "embedding.npy"
                if emb.exists():
                    self.face_database[person_dir.name] = np.load(emb)

    def _process_pending_faces(self) -> None:
        if not self.faces_dir.exists() or self.sface_model is None:
            return
        for person_dir in self.faces_dir.iterdir():
            if not person_dir.is_dir():
                continue
            pending = person_dir / "pending_embedding"
            if not pending.exists():
                continue
            face_path = person_dir / "face.jpg"
            if not face_path.exists():
                continue
            try:
                face_img = cv2.imread(str(face_path))
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                if face_rgb.shape[:2] != (112, 112):
                    face_rgb = cv2.resize(face_rgb, (112, 112))
                embedding = self._get_embedding_sync(face_rgb)
                np.save(person_dir / "embedding.npy", embedding)
                self.face_database[person_dir.name] = embedding
                pending.unlink()
            except Exception:
                pass

    def _save_face_to_disk(self, name: str, embedding: np.ndarray, face_image_rgb: np.ndarray) -> None:
        person_dir = self.faces_dir / name
        person_dir.mkdir(exist_ok=True)
        np.save(person_dir / "embedding.npy", embedding)
        cv2.imwrite(str(person_dir / "face.jpg"), cv2.cvtColor(face_image_rgb, cv2.COLOR_RGB2BGR))

    def list_identities(self) -> List[str]:
        return sorted(self.face_database.keys())

    def delete_identity(self, name: str) -> bool:
        with self._lock:
            person_dir = self.faces_dir / name
            on_disk = person_dir.exists()
            if name in self.face_database:
                del self.face_database[name]
            if on_disk:
                shutil.rmtree(person_dir)
                return True
            return False

    def _decode_yunet_output(
        self,
        cls_outs: Any,
        box_outs: Any,
        obj_outs: Any,
        kpt_outs: Any,
        threshold: float = 0.35,
    ) -> List[Dict]:
        from models.experimental.yunet.common import DEFAULT_NMS_IOU_THRESHOLD, STRIDES

        detections: List[Dict] = []
        for scale_idx in range(3):
            cls_out = ttnn.to_torch(cls_outs[scale_idx]).float().permute(0, 3, 1, 2)
            box_out = ttnn.to_torch(box_outs[scale_idx]).float().permute(0, 3, 1, 2)
            obj_out = ttnn.to_torch(obj_outs[scale_idx]).float().permute(0, 3, 1, 2)
            kpt_out = ttnn.to_torch(kpt_outs[scale_idx]).float().permute(0, 3, 1, 2)
            stride = STRIDES[scale_idx]
            score = cls_out.sigmoid() * obj_out.sigmoid()
            high_conf = score > threshold
            if not high_conf.any():
                continue
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

        detections.sort(key=lambda x: x["conf"], reverse=True)
        keep: List[Dict] = []
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

    def _detect_faces_sync(self, frame_rgb: np.ndarray) -> List[Dict]:
        h, w = frame_rgb.shape[:2]
        img_resized = cv2.resize(frame_rgb, (640, 640))
        tensor = torch.from_numpy(img_resized.astype(np.float32)).unsqueeze(0).to(torch.bfloat16)
        tt_input = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        cls_out, box_out, obj_out, kpt_out = self.yunet_model(tt_input)
        ttnn.synchronize_device(self.device)
        detections = self._decode_yunet_output(cls_out, box_out, obj_out, kpt_out, 0.35)
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

    def _get_embedding_sync(self, face_image: np.ndarray) -> np.ndarray:
        face_tensor = torch.from_numpy(face_image.astype(np.float32)).unsqueeze(0)
        tt_input = ttnn.from_torch(face_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        tt_output = self.sface_model(tt_input)
        return ttnn.to_torch(tt_output).float().numpy().flatten()

    @staticmethod
    def _align_face_keypoints(image: np.ndarray, keypoints: List, target_size: int = 112) -> np.ndarray:
        if len(keypoints) < 5:
            return cv2.resize(image, (target_size, target_size))
        left_eye = np.array(keypoints[0], dtype=np.float32)
        right_eye = np.array(keypoints[1], dtype=np.float32)
        nose = np.array(keypoints[2], dtype=np.float32)
        src_pts = np.float32([left_eye, right_eye, nose])
        dst_pts = np.float32(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
            ]
        )
        m = cv2.getAffineTransform(src_pts, dst_pts)
        return cv2.warpAffine(image, m, (target_size, target_size), borderMode=cv2.BORDER_REPLICATE)

    def _match_sync(self, embedding: np.ndarray) -> Tuple[str, float]:
        if not self.face_database:
            return "Unknown", 0.0
        best_name = "Unknown"
        best_score = 0.0
        for name, db_emb in self.face_database.items():
            score = float(
                np.dot(embedding, db_emb)
                / (np.linalg.norm(embedding) * np.linalg.norm(db_emb) + 1e-8)
            )
            if score > best_score:
                best_score = score
                best_name = name
        if best_score >= self.match_threshold:
            return best_name, best_score
        return "Unknown", best_score

    def recognize_image_rgb(self, frame_rgb: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """Returns list of face dicts and wall time in ms."""
        t0 = time.time()
        with self._lock:
            detections = self._detect_faces_sync(frame_rgb)
            results = []
            for det in detections:
                x1, y1, x2, y2 = map(int, det["box"])
                face_w, face_h = x2 - x1, y2 - y1
                if face_w < 30 or face_h < 30:
                    continue
                kps = det.get("keypoints")
                if kps and len(kps) >= 5:
                    aligned = self._align_face_keypoints(frame_rgb, kps, 112)
                else:
                    margin = int(max(face_w, face_h) * 0.1)
                    xa, ya = max(0, x1 - margin), max(0, y1 - margin)
                    xb, yb = min(frame_rgb.shape[1], x2 + margin), min(frame_rgb.shape[0], y2 + margin)
                    crop = frame_rgb[ya:yb, xa:xb]
                    aligned = cv2.resize(crop, (112, 112))
                emb = self._get_embedding_sync(aligned)
                identity, sim = self._match_sync(emb)
                results.append(
                    {
                        "box": [float(det["box"][0]), float(det["box"][1]), float(det["box"][2]), float(det["box"][3])],
                        "identity": identity,
                        "similarity": sim,
                        "confidence": float(det["conf"]),
                        "keypoints": det.get("keypoints"),
                    }
                )
        elapsed_ms = (time.time() - t0) * 1000.0
        return results, elapsed_ms

    def register_image_rgb(self, frame_rgb: np.ndarray, name: str) -> Tuple[bool, str, Optional[float]]:
        """Register single best face. Returns (ok, message, detection_confidence)."""
        with self._lock:
            if name in self.face_database:
                return False, f"Identity '{name}' already exists", None
            detections = self._detect_faces_sync(frame_rgb)
            if not detections:
                return False, "No face detected", None
            det = max(detections, key=lambda d: d["conf"])
            x1, y1, x2, y2 = map(int, det["box"])
            kps = det.get("keypoints")
            if kps and len(kps) >= 5:
                aligned = self._align_face_keypoints(frame_rgb, kps, 112)
            else:
                margin = int(max(x2 - x1, y2 - y1) * 0.1)
                xa, ya = max(0, x1 - margin), max(0, y1 - margin)
                xb, yb = min(frame_rgb.shape[1], x2 + margin), min(frame_rgb.shape[0], y2 + margin)
                aligned = cv2.resize(frame_rgb[ya:yb, xa:xb], (112, 112))
            emb = self._get_embedding_sync(aligned)
            self.face_database[name] = emb
            self._save_face_to_disk(name, emb, aligned)
            return True, "Registered", float(det["conf"])
