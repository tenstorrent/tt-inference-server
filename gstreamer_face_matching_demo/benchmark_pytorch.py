#!/usr/bin/env python3
"""
SFace Full Pipeline Benchmark - PyTorch Reference (CPU)

Same pipeline as TTNN benchmark but using PyTorch on CPU for comparison.
  Image → YuNet (detect) → Align → SFace (embed) → Match

Usage:
    cd ~/teja/tt-metal
    source python_env/bin/activate
    python ~/teja/tt-inference-server/gstreamer_face_matching_demo/benchmark_pytorch.py --pairs 200
"""

import sys
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Tuple, List, Optional

# Add tt-metal to path
TT_METAL_PATH = Path.home() / "teja" / "tt-metal"
sys.path.insert(0, str(TT_METAL_PATH))

# Dataset paths
LFW_FUNNELED_DIR = Path.home() / "scikit_learn_data" / "lfw_home" / "lfw_funneled"


def load_lfw_pairs_funneled(max_pairs: int = 1000) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Load LFW pairs from funneled images (250x250)."""
    pairs_file = Path.home() / "scikit_learn_data" / "lfw_home" / "pairs.txt"
    
    if not pairs_file.exists() or not LFW_FUNNELED_DIR.exists():
        print("[ERROR] LFW funneled not found.")
        return None, None
    
    print(f"[Dataset] Loading LFW funneled pairs...")
    
    pairs = []
    labels = []
    
    with open(pairs_file, 'r') as f:
        lines = f.readlines()[1:]
    
    n_pos, n_neg = 0, 0
    half_max = max_pairs // 2
    
    for line in lines:
        parts = line.strip().split('\t')
        
        if len(parts) == 3 and n_pos < half_max:
            name, n1, n2 = parts
            img1_path = LFW_FUNNELED_DIR / name / f"{name}_{int(n1):04d}.jpg"
            img2_path = LFW_FUNNELED_DIR / name / f"{name}_{int(n2):04d}.jpg"
            
            if img1_path.exists() and img2_path.exists():
                img1 = cv2.imread(str(img1_path))
                img2 = cv2.imread(str(img2_path))
                if img1 is not None and img2 is not None:
                    pairs.append((img1, img2))
                    labels.append(1)
                    n_pos += 1
                    
        elif len(parts) == 4 and n_neg < half_max:
            name1, n1, name2, n2 = parts
            img1_path = LFW_FUNNELED_DIR / name1 / f"{name1}_{int(n1):04d}.jpg"
            img2_path = LFW_FUNNELED_DIR / name2 / f"{name2}_{int(n2):04d}.jpg"
            
            if img1_path.exists() and img2_path.exists():
                img1 = cv2.imread(str(img1_path))
                img2 = cv2.imread(str(img2_path))
                if img1 is not None and img2 is not None:
                    pairs.append((img1, img2))
                    labels.append(0)
                    n_neg += 1
        
        if n_pos >= half_max and n_neg >= half_max:
            break
    
    print(f"[Dataset] Loaded {len(pairs)} pairs ({n_pos} positive, {n_neg} negative)")
    return pairs, np.array(labels)


def load_pytorch_models():
    """Load PyTorch YuNet and SFace models (CPU)."""
    
    # Load YuNet PyTorch
    print("[YuNet] Loading PyTorch model...")
    from models.experimental.yunet.common import load_torch_model, get_default_weights_path
    yunet = load_torch_model(get_default_weights_path())
    yunet.eval()
    print("[YuNet] Ready (CPU)!")
    
    # Load SFace PyTorch
    print("[SFace] Loading PyTorch model...")
    from models.experimental.sface.reference.sface_model import load_sface_from_onnx
    from models.experimental.sface.common import get_sface_onnx_path
    sface = load_sface_from_onnx(get_sface_onnx_path())
    sface.eval()
    print("[SFace] Ready (CPU)!")
    
    return yunet, sface


def detect_face_yunet_pytorch(image_bgr: np.ndarray, yunet) -> Optional[dict]:
    """Run YuNet face detection on PyTorch (CPU)."""
    from models.experimental.yunet.common import STRIDES
    
    h_orig, w_orig = image_bgr.shape[:2]
    
    # Resize to 640x640
    img_resized = cv2.resize(image_bgr, (640, 640))
    
    # To tensor (NCHW for PyTorch)
    tensor = torch.from_numpy(img_resized.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    
    # Run YuNet
    with torch.no_grad():
        outputs = yunet(tensor)
    
    # PyTorch YuNet outputs flattened format:
    # cls: (1, 8400, 1), box: (1, 8400, 4), obj: (1, 8400), kpt: (1, 8400, 10)
    cls_out = outputs[0]  # (1, 8400, 1)
    box_out = outputs[1]  # (1, 8400, 4)
    obj_out = outputs[2]  # (1, 8400)
    kpt_out = outputs[3]  # (1, 8400, 10)
    
    # Compute scores
    score = cls_out.squeeze(-1).sigmoid() * obj_out.sigmoid()  # (1, 8400)
    
    detections = []
    threshold = 0.5
    
    # Find high confidence detections
    high_conf = score > threshold
    if high_conf.any():
        indices = torch.where(high_conf)
        for i in range(len(indices[0])):
            b, idx = indices[0][i], indices[1][i]
            conf = score[b, idx].item()
            
            # Get box (already decoded by model)
            box = box_out[b, idx].tolist()  # [x1, y1, x2, y2] normalized or pixel?
            
            # Get keypoints
            kpts = kpt_out[b, idx].tolist()  # 10 values = 5 keypoints
            keypoints = []
            for k in range(5):
                kx = kpts[k * 2] * w_orig / 640 if kpts[k * 2] < 10 else kpts[k * 2] * w_orig / 640
                ky = kpts[k * 2 + 1] * h_orig / 640 if kpts[k * 2 + 1] < 10 else kpts[k * 2 + 1] * h_orig / 640
                keypoints.append([kx, ky])
            
            detections.append({"conf": conf, "keypoints": keypoints})
    
    if not detections:
        return None
    
    return max(detections, key=lambda x: x["conf"])


def align_face(image_bgr: np.ndarray, detection: dict, target_size: int = 112) -> np.ndarray:
    """Align face using 5-point keypoints."""
    keypoints = detection.get("keypoints")
    
    if keypoints is None or len(keypoints) < 5:
        return cv2.resize(image_bgr, (target_size, target_size))
    
    dst_pts = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)
    
    src_pts = np.array(keypoints[:5], dtype=np.float32)
    tform, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    
    if tform is None:
        return cv2.resize(image_bgr, (target_size, target_size))
    
    return cv2.warpAffine(image_bgr, tform, (target_size, target_size))


def get_embedding_sface_pytorch(face_bgr: np.ndarray, sface) -> np.ndarray:
    """Get embedding using PyTorch SFace (CPU)."""
    # SFace expects NCHW
    face_tensor = torch.from_numpy(face_bgr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    
    with torch.no_grad():
        embedding = sface(face_tensor)
    
    return embedding.numpy().flatten()


def process_image_pytorch(image_bgr: np.ndarray, yunet, sface) -> Tuple[np.ndarray, dict]:
    """Full pipeline using PyTorch on CPU."""
    timings = {}
    
    # 1. YuNet detection
    t0 = time.time()
    detection = detect_face_yunet_pytorch(image_bgr, yunet)
    timings['yunet'] = time.time() - t0
    
    if detection is None:
        t1 = time.time()
        face = cv2.resize(image_bgr, (112, 112))
        timings['align'] = time.time() - t1
    else:
        t1 = time.time()
        face = align_face(image_bgr, detection, 112)
        timings['align'] = time.time() - t1
    
    # 3. SFace embedding
    t2 = time.time()
    embedding = get_embedding_sface_pytorch(face, sface)
    timings['sface'] = time.time() - t2
    
    timings['total'] = timings['yunet'] + timings['align'] + timings['sface']
    
    return embedding, timings


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))


def compute_tpr_at_far(scores: np.ndarray, labels: np.ndarray, target_far: float) -> Tuple[float, float]:
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    
    thresholds = np.sort(np.unique(scores))[::-1]
    best_tpr, best_thresh = 0.0, 0.0
    
    for thresh in thresholds:
        fp = np.sum(negative_scores >= thresh)
        far = fp / len(negative_scores) if len(negative_scores) > 0 else 0
        tp = np.sum(positive_scores >= thresh)
        tpr = tp / len(positive_scores) if len(positive_scores) > 0 else 0
        
        if far <= target_far and tpr > best_tpr:
            best_tpr, best_thresh = tpr, thresh
    
    return best_tpr, best_thresh


def run_benchmark(max_pairs: int = 200):
    """Run PyTorch CPU benchmark."""
    print("=" * 60)
    print("  SFace PYTORCH (CPU) Benchmark")
    print("  YuNet (detect) → Align → SFace (embed) → Match")
    print("=" * 60)
    
    # 1. Load dataset
    print("\n[1/3] Loading LFW funneled pairs...")
    pairs, labels = load_lfw_pairs_funneled(max_pairs)
    
    if pairs is None:
        return
    
    # 2. Load PyTorch models
    print("\n[2/3] Loading PyTorch models (CPU)...")
    yunet, sface = load_pytorch_models()
    
    # 3. Process pairs
    print(f"\n[3/3] Processing {len(pairs)} pairs (PyTorch CPU)...")
    scores = []
    all_timings = {'yunet': [], 'align': [], 'sface': [], 'total': []}
    
    for i, (img1, img2) in enumerate(pairs):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(pairs)}...")
        
        emb1, t1 = process_image_pytorch(img1, yunet, sface)
        emb2, t2 = process_image_pytorch(img2, yunet, sface)
        
        for k in all_timings:
            all_timings[k].append(t1[k])
            all_timings[k].append(t2[k])
        
        sim = cosine_similarity(emb1, emb2)
        scores.append(sim)
    
    scores = np.array(scores)
    
    # Compute metrics
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    
    tpr_01, _ = compute_tpr_at_far(scores, labels, 0.001)
    tpr_1, _ = compute_tpr_at_far(scores, labels, 0.01)
    
    # Print results
    print("\n" + "=" * 60)
    print("  PYTORCH (CPU) BENCHMARK RESULTS")
    print("=" * 60)
    
    print(f"\n📊 ACCURACY:")
    print(f"  TPR @ 0.1% FAR: {tpr_01*100:.2f}%")
    print(f"  TPR @ 1.0% FAR: {tpr_1*100:.2f}%")
    
    print(f"\n📉 Score Distribution:")
    print(f"  Mean Positive: {positive_scores.mean():.4f}")
    print(f"  Mean Negative: {negative_scores.mean():.4f}")
    
    print(f"\n⚡ LATENCY (per image, PyTorch CPU):")
    print(f"  Total P95: {np.percentile(all_timings['total'], 95)*1000:.1f} ms")
    print(f"  Total Mean: {np.mean(all_timings['total'])*1000:.1f} ms")
    print(f"  Breakdown:")
    print(f"    YuNet (detect):  Mean {np.mean(all_timings['yunet'])*1000:.1f} ms")
    print(f"    Align:           Mean {np.mean(all_timings['align'])*1000:.2f} ms")
    print(f"    SFace (embed):   Mean {np.mean(all_timings['sface'])*1000:.1f} ms")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=200)
    args = parser.parse_args()
    
    run_benchmark(max_pairs=args.pairs)
