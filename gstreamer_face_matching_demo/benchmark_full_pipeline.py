#!/usr/bin/env python3
"""
SFace Full Pipeline Benchmark (YuNet + SFace)

Tests the COMPLETE pipeline as used in the demo:
  Image → YuNet (detect + keypoints) → Align → SFace → Match

Uses LFW funneled images (250x250) where YuNet can detect faces.

Customer Requirements:
  - Accuracy: >98% TPR @ 0.1% FAR
  - Latency: 400-500ms P95 end-to-end

Usage:
    cd ~/teja/tt-metal
    source python_env/bin/activate
    export PYTHONPATH="/home/ttuser/.local/lib/python3.10/site-packages:$PYTHONPATH"
    python ~/teja/tt-inference-server/gstreamer_face_matching_demo/benchmark_full_pipeline.py --pairs 500
"""

import sys
import time
import pickle
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Tuple, List, Optional

# Add tt-metal to path
TT_METAL_PATH = Path.home() / "teja" / "tt-metal"
sys.path.insert(0, str(TT_METAL_PATH))

# Dataset paths
INSIGHTFACE_DIR = Path.home() / "datasets" / "faces_webface_112x112"
LFW_FUNNELED_DIR = Path.home() / "scikit_learn_data" / "lfw_home" / "lfw_funneled"


def load_lfw_pairs_funneled(max_pairs: int = 1000) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Load LFW pairs from funneled images (250x250) for full pipeline testing.
    Uses pairs.txt to get matching pairs.
    """
    pairs_file = Path.home() / "scikit_learn_data" / "lfw_home" / "pairs.txt"
    
    if not pairs_file.exists() or not LFW_FUNNELED_DIR.exists():
        print("[ERROR] LFW funneled not found. Using InsightFace pre-aligned instead.")
        return None, None
    
    print(f"[Dataset] Loading LFW funneled pairs from {LFW_FUNNELED_DIR}...")
    
    pairs = []
    labels = []
    
    with open(pairs_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    n_pos, n_neg = 0, 0
    half_max = max_pairs // 2
    
    for line in lines:
        parts = line.strip().split('\t')
        
        if len(parts) == 3 and n_pos < half_max:
            # Positive pair
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
            # Negative pair
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
    if pairs:
        print(f"[Dataset] Image shape: {pairs[0][0].shape}")
    
    return pairs, np.array(labels)


def init_device():
    """Initialize Tenstorrent device."""
    import ttnn
    import ttnn.distributed as dist
    from models.experimental.sface.common import SFACE_L1_SMALL_SIZE
    from models.experimental.yunet.common import YUNET_L1_SMALL_SIZE
    
    l1_size = max(SFACE_L1_SMALL_SIZE, YUNET_L1_SMALL_SIZE)
    
    print(f"[Device] Opening 1x1 mesh device (l1_small_size={l1_size})...")
    mesh = dist.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[0],
        l1_small_size=l1_size,
        trace_region_size=0
    )
    mesh.enable_program_cache()
    print("[Device] Ready!")
    return mesh


def load_models(device):
    """Load YuNet and SFace models."""
    import ttnn
    
    # Load YuNet
    print("[YuNet] Loading...")
    from models.experimental.yunet.tt.ttnn_yunet import create_yunet_model
    from models.experimental.yunet.common import load_torch_model, get_default_weights_path
    
    yunet_torch = load_torch_model(get_default_weights_path()).to(torch.bfloat16)
    yunet = create_yunet_model(device, yunet_torch)
    
    # Warmup YuNet
    warmup = torch.randn(1, 640, 640, 3, dtype=torch.float32).to(torch.bfloat16)
    warmup_tt = ttnn.from_torch(warmup, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    _ = yunet(warmup_tt)
    ttnn.synchronize_device(device)
    print("[YuNet] Ready!")
    
    # Load SFace
    print("[SFace] Loading...")
    from models.experimental.sface.tt.ttnn_sface import create_sface_model
    from models.experimental.sface.reference.sface_model import load_sface_from_onnx
    from models.experimental.sface.common import get_sface_onnx_path
    
    sface_torch = load_sface_from_onnx(get_sface_onnx_path())
    sface_torch.eval()
    sface = create_sface_model(device, sface_torch)
    
    # Warmup SFace
    warmup = torch.randn(1, 112, 112, 3, dtype=torch.float32)
    warmup_tt = ttnn.from_torch(warmup, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    _ = sface(warmup_tt)
    ttnn.synchronize_device(device)
    print("[SFace] Ready!")
    
    return yunet, sface


def detect_face_yunet(image_bgr: np.ndarray, yunet, device) -> Optional[dict]:
    """Run YuNet face detection."""
    import ttnn
    from models.experimental.yunet.common import STRIDES
    
    h_orig, w_orig = image_bgr.shape[:2]
    
    # Resize to 640x640
    img_resized = cv2.resize(image_bgr, (640, 640))
    
    # To tensor (NHWC)
    tensor = torch.from_numpy(img_resized.astype(np.float32)).unsqueeze(0).to(torch.bfloat16)
    tt_input = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    
    # Run YuNet
    cls_out, box_out, obj_out, kpt_out = yunet(tt_input)
    ttnn.synchronize_device(device)
    
    # Decode detections
    detections = []
    threshold = 0.5
    
    for scale_idx in range(3):
        cls = ttnn.to_torch(cls_out[scale_idx]).float().permute(0, 3, 1, 2)
        box = ttnn.to_torch(box_out[scale_idx]).float().permute(0, 3, 1, 2)
        obj = ttnn.to_torch(obj_out[scale_idx]).float().permute(0, 3, 1, 2)
        kpt = ttnn.to_torch(kpt_out[scale_idx]).float().permute(0, 3, 1, 2)
        
        stride = STRIDES[scale_idx]
        score = cls.sigmoid() * obj.sigmoid()
        
        high_conf = score > threshold
        if high_conf.any():
            indices = torch.where(high_conf)
            for i in range(len(indices[0])):
                b, c, gh, gw = indices[0][i], indices[1][i], indices[2][i], indices[3][i]
                conf = score[b, c, gh, gw].item()
                anchor_x, anchor_y = gw.item() * stride, gh.item() * stride
                
                # Keypoints
                keypoints = []
                for k in range(5):
                    kpt_dx = kpt[b, k * 2, gh, gw].item()
                    kpt_dy = kpt[b, k * 2 + 1, gh, gw].item()
                    kx = (kpt_dx * stride + anchor_x) * w_orig / 640
                    ky = (kpt_dy * stride + anchor_y) * h_orig / 640
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
    
    # ArcFace alignment template
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


def get_embedding_sface(face_bgr: np.ndarray, sface, device) -> np.ndarray:
    """Get embedding from aligned face."""
    import ttnn
    
    face_tensor = torch.from_numpy(face_bgr.astype(np.float32)).unsqueeze(0)
    tt_input = ttnn.from_torch(face_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = sface(tt_input)
    ttnn.synchronize_device(device)
    
    return ttnn.to_torch(tt_output).float().numpy().flatten()


def process_image_full_pipeline(image_bgr: np.ndarray, yunet, sface, device) -> Tuple[Optional[np.ndarray], dict]:
    """
    Full pipeline: YuNet detect → Align → SFace embed
    Returns (embedding, timing_dict) or (None, timing_dict) if detection fails
    """
    timings = {}
    
    # 1. YuNet detection
    t0 = time.time()
    detection = detect_face_yunet(image_bgr, yunet, device)
    timings['yunet'] = time.time() - t0
    
    if detection is None:
        # Fallback: use center crop
        t1 = time.time()
        face = cv2.resize(image_bgr, (112, 112))
        timings['align'] = time.time() - t1
    else:
        # 2. Align face
        t1 = time.time()
        face = align_face(image_bgr, detection, 112)
        timings['align'] = time.time() - t1
    
    # 3. SFace embedding
    t2 = time.time()
    embedding = get_embedding_sface(face, sface, device)
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


def run_benchmark(max_pairs: int = 500):
    """Run full pipeline benchmark."""
    print("=" * 60)
    print("  SFace FULL PIPELINE Benchmark")
    print("  YuNet (detect) → Align → SFace (embed) → Match")
    print("  Target: >98% TPR @ 0.1% FAR, <500ms P95 latency")
    print("=" * 60)
    
    # 1. Load dataset
    print("\n[1/4] Loading LFW funneled pairs...")
    pairs, labels = load_lfw_pairs_funneled(max_pairs)
    
    if pairs is None or len(pairs) == 0:
        print("[ERROR] Could not load LFW funneled. Exiting.")
        return
    
    # 2. Initialize device and models
    print("\n[2/4] Initializing device and models...")
    device = init_device()
    
    try:
        yunet, sface = load_models(device)
        
        # 3. Process pairs
        print(f"\n[3/4] Processing {len(pairs)} pairs (full pipeline)...")
        scores = []
        all_timings = {'yunet': [], 'align': [], 'sface': [], 'total': []}
        failed_detections = 0
        
        for i, (img1, img2) in enumerate(pairs):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(pairs)}...")
            
            # Process both images through full pipeline
            emb1, t1 = process_image_full_pipeline(img1, yunet, sface, device)
            emb2, t2 = process_image_full_pipeline(img2, yunet, sface, device)
            
            # Accumulate timings
            for k in all_timings:
                all_timings[k].append(t1[k])
                all_timings[k].append(t2[k])
            
            # Compute similarity
            sim = cosine_similarity(emb1, emb2)
            scores.append(sim)
        
        scores = np.array(scores)
        
        # 4. Compute metrics
        print("\n[4/4] Computing metrics...")
        
        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]
        
        tpr_01, thresh_01 = compute_tpr_at_far(scores, labels, 0.001)
        tpr_1, thresh_1 = compute_tpr_at_far(scores, labels, 0.01)
        tpr_10, thresh_10 = compute_tpr_at_far(scores, labels, 0.1)
        
        # Best accuracy
        thresholds = np.linspace(scores.min(), scores.max(), 1000)
        best_acc, best_thresh = 0, 0
        for t in thresholds:
            tp = np.sum(positive_scores >= t)
            tn = np.sum(negative_scores < t)
            acc = (tp + tn) / len(scores)
            if acc > best_acc:
                best_acc, best_thresh = acc, t
        
        # Print results
        print("\n" + "=" * 60)
        print("  FULL PIPELINE BENCHMARK RESULTS")
        print("=" * 60)
        
        acc_pass = "✅ PASS" if tpr_01 >= 0.98 else "❌ FAIL"
        lat_pass = "✅ PASS" if np.percentile(all_timings['total'], 95) * 1000 <= 500 else "❌ FAIL"
        
        print(f"\n📊 ACCURACY (Customer Requirement):")
        print(f"  TPR @ 0.1% FAR: {tpr_01*100:.2f}% (Required: >98%) {acc_pass}")
        print(f"  TPR @ 1.0% FAR: {tpr_1*100:.2f}%")
        print(f"  TPR @ 10.0% FAR: {tpr_10*100:.2f}%")
        print(f"  Best Accuracy: {best_acc*100:.2f}%")
        
        print(f"\n📉 Score Distribution:")
        print(f"  Mean Positive: {positive_scores.mean():.4f}")
        print(f"  Mean Negative: {negative_scores.mean():.4f}")
        print(f"  Separation: {positive_scores.mean() - negative_scores.mean():.4f}")
        
        print(f"\n⚡ LATENCY (per image, full pipeline):")
        print(f"  Total P95: {np.percentile(all_timings['total'], 95)*1000:.1f} ms {lat_pass}")
        print(f"  Total Mean: {np.mean(all_timings['total'])*1000:.1f} ms")
        print(f"  Breakdown:")
        print(f"    YuNet (detect):  Mean {np.mean(all_timings['yunet'])*1000:.1f} ms, P95 {np.percentile(all_timings['yunet'], 95)*1000:.1f} ms")
        print(f"    Align:           Mean {np.mean(all_timings['align'])*1000:.2f} ms")
        print(f"    SFace (embed):   Mean {np.mean(all_timings['sface'])*1000:.1f} ms, P95 {np.percentile(all_timings['sface'], 95)*1000:.1f} ms")
        
        print("\n" + "=" * 60)
        
    finally:
        import ttnn
        ttnn.close_device(device)
        print("\n[Device] Closed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SFace Full Pipeline Benchmark")
    parser.add_argument("--pairs", type=int, default=500, help="Number of pairs to test")
    args = parser.parse_args()
    
    run_benchmark(max_pairs=args.pairs)
