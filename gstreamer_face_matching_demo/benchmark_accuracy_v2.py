#!/usr/bin/env python3
"""
Face Recognition Accuracy Benchmark v2

Uses sklearn LFW (already cached, no download needed).
LFW images are pre-cropped faces, so we skip YuNet and go directly to SFace.

Customer Requirement: >98% TPR at 0.1% FAR

Usage:
    cd ~/teja/tt-metal
    source python_env/bin/activate
    export PYTHONPATH="/home/ttuser/.local/lib/python3.10/site-packages:$PYTHONPATH"
    python ~/teja/tt-inference-server/gstreamer_face_matching_demo/benchmark_accuracy_v2.py
"""

import sys
import os
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Tuple, List, Optional

# Add tt-metal to path
TT_METAL_PATH = Path.home() / "teja" / "tt-metal"
sys.path.insert(0, str(TT_METAL_PATH))


def load_lfw_sklearn(max_pairs: int = 500):
    """
    Load LFW pairs using sklearn (uses cached data, no internet needed).
    
    Returns balanced positive/negative pairs from people with multiple images.
    """
    from sklearn.datasets import fetch_lfw_pairs
    
    print("[Dataset] Loading sklearn LFW pairs (cached)...")
    
    # fetch_lfw_pairs gives us pre-made verification pairs
    # It returns images as (n_pairs, 2, h, w) and labels
    lfw = fetch_lfw_pairs(subset='test', color=True, resize=1.0)
    
    # lfw.pairs shape: (n_pairs, 2, 62, 47, 3) for color
    # lfw.target: 1=same, 0=different
    
    images = lfw.pairs  # (n, 2, h, w, c)
    labels = lfw.target
    
    # Balance the dataset
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    
    half = min(max_pairs // 2, len(pos_idx), len(neg_idx))
    
    selected_pos = pos_idx[:half]
    selected_neg = neg_idx[:half]
    selected = np.concatenate([selected_pos, selected_neg])
    
    pairs = [(images[i, 0], images[i, 1]) for i in selected]
    labels_out = np.concatenate([np.ones(half), np.zeros(half)]).astype(int)
    
    print(f"[Dataset] Loaded {len(pairs)} pairs ({half} positive, {half} negative)")
    print(f"[Dataset] Image shape: {images[0, 0].shape}")
    
    return pairs, labels_out


def init_device():
    """Initialize Tenstorrent device."""
    import ttnn
    import ttnn.distributed as dist
    from models.experimental.sface.common import SFACE_L1_SMALL_SIZE
    
    print(f"[Device] Opening 1x1 mesh device...")
    mesh = dist.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[0],
        l1_small_size=SFACE_L1_SMALL_SIZE,
        trace_region_size=0
    )
    mesh.enable_program_cache()
    print("[Device] Ready!")
    return mesh


def load_sface(device):
    """Load SFace model only (no YuNet needed for pre-cropped faces)."""
    import ttnn
    
    print("[SFace] Loading...")
    from models.experimental.sface.tt.ttnn_sface import create_sface_model
    from models.experimental.sface.reference.sface_model import load_sface_from_onnx
    from models.experimental.sface.common import get_sface_onnx_path
    
    sface_torch = load_sface_from_onnx(get_sface_onnx_path())
    sface_torch.eval()
    sface = create_sface_model(device, sface_torch)
    
    # Warmup
    warmup = torch.randn(1, 112, 112, 3, dtype=torch.float32)
    warmup_tt = ttnn.from_torch(warmup, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    _ = sface(warmup_tt)
    ttnn.synchronize_device(device)
    print("[SFace] Ready!")
    
    return sface


def preprocess_face(face_img: np.ndarray, target_size: int = 112) -> np.ndarray:
    """
    Preprocess face for SFace.
    
    LFW sklearn images are (62, 47, 3) uint8 in range [0, 255].
    SFace expects (112, 112, 3) BGR float32.
    """
    # Ensure uint8
    if face_img.dtype != np.uint8:
        face_img = (face_img * 255).astype(np.uint8) if face_img.max() <= 1.0 else face_img.astype(np.uint8)
    
    # RGB to BGR (SFace is OpenCV-based)
    face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
    
    # Resize to 112x112 with INTER_LINEAR (matches training)
    face_resized = cv2.resize(face_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return face_resized.astype(np.float32)


def get_embedding(face: np.ndarray, sface, device) -> np.ndarray:
    """Get 128-dim embedding from preprocessed face (112x112 BGR float32)."""
    import ttnn
    
    face_tensor = torch.from_numpy(face).unsqueeze(0)  # (1, 112, 112, 3)
    tt_input = ttnn.from_torch(face_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = sface(tt_input)
    ttnn.synchronize_device(device)
    
    return ttnn.to_torch(tt_output).float().numpy().flatten()


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity."""
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))


def compute_tpr_at_far(scores: np.ndarray, labels: np.ndarray, target_far: float) -> Tuple[float, float]:
    """Compute TPR at specific FAR."""
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
    """Run benchmark with SFace on LFW pre-cropped faces."""
    print("=" * 60)
    print("  Face Recognition Benchmark v2")
    print("  Pipeline: LFW face crops → Resize 112x112 → SFace → Match")
    print("  Target: >98% TPR @ 0.1% FAR")
    print("=" * 60)
    
    # 1. Load dataset (uses cached sklearn LFW)
    print("\n[1/4] Loading LFW dataset...")
    pairs, labels = load_lfw_sklearn(max_pairs=max_pairs)
    
    # 2. Initialize device and model
    print("\n[2/4] Initializing device and model...")
    device = init_device()
    
    try:
        sface = load_sface(device)
        
        # 3. Process pairs
        print(f"\n[3/4] Processing {len(pairs)} pairs...")
        scores = []
        times = []
        
        for i, (img1, img2) in enumerate(pairs):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(pairs)}...")
            
            t_start = time.time()
            
            # Preprocess (resize to 112x112, RGB→BGR)
            face1 = preprocess_face(img1)
            face2 = preprocess_face(img2)
            
            # Get embeddings
            emb1 = get_embedding(face1, sface, device)
            emb2 = get_embedding(face2, sface, device)
            
            # Compute similarity
            sim = cosine_similarity(emb1, emb2)
            scores.append(sim)
            times.append(time.time() - t_start)
        
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
        print("  BENCHMARK RESULTS")
        print("=" * 60)
        
        customer_pass = "✅ PASS" if tpr_01 >= 0.98 else "❌ FAIL"
        
        print(f"\n📊 KEY METRIC (Customer Requirement):")
        print(f"  TPR @ 0.1% FAR: {tpr_01*100:.2f}% (Required: >98%) {customer_pass}")
        print(f"  Threshold: {thresh_01:.4f}")
        
        print(f"\n📈 Additional TPR Metrics:")
        print(f"  TPR @ 1.0% FAR: {tpr_1*100:.2f}%")
        print(f"  TPR @ 10.0% FAR: {tpr_10*100:.2f}%")
        
        print(f"\n📉 Score Distribution:")
        print(f"  Mean Positive: {positive_scores.mean():.4f} (std: {positive_scores.std():.4f})")
        print(f"  Mean Negative: {negative_scores.mean():.4f} (std: {negative_scores.std():.4f})")
        print(f"  Separation: {positive_scores.mean() - negative_scores.mean():.4f}")
        
        print(f"\n🎯 Best Accuracy: {best_acc*100:.2f}% @ threshold {best_thresh:.4f}")
        
        print(f"\n⚡ Latency (per pair):")
        print(f"  Mean: {np.mean(times)*1000:.1f} ms")
        print(f"  P95: {np.percentile(times, 95)*1000:.1f} ms")
        
        print("\n" + "=" * 60)
        
    finally:
        import ttnn
        ttnn.close_device(device)
        print("\n[Device] Closed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=500, help="Number of pairs to test")
    args = parser.parse_args()
    
    run_benchmark(max_pairs=args.pairs)
