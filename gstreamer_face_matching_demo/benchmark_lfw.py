#!/usr/bin/env python3
"""
Face Recognition Benchmark - PyTorch vs TTNN

Tests SFace model accuracy on LFW pre-aligned faces.
This isolates SFace performance without YuNet detection variability.

Uses sklearn LFW pairs (pre-cropped faces) resized to 112x112.

Usage:
    docker run ... face-matching-demo accuracy-benchmark
    docker run ... face-matching-demo accuracy-benchmark --pairs 500
"""

import sys
import os
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Tuple, List

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Add tt-metal to path
TT_METAL_PATH = Path("/home/container_app_user/tt-metal")
if TT_METAL_PATH.exists():
    sys.path.insert(0, str(TT_METAL_PATH))


def load_lfw_sklearn(max_pairs: int = 500):
    """Load LFW pairs using sklearn (pre-aligned face crops)."""
    from sklearn.datasets import fetch_lfw_pairs
    
    print("[Dataset] Loading sklearn LFW pairs (pre-aligned faces)...")
    
    lfw = fetch_lfw_pairs(subset='test', color=True, resize=1.0)
    
    images = lfw.pairs  # (n, 2, h, w, c) - faces already cropped
    labels = lfw.target
    
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    
    half = min(max_pairs // 2, len(pos_idx), len(neg_idx))
    
    selected_pos = pos_idx[:half]
    selected_neg = neg_idx[:half]
    selected = np.concatenate([selected_pos, selected_neg])
    
    pairs = [(images[i, 0], images[i, 1]) for i in selected]
    labels_out = np.concatenate([np.ones(half), np.zeros(half)]).astype(int)
    
    print(f"[Dataset] Loaded {len(pairs)} pairs ({half} positive, {half} negative)")
    print(f"[Dataset] Image shape: {images[0, 0].shape} (pre-aligned faces)")
    
    return pairs, labels_out


def preprocess_face(face_img: np.ndarray, target_size: int = 112) -> np.ndarray:
    """Preprocess face for SFace (resize to 112x112, RGB->BGR)."""
    if face_img.dtype != np.uint8:
        face_img = (face_img * 255).astype(np.uint8) if face_img.max() <= 1.0 else face_img.astype(np.uint8)
    
    # RGB to BGR (SFace trained on BGR)
    face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
    
    # Resize to 112x112
    face_resized = cv2.resize(face_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return face_resized.astype(np.float32)


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


def compute_best_accuracy(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    best_acc, best_thresh = 0, 0
    for t in thresholds:
        tp = np.sum(positive_scores >= t)
        tn = np.sum(negative_scores < t)
        acc = (tp + tn) / len(scores)
        if acc > best_acc:
            best_acc, best_thresh = acc, t
    
    return best_acc, best_thresh


# ===================== PyTorch Benchmark =====================

def run_pytorch_benchmark(pairs, labels):
    """Run PyTorch (CPU) benchmark - SFace only."""
    print("\n" + "=" * 60)
    print("  PYTORCH (CPU) BENCHMARK - SFace Only")
    print("=" * 60)
    
    from models.experimental.sface.reference.sface_model import load_sface_from_onnx
    from models.experimental.sface.common import get_sface_onnx_path
    
    print("[SFace] Loading PyTorch model...")
    sface = load_sface_from_onnx(get_sface_onnx_path())
    sface.eval()
    
    # Warmup
    with torch.no_grad():
        dummy = torch.randn(1, 3, 112, 112)
        _ = sface(dummy)
    print("[SFace] Ready!")
    
    # Process pairs
    print(f"[Benchmark] Processing {len(pairs)} pairs...")
    scores = []
    times = []
    
    for i, (img1, img2) in enumerate(pairs):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(pairs)}...")
        
        # Preprocess (resize to 112x112)
        face1 = preprocess_face(img1)
        face2 = preprocess_face(img2)
        
        # PyTorch uses NCHW
        t1 = torch.from_numpy(face1).permute(2, 0, 1).unsqueeze(0)
        t2 = torch.from_numpy(face2).permute(2, 0, 1).unsqueeze(0)
        
        t_start = time.time()
        with torch.no_grad():
            emb1 = sface(t1).numpy().flatten()
            emb2 = sface(t2).numpy().flatten()
        times.append((time.time() - t_start) / 2)  # Per image
        
        sim = cosine_similarity(emb1, emb2)
        scores.append(sim)
    
    scores = np.array(scores)
    
    # Compute metrics
    tpr_01, _ = compute_tpr_at_far(scores, labels, 0.001)
    tpr_1, _ = compute_tpr_at_far(scores, labels, 0.01)
    best_acc, _ = compute_best_accuracy(scores, labels)
    
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    
    return {
        'tpr_01': tpr_01,
        'tpr_1': tpr_1,
        'best_acc': best_acc,
        'mean_positive': positive_scores.mean(),
        'mean_negative': negative_scores.mean(),
        'mean_latency_ms': np.mean(times) * 1000,
        'p95_latency_ms': np.percentile(times, 95) * 1000,
        'p99_latency_ms': np.percentile(times, 99) * 1000,
    }


# ===================== TTNN Benchmark =====================

def run_ttnn_benchmark(pairs, labels):
    """Run TTNN (Tenstorrent) benchmark - SFace only."""
    print("\n" + "=" * 60)
    print("  TTNN (TENSTORRENT) BENCHMARK - SFace Only")
    print("=" * 60)
    
    import ttnn
    import ttnn.distributed as dist
    from models.experimental.sface.common import SFACE_L1_SMALL_SIZE
    
    # Open device
    print("[Device] Opening mesh device (1x1)...")
    mesh = dist.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[0],
        l1_small_size=SFACE_L1_SMALL_SIZE,
        trace_region_size=0
    )
    mesh.enable_program_cache()
    print("[Device] Ready!")
    
    try:
        # Load SFace TTNN
        print("[SFace] Loading TTNN model...")
        from models.experimental.sface.tt.ttnn_sface import create_sface_model
        from models.experimental.sface.reference.sface_model import load_sface_from_onnx
        from models.experimental.sface.common import get_sface_onnx_path
        
        sface_torch = load_sface_from_onnx(get_sface_onnx_path())
        sface = create_sface_model(mesh, sface_torch)
        
        # Warmup
        print("[SFace] Warming up...")
        warmup = torch.randn(1, 112, 112, 3, dtype=torch.float32)
        warmup_tt = ttnn.from_torch(warmup, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        _ = sface(warmup_tt)
        ttnn.synchronize_device(mesh)
        print("[SFace] Ready!")
        
        # Process pairs
        print(f"[Benchmark] Processing {len(pairs)} pairs...")
        scores = []
        times = []
        
        for i, (img1, img2) in enumerate(pairs):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(pairs)}...")
            
            # Preprocess (resize to 112x112)
            face1 = preprocess_face(img1)
            face2 = preprocess_face(img2)
            
            # TTNN uses NHWC
            t1 = torch.from_numpy(face1).unsqueeze(0)  # (1, 112, 112, 3)
            t2 = torch.from_numpy(face2).unsqueeze(0)
            
            t_start = time.time()
            
            tt1 = ttnn.from_torch(t1, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
            tt2 = ttnn.from_torch(t2, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
            
            out1 = sface(tt1)
            out2 = sface(tt2)
            ttnn.synchronize_device(mesh)
            
            emb1 = ttnn.to_torch(out1).float().numpy().flatten()
            emb2 = ttnn.to_torch(out2).float().numpy().flatten()
            
            times.append((time.time() - t_start) / 2)  # Per image
            
            sim = cosine_similarity(emb1, emb2)
            scores.append(sim)
        
        scores = np.array(scores)
        
        # Compute metrics
        tpr_01, _ = compute_tpr_at_far(scores, labels, 0.001)
        tpr_1, _ = compute_tpr_at_far(scores, labels, 0.01)
        best_acc, _ = compute_best_accuracy(scores, labels)
        
        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]
        
        return {
            'tpr_01': tpr_01,
            'tpr_1': tpr_1,
            'best_acc': best_acc,
            'mean_positive': positive_scores.mean(),
            'mean_negative': negative_scores.mean(),
            'mean_latency_ms': np.mean(times) * 1000,
            'p95_latency_ms': np.percentile(times, 95) * 1000,
            'p99_latency_ms': np.percentile(times, 99) * 1000,
        }
        
    finally:
        ttnn.close_device(mesh)
        print("[Device] Closed.")


# ===================== Main =====================

def print_comparison(pytorch_results, ttnn_results):
    """Print side-by-side comparison."""
    print("\n")
    print("=" * 70)
    print("  FACE RECOGNITION BENCHMARK RESULTS")
    print("  Model: SFace (Face Embedding)")
    print("  Dataset: LFW Pre-aligned Faces")
    print("=" * 70)
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                         ACCURACY                                    │")
    print("├─────────────────────────┬─────────────────┬─────────────────────────┤")
    print("│ Metric                  │ PyTorch (CPU)   │ TTNN (Tenstorrent)      │")
    print("├─────────────────────────┼─────────────────┼─────────────────────────┤")
    print(f"│ TPR @ 0.1% FAR          │ {pytorch_results['tpr_01']*100:>12.2f}%  │ {ttnn_results['tpr_01']*100:>20.2f}%  │")
    print(f"│ TPR @ 1.0% FAR          │ {pytorch_results['tpr_1']*100:>12.2f}%  │ {ttnn_results['tpr_1']*100:>20.2f}%  │")
    print(f"│ Best Accuracy           │ {pytorch_results['best_acc']*100:>12.2f}%  │ {ttnn_results['best_acc']*100:>20.2f}%  │")
    print("├─────────────────────────┼─────────────────┼─────────────────────────┤")
    print(f"│ Mean Positive Score     │ {pytorch_results['mean_positive']:>14.4f}  │ {ttnn_results['mean_positive']:>22.4f}  │")
    print(f"│ Mean Negative Score     │ {pytorch_results['mean_negative']:>14.4f}  │ {ttnn_results['mean_negative']:>22.4f}  │")
    print("└─────────────────────────┴─────────────────┴─────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                         LATENCY (per image)                         │")
    print("├─────────────────────────┬─────────────────┬─────────────────────────┤")
    print("│ Metric                  │ PyTorch (CPU)   │ TTNN (Tenstorrent)      │")
    print("├─────────────────────────┼─────────────────┼─────────────────────────┤")
    print(f"│ Mean Latency            │ {pytorch_results['mean_latency_ms']:>12.1f} ms │ {ttnn_results['mean_latency_ms']:>19.1f} ms │")
    print(f"│ P95 Latency             │ {pytorch_results['p95_latency_ms']:>12.1f} ms │ {ttnn_results['p95_latency_ms']:>19.1f} ms │")
    print(f"│ P99 Latency             │ {pytorch_results['p99_latency_ms']:>12.1f} ms │ {ttnn_results['p99_latency_ms']:>19.1f} ms │")
    print("└─────────────────────────┴─────────────────┴─────────────────────────┘")
    
    # Speedup
    speedup = pytorch_results['p95_latency_ms'] / ttnn_results['p95_latency_ms']
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                         SUMMARY                                     │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    
    if speedup >= 1.0:
        print(f"│ Speedup (P95):          {speedup:>6.1f}x faster on Tenstorrent                │")
    else:
        print(f"│ Latency:                {1/speedup:>6.1f}x slower (single-image overhead)      │")
    
    # Customer requirement check
    customer_pass = ttnn_results['tpr_01'] >= 0.98
    status = "✅ PASS" if customer_pass else "❌ FAIL"
    print(f"│ Customer Req (>98% TPR @ 0.1% FAR): {status}                          │")
    
    # Accuracy match
    acc_diff = abs(pytorch_results['best_acc'] - ttnn_results['best_acc']) * 100
    if acc_diff < 1.0:
        print(f"│ Accuracy Match: ✅ TTNN matches PyTorch (diff: {acc_diff:.2f}%)              │")
    else:
        print(f"│ Accuracy Match: ⚠️  Diff: {acc_diff:.2f}%                                    │")
    
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n📝 Note: Latency measured for SINGLE image inference.")
    print("   For throughput, use batch processing or 8-device parallel mode.")
    print("\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Face Recognition Benchmark")
    parser.add_argument("--pairs", type=int, default=500, help="Number of pairs to test")
    parser.add_argument("--pytorch-only", action="store_true", help="Only run PyTorch benchmark")
    parser.add_argument("--ttnn-only", action="store_true", help="Only run TTNN benchmark")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Face Recognition Benchmark")
    print("  Model: SFace (Face Embedding Network)")
    print("  Dataset: LFW Pre-aligned Faces (sklearn)")
    print("  Test: Compare PyTorch vs TTNN on SAME aligned inputs")
    print("  Customer Requirement: >98% TPR @ 0.1% FAR")
    print("=" * 70)
    
    # Load dataset
    print("\n[1/3] Loading LFW pre-aligned faces...")
    pairs, labels = load_lfw_sklearn(max_pairs=args.pairs)
    
    if pairs is None or len(pairs) == 0:
        print("[ERROR] Failed to load LFW dataset")
        return
    
    pytorch_results = None
    ttnn_results = None
    
    # Run benchmarks
    if not args.ttnn_only:
        print("\n[2/3] Running PyTorch (CPU) benchmark...")
        pytorch_results = run_pytorch_benchmark(pairs, labels)
    
    if not args.pytorch_only:
        print("\n[3/3] Running TTNN (Tenstorrent) benchmark...")
        ttnn_results = run_ttnn_benchmark(pairs, labels)
    
    # Print results
    if pytorch_results and ttnn_results:
        print_comparison(pytorch_results, ttnn_results)
    elif pytorch_results:
        print("\n" + "=" * 60)
        print("  PyTorch (CPU) Results")
        print("=" * 60)
        print(f"  TPR @ 0.1% FAR: {pytorch_results['tpr_01']*100:.2f}%")
        print(f"  TPR @ 1.0% FAR: {pytorch_results['tpr_1']*100:.2f}%")
        print(f"  Best Accuracy: {pytorch_results['best_acc']*100:.2f}%")
        print(f"  P95 Latency: {pytorch_results['p95_latency_ms']:.1f} ms")
    elif ttnn_results:
        print("\n" + "=" * 60)
        print("  TTNN (Tenstorrent) Results")
        print("=" * 60)
        print(f"  TPR @ 0.1% FAR: {ttnn_results['tpr_01']*100:.2f}%")
        print(f"  TPR @ 1.0% FAR: {ttnn_results['tpr_1']*100:.2f}%")
        print(f"  Best Accuracy: {ttnn_results['best_acc']*100:.2f}%")
        print(f"  P95 Latency: {ttnn_results['p95_latency_ms']:.1f} ms")


if __name__ == "__main__":
    main()
