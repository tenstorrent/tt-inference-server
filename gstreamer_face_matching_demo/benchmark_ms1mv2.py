#!/usr/bin/env python3
"""
SFace Accuracy Benchmark using InsightFace Validation Sets

Uses pre-aligned 112x112 face pairs from InsightFace (.bin format).
Available datasets:
  - lfw.bin: LFW (6000 pairs)
  - cfp_fp.bin: CFP Frontal-Profile (7000 pairs)
  - agedb_30.bin: AgeDB-30 (6000 pairs)
  - calfw.bin: Cross-Age LFW (6000 pairs)
  - cplfw.bin: Cross-Pose LFW (6000 pairs)

Customer Requirement: >98% TPR at FAR=0.1%

Usage:
    cd ~/teja/tt-metal
    source python_env/bin/activate
    export PYTHONPATH="/home/ttuser/.local/lib/python3.10/site-packages:$PYTHONPATH"
    
    # LFW benchmark (default)
    python ~/teja/tt-inference-server/gstreamer_face_matching_demo/benchmark_ms1mv2.py
    
    # CFP-FP (harder - frontal vs profile)
    python ~/teja/tt-inference-server/gstreamer_face_matching_demo/benchmark_ms1mv2.py --dataset cfp_fp
    
    # Quick test with fewer pairs
    python ~/teja/tt-inference-server/gstreamer_face_matching_demo/benchmark_ms1mv2.py --pairs 500
"""

import sys
import time
import pickle
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Tuple, List

# Add tt-metal to path
TT_METAL_PATH = Path.home() / "teja" / "tt-metal"
sys.path.insert(0, str(TT_METAL_PATH))

# Dataset path
DATA_DIR = Path.home() / "datasets" / "faces_webface_112x112"


def load_bin_dataset(dataset_name: str = "lfw", max_pairs: int = None) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load InsightFace .bin validation dataset.
    
    Returns:
        images: List of 112x112 BGR images (already aligned)
        issame: numpy array of bool (True = same person)
    """
    bin_path = DATA_DIR / f"{dataset_name}.bin"
    
    if not bin_path.exists():
        available = [f.stem for f in DATA_DIR.glob("*.bin")]
        print(f"[ERROR] Dataset '{dataset_name}' not found at {bin_path}")
        print(f"[INFO] Available: {available}")
        return None, None
    
    print(f"[Dataset] Loading {bin_path}...")
    
    with open(bin_path, 'rb') as f:
        bins, issame = pickle.load(f, encoding='bytes')
    
    # Decode JPEG images
    images = []
    for img_bytes in bins:
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # BGR, 112x112
        images.append(img)
    
    issame = np.array(issame)
    
    # Limit pairs if specified
    if max_pairs and max_pairs < len(issame):
        images = images[:max_pairs * 2]
        issame = issame[:max_pairs]
    
    n_pos = np.sum(issame)
    n_neg = len(issame) - n_pos
    print(f"[Dataset] Loaded {len(issame)} pairs ({n_pos} positive, {n_neg} negative)")
    print(f"[Dataset] Image shape: {images[0].shape}")
    
    return images, issame


def init_device():
    """Initialize Tenstorrent device (T3K compatible)."""
    import ttnn
    import ttnn.distributed as dist
    from models.experimental.sface.common import SFACE_L1_SMALL_SIZE
    
    print("[Device] Opening 1x1 mesh device...")
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
    """Load SFace model."""
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


def get_embedding(face_bgr: np.ndarray, sface, device) -> np.ndarray:
    """Get 128-dim embedding from aligned face (112x112 BGR)."""
    import ttnn
    
    # Images are already 112x112 BGR aligned from InsightFace
    face_tensor = torch.from_numpy(face_bgr.astype(np.float32)).unsqueeze(0)  # (1, 112, 112, 3)
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


def run_benchmark(dataset: str = "lfw", max_pairs: int = None):
    """Run InsightFace validation benchmark."""
    print("=" * 60)
    print("  SFace Accuracy Benchmark")
    print("  Using InsightFace aligned 112x112 validation pairs")
    print("  Target: >98% TPR @ 0.1% FAR")
    print("=" * 60)
    
    # 1. Load dataset
    print(f"\n[1/4] Loading {dataset.upper()} dataset...")
    images, issame = load_bin_dataset(dataset, max_pairs)
    
    if images is None:
        return
    
    # 2. Initialize device and model
    print("\n[2/4] Initializing device and model...")
    device = init_device()
    
    try:
        sface = load_sface(device)
        
        # 3. Compute embeddings for all images
        print(f"\n[3/4] Computing embeddings for {len(images)} images...")
        embeddings = []
        times = []
        
        for i, img in enumerate(images):
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(images)}...")
            
            t_start = time.time()
            emb = get_embedding(img, sface, device)
            times.append(time.time() - t_start)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # 4. Compute similarity scores for pairs
        print("\n[4/4] Computing pair similarities...")
        scores = []
        labels = []
        
        for i in range(len(issame)):
            emb1 = embeddings[2 * i]
            emb2 = embeddings[2 * i + 1]
            sim = cosine_similarity(emb1, emb2)
            scores.append(sim)
            labels.append(1 if issame[i] else 0)
        
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Compute metrics
        print("\n" + "=" * 60)
        print(f"  BENCHMARK RESULTS - {dataset.upper()}")
        print("=" * 60)
        
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
        
        print(f"\n⚡ Latency (per image):")
        print(f"  Mean: {np.mean(times)*1000:.1f} ms")
        print(f"  P95: {np.percentile(times, 95)*1000:.1f} ms")
        
        print("\n" + "=" * 60)
        
        return {
            'tpr_01': tpr_01,
            'tpr_1': tpr_1,
            'tpr_10': tpr_10,
            'best_acc': best_acc,
            'mean_positive': positive_scores.mean(),
            'mean_negative': negative_scores.mean(),
        }
        
    finally:
        import ttnn
        ttnn.close_device(device)
        print("\n[Device] Closed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SFace InsightFace Benchmark")
    parser.add_argument("--dataset", type=str, default="lfw", 
                        choices=["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"],
                        help="Validation dataset to use")
    parser.add_argument("--pairs", type=int, default=None, help="Limit number of pairs (for quick testing)")
    args = parser.parse_args()
    
    run_benchmark(dataset=args.dataset, max_pairs=args.pairs)
