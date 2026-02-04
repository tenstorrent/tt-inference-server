#!/usr/bin/env python3
"""
Face Recognition Accuracy Benchmark (TPR @ FAR)

Customer Requirement: >98% TPR at 0.1% FAR

Uses LFW (Labeled Faces in the Wild) dataset.
Since LFW images are already pre-cropped faces, we SKIP face detection
and directly feed to SFace for embedding extraction.

Pipeline: LFW image (62x47) → Resize (112x112) → SFace → Embedding → Cosine similarity

Usage:
    cd ~/teja/tt-metal
    source python_env/bin/activate
    python ~/teja/tt-inference-server/gstreamer_face_matching_demo/benchmark_accuracy.py
"""

import sys
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Tuple

# Add tt-metal to path
TT_METAL_PATH = Path.home() / "teja" / "tt-metal"
sys.path.insert(0, str(TT_METAL_PATH))


def download_lfw_dataset():
    """Download LFW pairs dataset using scikit-learn."""
    try:
        from sklearn.datasets import fetch_lfw_pairs
    except ImportError:
        print("ERROR: scikit-learn not installed")
        print("  Install: pip install scikit-learn")
        return None, None
    
    print("[Dataset] Downloading LFW pairs via scikit-learn...")
    print("  (First run downloads ~200MB, cached after)")
    
    # 'test' subset = 1000 pairs (500 positive, 500 negative)
    lfw = fetch_lfw_pairs(subset='test', color=True, resize=1.0)
    
    # lfw.pairs shape: (n_pairs, 2, H, W, C) - typically (1000, 2, 62, 47, 3)
    # lfw.target: 1 = same person, 0 = different person
    pairs = lfw.pairs
    labels = lfw.target
    
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    print(f"[Dataset] Loaded {len(labels)} pairs ({n_pos} positive, {n_neg} negative)")
    print(f"[Dataset] Image size: {pairs.shape[2]}x{pairs.shape[3]}")
    
    return pairs, labels


def init_device():
    """Initialize Tenstorrent device with 1x1 mesh (for T3K compatibility)."""
    import ttnn
    import ttnn.distributed as dist
    
    # Get L1 size from SFace model
    from models.experimental.sface.common import SFACE_L1_SMALL_SIZE
    
    print(f"[Device] Opening 1x1 mesh device (l1_small_size={SFACE_L1_SMALL_SIZE})...")
    
    # Use 1x1 mesh to avoid T3K L1 clash issues
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
    """Load SFace face recognition model."""
    from models.experimental.sface.tt.ttnn_sface import create_sface_model
    from models.experimental.sface.reference.sface_model import load_sface_from_onnx
    from models.experimental.sface.common import get_sface_onnx_path
    
    print("[SFace] Loading model...")
    onnx_path = get_sface_onnx_path()
    torch_model = load_sface_from_onnx(onnx_path)
    torch_model.eval()
    ttnn_model = create_sface_model(device, torch_model)
    
    # Warmup with correct input shape [B, H, W, C] NHWC
    print("[SFace] Warming up (compiling kernels)...")
    import ttnn
    warmup = torch.randint(0, 256, (1, 112, 112, 3), dtype=torch.float32)
    warmup_tt = ttnn.from_torch(warmup, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    _ = ttnn_model(warmup_tt)
    ttnn.synchronize_device(device)
    
    print("[SFace] Ready!")
    return ttnn_model


def preprocess_face(image: np.ndarray, target_size: int = 112) -> np.ndarray:
    """
    Preprocess LFW face image for SFace.
    
    LFW images are already face crops, so we just:
    1. Resize to 112x112
    2. Keep as 0-255 uint8 (SFace model does normalization internally)
    
    Args:
        image: numpy array [H, W, 3] RGB, 0-1 float or 0-255 uint8
        target_size: output size (112 for SFace)
        
    Returns:
        face: numpy array [112, 112, 3] uint8
    """
    # Convert 0-1 float to 0-255 uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Resize to target size
    face = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return face


def get_embedding(face_image: np.ndarray, sface_model, device) -> np.ndarray:
    """
    Get 128-dim embedding from face image.
    
    Args:
        face_image: [112, 112, 3] uint8 RGB image
        sface_model: TTNN SFace model
        device: TTNN device
        
    Returns:
        embedding: [128] float32 L2-normalized embedding
    """
    import ttnn
    
    # Convert to float32 and add batch dim: [1, 112, 112, 3] NHWC
    face_float = face_image.astype(np.float32)
    face_tensor = torch.from_numpy(face_float).unsqueeze(0)
    
    # Send to device
    tt_input = ttnn.from_torch(face_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    
    # Run inference
    tt_output = sface_model(tt_input)
    ttnn.synchronize_device(device)
    
    # Get embedding
    embedding = ttnn.to_torch(tt_output).float().numpy().flatten()
    
    return embedding


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    dot = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    return float(dot / (norm1 * norm2 + 1e-8))


def compute_tpr_at_far(scores: np.ndarray, labels: np.ndarray, target_far: float) -> Tuple[float, float]:
    """
    Compute TPR at a specific FAR.
    
    Args:
        scores: similarity scores for all pairs
        labels: 1 for same person, 0 for different
        target_far: target false acceptance rate (e.g., 0.001 for 0.1%)
        
    Returns: (tpr, threshold)
    """
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    
    # Sort thresholds from high to low
    thresholds = np.sort(np.unique(scores))[::-1]
    
    best_tpr = 0.0
    best_thresh = 0.0
    
    for thresh in thresholds:
        # FAR = false positives / total negatives
        fp = np.sum(negative_scores >= thresh)
        far = fp / len(negative_scores)
        
        # TPR = true positives / total positives  
        tp = np.sum(positive_scores >= thresh)
        tpr = tp / len(positive_scores)
        
        if far <= target_far and tpr > best_tpr:
            best_tpr = tpr
            best_thresh = thresh
    
    return best_tpr, best_thresh


def run_benchmark():
    """Run the full TPR@FAR benchmark."""
    print("=" * 60)
    print("  Face Recognition Accuracy Benchmark")
    print("  Target: >98% TPR @ 0.1% FAR")
    print("=" * 60)
    print()
    print("  NOTE: LFW images are pre-cropped faces")
    print("        → Skipping YuNet face detection")
    print("        → Directly resizing to 112x112 for SFace")
    print()
    
    # 1. Download dataset
    print("[1/4] Loading LFW dataset...")
    pairs, labels = download_lfw_dataset()
    if pairs is None:
        return
    
    # 2. Initialize device
    print("\n[2/4] Initializing Tenstorrent device...")
    device = init_device()
    
    try:
        # 3. Load model (only SFace needed for pre-cropped faces)
        print("\n[3/4] Loading SFace model...")
        sface = load_sface(device)
        
        # 4. Process all pairs
        print("\n[4/4] Processing pairs...")
        print(f"  Total pairs: {len(labels)}")
        
        scores = []
        times = []
        
        for i in range(len(labels)):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(labels)}...")
            
            t_start = time.time()
            
            # Get image pair
            img1 = pairs[i, 0]  # [62, 47, 3] or similar
            img2 = pairs[i, 1]
            
            # Preprocess: just resize to 112x112
            face1 = preprocess_face(img1, 112)
            face2 = preprocess_face(img2, 112)
            
            # Get embeddings
            emb1 = get_embedding(face1, sface, device)
            emb2 = get_embedding(face2, sface, device)
            
            # Compute similarity
            sim = cosine_similarity(emb1, emb2)
            scores.append(sim)
            times.append(time.time() - t_start)
        
        scores = np.array(scores)
        
        # 5. Compute metrics
        print("\n[5/5] Computing metrics...")
        
        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]
        
        # TPR at various FAR levels
        tpr_01, thresh_01 = compute_tpr_at_far(scores, labels, 0.001)  # 0.1%
        tpr_1, thresh_1 = compute_tpr_at_far(scores, labels, 0.01)     # 1%
        tpr_10, thresh_10 = compute_tpr_at_far(scores, labels, 0.1)    # 10%
        
        # Best accuracy
        thresholds = np.linspace(scores.min(), scores.max(), 1000)
        best_acc = 0
        best_acc_thresh = 0
        for t in thresholds:
            tp = np.sum(positive_scores >= t)
            tn = np.sum(negative_scores < t)
            acc = (tp + tn) / len(scores)
            if acc > best_acc:
                best_acc = acc
                best_acc_thresh = t
        
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
        print(f"  Mean Positive (same person): {positive_scores.mean():.4f}")
        print(f"  Std Positive: {positive_scores.std():.4f}")
        print(f"  Mean Negative (different): {negative_scores.mean():.4f}")
        print(f"  Std Negative: {negative_scores.std():.4f}")
        print(f"  Separation: {positive_scores.mean() - negative_scores.mean():.4f}")
        
        print(f"\n🎯 Best Accuracy:")
        print(f"  Accuracy: {best_acc*100:.2f}%")
        print(f"  Threshold: {best_acc_thresh:.4f}")
        
        print(f"\n⚡ Latency (per pair, 2 images):")
        print(f"  Mean: {np.mean(times)*1000:.1f} ms")
        print(f"  P95: {np.percentile(times, 95)*1000:.1f} ms")
        print(f"  Per-image: {np.mean(times)*1000/2:.1f} ms")
        
        print("\n" + "=" * 60)
        
        # Debug: print score distribution histogram
        print("\n📊 Score Histogram (for debugging):")
        print(f"  Positive scores: min={positive_scores.min():.3f}, max={positive_scores.max():.3f}")
        print(f"  Negative scores: min={negative_scores.min():.3f}, max={negative_scores.max():.3f}")
        
        return {
            "tpr_at_01_far": tpr_01,
            "tpr_at_1_far": tpr_1,
            "tpr_at_10_far": tpr_10,
            "best_accuracy": best_acc,
            "mean_positive": float(positive_scores.mean()),
            "mean_negative": float(negative_scores.mean()),
        }
        
    finally:
        import ttnn
        ttnn.close_device(device)
        print("\n[Device] Closed.")


if __name__ == "__main__":
    run_benchmark()
