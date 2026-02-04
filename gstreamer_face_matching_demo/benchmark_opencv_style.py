#!/usr/bin/env python3
"""
SFace Benchmark - OpenCV Zoo Style

Matches the evaluation methodology from:
https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface

Key differences from previous benchmarks:
1. Uses L2 distance (not cosine similarity)
2. Uses 10-fold cross-validation
3. Reports accuracy (not TPR@FAR)
4. L2 normalizes embeddings

Expected result: ~99.40% accuracy (matching OpenCV Zoo)

Usage:
    cd ~/teja/tt-metal
    source python_env/bin/activate
    export PYTHONPATH="/home/ttuser/.local/lib/python3.10/site-packages:$PYTHONPATH"
    
    # PyTorch reference
    python benchmark_opencv_style.py --backend pytorch
    
    # TTNN (Tenstorrent)
    python benchmark_opencv_style.py --backend ttnn
"""

import sys
import time
import pickle
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import KFold
import sklearn.preprocessing
from tqdm import tqdm

# Add tt-metal to path
TT_METAL_PATH = Path.home() / "teja" / "tt-metal"
sys.path.insert(0, str(TT_METAL_PATH))

# Dataset path (InsightFace pre-aligned)
DATA_DIR = Path.home() / "datasets" / "faces_webface_112x112"


def load_lfw_bin() -> Tuple[List[np.ndarray], np.ndarray]:
    """Load pre-aligned LFW pairs from InsightFace .bin format."""
    bin_path = DATA_DIR / "lfw.bin"
    
    print(f"[Dataset] Loading {bin_path}...")
    
    with open(bin_path, 'rb') as f:
        bins, issame = pickle.load(f, encoding='bytes')
    
    images = []
    for img_bytes in bins:
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # BGR, 112x112
        images.append(img)
    
    issame = np.array(issame)
    print(f"[Dataset] Loaded {len(images)} images, {len(issame)} pairs")
    
    return images, issame


# ============== OpenCV Zoo Evaluation Functions ==============
# From: https://github.com/opencv/opencv_zoo/blob/main/tools/eval/datasets/lfw.py

def calculate_accuracy(threshold, dist, actual_issame):
    """Calculate accuracy at given threshold."""
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    """Calculate ROC with 10-fold cross-validation (OpenCV Zoo method)."""
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    
    # L2 distance (OpenCV Zoo uses this, not cosine similarity)
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find best threshold on train set
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        
        # Evaluate on test set
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set])
        
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
    
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def evaluate_opencv_style(embeddings, actual_issame, nrof_folds=10):
    """Evaluate using OpenCV Zoo methodology."""
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    
    tpr, fpr, accuracy = calculate_roc(
        thresholds, embeddings1, embeddings2, 
        np.asarray(actual_issame), nrof_folds=nrof_folds
    )
    
    return np.mean(accuracy), np.std(accuracy)


# ============== Model Loading ==============

def load_pytorch_sface():
    """Load PyTorch SFace model."""
    from models.experimental.sface.reference.sface_model import load_sface_from_onnx
    from models.experimental.sface.common import get_sface_onnx_path
    
    model = load_sface_from_onnx(get_sface_onnx_path())
    model.eval()
    return model


def load_ttnn_sface(device):
    """Load TTNN SFace model."""
    import ttnn
    from models.experimental.sface.tt.ttnn_sface import create_sface_model
    from models.experimental.sface.reference.sface_model import load_sface_from_onnx
    from models.experimental.sface.common import get_sface_onnx_path
    
    pytorch_model = load_sface_from_onnx(get_sface_onnx_path())
    pytorch_model.eval()
    ttnn_model = create_sface_model(device, pytorch_model)
    
    # Warmup
    warmup = torch.randn(1, 112, 112, 3, dtype=torch.float32)
    warmup_tt = ttnn.from_torch(warmup, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    _ = ttnn_model(warmup_tt)
    ttnn.synchronize_device(device)
    
    return ttnn_model


def get_embedding_pytorch(face_bgr: np.ndarray, model) -> np.ndarray:
    """Get embedding using PyTorch (NCHW input)."""
    # PyTorch expects NCHW
    face_tensor = torch.from_numpy(face_bgr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    
    with torch.no_grad():
        embedding = model(face_tensor)
    
    return embedding.numpy().flatten()


def get_embedding_ttnn(face_bgr: np.ndarray, model, device) -> np.ndarray:
    """Get embedding using TTNN (NHWC input)."""
    import ttnn
    
    # TTNN expects NHWC
    face_tensor = torch.from_numpy(face_bgr.astype(np.float32)).unsqueeze(0)
    tt_input = ttnn.from_torch(face_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = model(tt_input)
    ttnn.synchronize_device(device)
    
    return ttnn.to_torch(tt_output).float().numpy().flatten()


def run_benchmark(backend: str = "pytorch"):
    """Run OpenCV Zoo style benchmark."""
    print("=" * 60)
    print(f"  SFace Benchmark - OpenCV Zoo Style")
    print(f"  Backend: {backend.upper()}")
    print(f"  Method: 10-fold CV, L2 distance, accuracy metric")
    print(f"  Expected: ~99.40% (matching OpenCV Zoo)")
    print("=" * 60)
    
    # 1. Load dataset
    print("\n[1/4] Loading LFW dataset (pre-aligned 112x112)...")
    images, issame = load_lfw_bin()
    
    # 2. Initialize model
    print(f"\n[2/4] Loading {backend.upper()} model...")
    
    device = None
    if backend == "ttnn":
        import ttnn
        import ttnn.distributed as dist
        from models.experimental.sface.common import SFACE_L1_SMALL_SIZE
        
        device = dist.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 1),
            physical_device_ids=[0],
            l1_small_size=SFACE_L1_SMALL_SIZE,
            trace_region_size=0
        )
        device.enable_program_cache()
        model = load_ttnn_sface(device)
        get_embedding = lambda img: get_embedding_ttnn(img, model, device)
    else:
        model = load_pytorch_sface()
        get_embedding = lambda img: get_embedding_pytorch(img, model)
    
    print(f"[{backend.upper()}] Ready!")
    
    try:
        # 3. Compute all embeddings
        print(f"\n[3/4] Computing embeddings for {len(images)} images...")
        embeddings = np.zeros((len(images), 128))
        times = []
        
        for idx, img in enumerate(tqdm(images, desc="Processing")):
            t_start = time.time()
            embeddings[idx] = get_embedding(img)
            times.append(time.time() - t_start)
        
        # 4. L2 normalize embeddings (OpenCV Zoo does this)
        print("\n[4/4] Evaluating with 10-fold cross-validation...")
        embeddings = sklearn.preprocessing.normalize(embeddings)
        
        # Evaluate
        acc_mean, acc_std = evaluate_opencv_style(embeddings, issame, nrof_folds=10)
        
        # Print results
        print("\n" + "=" * 60)
        print(f"  {backend.upper()} RESULTS (OpenCV Zoo Style)")
        print("=" * 60)
        
        opencv_ref = 0.9940
        match = "✅ MATCH" if abs(acc_mean - opencv_ref) < 0.01 else "⚠️ DIFF"
        
        print(f"\n📊 ACCURACY (10-fold CV):")
        print(f"  {backend.upper()}: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
        print(f"  OpenCV Zoo reference: {opencv_ref*100:.2f}%")
        print(f"  Comparison: {match}")
        
        print(f"\n⚡ LATENCY (per image):")
        print(f"  Mean: {np.mean(times)*1000:.1f} ms")
        print(f"  P95: {np.percentile(times, 95)*1000:.1f} ms")
        
        print("\n" + "=" * 60)
        
        return acc_mean, acc_std
        
    finally:
        if device is not None:
            import ttnn
            ttnn.close_device(device)
            print("\n[Device] Closed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SFace OpenCV Zoo Style Benchmark")
    parser.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "ttnn"])
    args = parser.parse_args()
    
    run_benchmark(backend=args.backend)
