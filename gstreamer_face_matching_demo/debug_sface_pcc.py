#!/usr/bin/env python3
"""
Debug script to compare PyTorch reference SFace vs TTNN SFace.

This helps identify if the accuracy issue is:
1. In the reference model itself (bad ONNX weights)
2. In the TTNN porting (quantization, ops differences)

Usage:
    cd ~/teja/tt-metal
    source python_env/bin/activate
    python ~/teja/tt-inference-server/gstreamer_face_matching_demo/debug_sface_pcc.py
"""

import sys
import numpy as np
import cv2
import torch
from pathlib import Path

# Add tt-metal to path
TT_METAL_PATH = Path.home() / "teja" / "tt-metal"
sys.path.insert(0, str(TT_METAL_PATH))


def test_reference_model_on_lfw(max_pairs=100):
    """Test PyTorch reference model directly (no TTNN) on LFW."""
    from sklearn.datasets import fetch_lfw_pairs
    from models.experimental.sface.reference.sface_model import load_sface_from_onnx
    from models.experimental.sface.common import get_sface_onnx_path
    
    print("=" * 60)
    print("  SFace Reference Model Test (PyTorch, no TTNN)")
    print("=" * 60)
    
    # Load LFW
    print("\n[1/3] Loading LFW dataset...")
    lfw = fetch_lfw_pairs(subset='test', color=True, resize=1.0)
    pairs = lfw.pairs[:max_pairs]
    labels = lfw.target[:max_pairs]
    print(f"  Using {len(labels)} pairs (max_pairs={max_pairs})")
    
    # Load reference model
    print("\n[2/3] Loading PyTorch reference SFace...")
    onnx_path = get_sface_onnx_path()
    model = load_sface_from_onnx(onnx_path)
    model.eval()
    print("  Model loaded!")
    
    # Process pairs
    print("\n[3/3] Processing pairs with PyTorch reference...")
    scores = []
    
    with torch.no_grad():
        for i in range(len(labels)):
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(labels)}...")
            
            img1 = pairs[i, 0]
            img2 = pairs[i, 1]
            
            # Convert to uint8
            if img1.max() <= 1.0:
                img1 = (img1 * 255).astype(np.uint8)
                img2 = (img2 * 255).astype(np.uint8)
            
            # Resize to 112x112
            face1 = cv2.resize(img1, (112, 112))
            face2 = cv2.resize(img2, (112, 112))
            
            # Convert to tensor [B, C, H, W] - PyTorch expects NCHW!
            # Reference model does: x = (x - 127.5) * 0.0078125 internally
            # But it expects NCHW input, not NHWC!
            t1 = torch.from_numpy(face1.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
            t2 = torch.from_numpy(face2.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
            
            # Get embeddings
            emb1 = model(t1).numpy().flatten()
            emb2 = model(t2).numpy().flatten()
            
            # Cosine similarity
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
            scores.append(sim)
    
    scores = np.array(scores)
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    
    # Compute TPR at FAR
    def tpr_at_far(scores, labels, target_far):
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        thresholds = np.sort(np.unique(scores))[::-1]
        best_tpr, best_thresh = 0.0, 0.0
        for t in thresholds:
            far = np.sum(neg >= t) / len(neg)
            tpr = np.sum(pos >= t) / len(pos)
            if far <= target_far and tpr > best_tpr:
                best_tpr, best_thresh = tpr, t
        return best_tpr, best_thresh
    
    tpr_01, _ = tpr_at_far(scores, labels, 0.001)
    tpr_1, _ = tpr_at_far(scores, labels, 0.01)
    
    print("\n" + "=" * 60)
    print("  PYTORCH REFERENCE RESULTS")
    print("=" * 60)
    print(f"\n  TPR @ 0.1% FAR: {tpr_01*100:.2f}%")
    print(f"  TPR @ 1.0% FAR: {tpr_1*100:.2f}%")
    print(f"\n  Mean Positive: {positive_scores.mean():.4f} (std: {positive_scores.std():.4f})")
    print(f"  Mean Negative: {negative_scores.mean():.4f} (std: {negative_scores.std():.4f})")
    print(f"  Separation: {positive_scores.mean() - negative_scores.mean():.4f}")
    print("=" * 60)
    
    return scores, labels


def test_pcc_single_image():
    """Test PCC between PyTorch and TTNN on single image."""
    import ttnn
    import ttnn.distributed as dist
    from models.experimental.sface.reference.sface_model import load_sface_from_onnx
    from models.experimental.sface.tt.ttnn_sface import create_sface_model
    from models.experimental.sface.common import get_sface_onnx_path, SFACE_L1_SMALL_SIZE
    
    print("\n" + "=" * 60)
    print("  PCC Test: PyTorch vs TTNN")
    print("=" * 60)
    
    # Load PyTorch model
    print("\n[1/4] Loading PyTorch reference...")
    onnx_path = get_sface_onnx_path()
    torch_model = load_sface_from_onnx(onnx_path)
    torch_model.eval()
    
    # Initialize TTNN device
    print("\n[2/4] Initializing TTNN device...")
    mesh = dist.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[0],
        l1_small_size=SFACE_L1_SMALL_SIZE,
        trace_region_size=0
    )
    mesh.enable_program_cache()
    
    try:
        # Load TTNN model
        print("\n[3/4] Loading TTNN model...")
        ttnn_model = create_sface_model(mesh, torch_model)
        
        # Warmup
        warmup = torch.randint(0, 256, (1, 112, 112, 3), dtype=torch.float32)
        warmup_tt = ttnn.from_torch(warmup, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        _ = ttnn_model(warmup_tt)
        ttnn.synchronize_device(mesh)
        
        # Test with random image
        print("\n[4/4] Comparing outputs...")
        test_img = torch.randint(0, 256, (1, 112, 112, 3), dtype=torch.float32)
        
        # PyTorch expects NCHW
        torch_input = test_img.squeeze(0).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 112, 112]
        
        # TTNN expects NHWC
        ttnn_input = ttnn.from_torch(test_img, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        
        # Run both
        with torch.no_grad():
            torch_out = torch_model(torch_input).numpy().flatten()
        
        ttnn_out = ttnn_model(ttnn_input)
        ttnn.synchronize_device(mesh)
        ttnn_out = ttnn.to_torch(ttnn_out).float().numpy().flatten()
        
        # Compute PCC
        correlation = np.corrcoef(torch_out, ttnn_out)[0, 1]
        cosine_sim = np.dot(torch_out, ttnn_out) / (np.linalg.norm(torch_out) * np.linalg.norm(ttnn_out))
        max_diff = np.max(np.abs(torch_out - ttnn_out))
        
        print("\n" + "=" * 60)
        print("  PCC RESULTS")
        print("=" * 60)
        print(f"\n  Pearson Correlation (PCC): {correlation:.6f}")
        print(f"  Cosine Similarity: {cosine_sim:.6f}")
        print(f"  Max Absolute Difference: {max_diff:.6f}")
        print(f"\n  PyTorch embedding norm: {np.linalg.norm(torch_out):.4f}")
        print(f"  TTNN embedding norm: {np.linalg.norm(ttnn_out):.4f}")
        print(f"\n  PyTorch first 5 values: {torch_out[:5]}")
        print(f"  TTNN first 5 values: {ttnn_out[:5]}")
        print("=" * 60)
        
        if correlation > 0.99:
            print("\n✅ TTNN model matches PyTorch reference well!")
        elif correlation > 0.95:
            print("\n⚠️ TTNN model has some accuracy loss vs PyTorch")
        else:
            print("\n❌ TTNN model significantly differs from PyTorch!")
        
    finally:
        ttnn.close_device(mesh)


if __name__ == "__main__":
    # First test: Is reference model itself accurate on LFW?
    test_reference_model_on_lfw()
    
    # Second test: Does TTNN match PyTorch?
    test_pcc_single_image()
