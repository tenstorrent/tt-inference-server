#!/usr/bin/env python3
"""
Quick debug: Test PyTorch SFace with RGB vs BGR on LFW.
Also try different resize methods.
"""

import sys
import numpy as np
import cv2
import torch
from pathlib import Path

TT_METAL_PATH = Path.home() / "teja" / "tt-metal"
sys.path.insert(0, str(TT_METAL_PATH))

def test_rgb_vs_bgr():
    """Test if BGR conversion helps accuracy."""
    from sklearn.datasets import fetch_lfw_pairs
    from models.experimental.sface.reference.sface_model import load_sface_from_onnx
    from models.experimental.sface.common import get_sface_onnx_path
    
    print("Loading LFW (100 pairs)...")
    lfw = fetch_lfw_pairs(subset='test', color=True, resize=1.0)
    pairs = lfw.pairs[:100]
    labels = lfw.target[:100]
    
    print("Loading SFace...")
    model = load_sface_from_onnx(get_sface_onnx_path())
    model.eval()
    
    def run_test(name, preprocess_fn):
        scores = []
        with torch.no_grad():
            for i in range(len(labels)):
                img1 = pairs[i, 0]
                img2 = pairs[i, 1]
                
                # Convert 0-1 to 0-255
                if img1.max() <= 1.0:
                    img1 = (img1 * 255).astype(np.uint8)
                    img2 = (img2 * 255).astype(np.uint8)
                
                # Apply preprocessing
                face1 = preprocess_fn(img1)
                face2 = preprocess_fn(img2)
                
                # To tensor NCHW
                t1 = torch.from_numpy(face1.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
                t2 = torch.from_numpy(face2.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
                
                emb1 = model(t1).numpy().flatten()
                emb2 = model(t2).numpy().flatten()
                
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
                scores.append(sim)
        
        scores = np.array(scores)
        pos = scores[labels[:100] == 1]
        neg = scores[labels[:100] == 0]
        
        # Quick accuracy estimate
        acc = max(
            np.mean((scores >= 0.3) == labels[:100]),
            np.mean((scores >= 0.4) == labels[:100]),
            np.mean((scores >= 0.5) == labels[:100])
        )
        
        print(f"\n{name}:")
        print(f"  Mean Pos: {pos.mean():.3f}, Mean Neg: {neg.mean():.3f}")
        print(f"  Separation: {pos.mean() - neg.mean():.3f}")
        print(f"  Best Acc: {acc*100:.1f}%")
    
    # Test 1: RGB (current)
    def preprocess_rgb(img):
        return cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    
    # Test 2: BGR
    def preprocess_bgr(img):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
    
    # Test 3: RGB with INTER_CUBIC (higher quality)
    def preprocess_rgb_cubic(img):
        return cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)
    
    # Test 4: BGR with INTER_CUBIC
    def preprocess_bgr_cubic(img):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_CUBIC)
    
    # Test 5: RGB with padding to square first
    def preprocess_rgb_padded(img):
        h, w = img.shape[:2]
        if h != w:
            size = max(h, w)
            padded = np.zeros((size, size, 3), dtype=img.dtype)
            y_off = (size - h) // 2
            x_off = (size - w) // 2
            padded[y_off:y_off+h, x_off:x_off+w] = img
            img = padded
        return cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    
    print("\n" + "=" * 50)
    print("Testing preprocessing variants on 100 LFW pairs")
    print("=" * 50)
    
    run_test("1. RGB + INTER_LINEAR (current)", preprocess_rgb)
    run_test("2. BGR + INTER_LINEAR", preprocess_bgr)
    run_test("3. RGB + INTER_CUBIC", preprocess_rgb_cubic)
    run_test("4. BGR + INTER_CUBIC", preprocess_bgr_cubic)
    run_test("5. RGB + Padding + INTER_LINEAR", preprocess_rgb_padded)


if __name__ == "__main__":
    test_rgb_vs_bgr()
