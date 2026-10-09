# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Check image contents and imports without opening hardware or claiming accuracy."""

import hashlib
import importlib.metadata
import json
import os
import argparse
from pathlib import Path


def verify(bundle, expected_pins=None):
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in (expected_pins or {}).items():
        if expected != manifest[name]:
            raise ValueError(f"Build argument differs from the manifest: {name}")
    model = Path(os.environ["TT_METAL_HOME"]) / "models/demos/qwen38_27b_qb2"
    policy = model / "config" / manifest["precision_file"]
    os.environ["QWEN_PRECISION_CONFIG"] = str(policy)
    for name, digest in manifest["model_source_sha256"].items():
        path = model / (
            "config/" + manifest["precision_file"]
            if name == "effective_precision_override"
            else name
        )
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Model source differs from the G0 receipt: {name}")
    for package, expected in manifest["versions"].items():
        actual = importlib.metadata.version(package)
        if actual.split("+", 1)[0] != expected:
            raise ValueError(f"{package}: expected {expected}, got {actual}")
    from models.demos.qwen38_27b_qb2.demo.galaxy_serving import (
        qualified_groups,
        verify_qualified_source,
    )
    from models.demos.qwen38_27b_qb2.tt.precision import (
        load_precision,
        precision_fingerprint,
    )

    receipt_bytes = (bundle / "g0.json").read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != manifest["g0_receipt_sha256"]:
        raise ValueError("G0 receipt differs from the manifest")
    receipt = json.loads(receipt_bytes)
    qualified_groups(receipt)
    verify_qualified_source(receipt, model)
    if precision_fingerprint(load_precision()) != manifest["precision_sha256"]:
        raise ValueError("Effective model precision differs from the manifest")

    import ttnn
    from models.demos.qwen38_27b_qb2.tt.generator_vllm import Qwen38ForCausalLM

    if ttnn is None or Qwen38ForCausalLM is None:
        raise RuntimeError("Model or native runtime failed to import")
    print(
        "QWEN_IMAGE_SOURCE_AND_IMPORTS_PASS; no hardware opened; accuracy not qualified"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for component in ("metal", "model", "plugin"):
        parser.add_argument(f"--{component}-sha")
    pins = {
        key: value
        for key, value in vars(parser.parse_args()).items()
        if value is not None
    }
    verify(Path(__file__).resolve().parent, pins)
