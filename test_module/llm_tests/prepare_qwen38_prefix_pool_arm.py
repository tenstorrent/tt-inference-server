# SPDX-License-Identifier: Apache-2.0
"""Materialize reviewable checkpoint512 prefix-matrix patches; never deploys.

Max-pool arms require closed actual0.30 native receipts and a measured boundary
selection. Local0.26 tooling-calibration receipts cannot select serving pools.
"""

import argparse
import difflib
import json
import re
from pathlib import Path


def replace_once(text, pattern, replacement):
    result, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected one source-owned setting: {pattern}; got{count}")
    return result


def selected_pool(arm, selection, receipt, image):
    if arm == "reference":
        return 0, 1368, 23661297664
    shared = int(arm == "shared_max")
    if (receipt.get("vllm_distribution_version") or "").split("+")[0] != "0.30.0":
        raise ValueError(
            "Max-pool serving arm requires actual0.30 CI receipt; local native calibration is insufficient"
        )
    if receipt.get("capacity_image_digest") != image:
        raise ValueError(
            "Native receipt and serving matrix must use the same immutable image"
        )
    row = selection["arms"][str(shared)]
    if row["status"] != "native_max_safe_verified_at_one_ID_boundary":
        raise ValueError("Capacity search has not established an adjacent-ID boundary")
    if not (
        receipt["status"] == "pass"
        and receipt["close_mesh_device_returned"]
        and receipt["stress_cycles"] == 3
        and receipt["capacity_pool_ids"] == row["ids"]
        and receipt["shared"] == str(shared)
    ):
        raise ValueError("Expected closed three-cycle qualifying receipt for this arm")
    actual = receipt["physical_cache_pool"]["actual_bytes_per_device"]
    if actual != row["actual_physical_cache_bytes_per_device"]:
        raise ValueError("Selected IDs and actual physical bytes disagree")
    return shared, row["ids"], actual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arm", choices=["reference", "control_max", "shared_max"], required=True
    )
    p.add_argument("--selection", type=Path)
    p.add_argument("--receipt", type=Path)
    p.add_argument("--image", required=True)
    p.add_argument("--metal-ref", required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--tti-arm-ref")
    p.add_argument("--streams", type=int, choices=[32, 40], default=32)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", a.image):
        raise ValueError("Use the immutable reviewed source-overlay image digest")
    if not re.fullmatch(r"[0-9a-f]{40}", a.metal_ref):
        raise ValueError("Use the reviewed full metal source SHA")
    manifest = json.loads(a.source_manifest.read_text())
    if manifest.get(
        "metal_source"
    ) != a.metal_ref or "probe_shared_mlp_memory.py" not in manifest.get(
        "diagnostics_sha256", {}
    ):
        raise ValueError(
            "Image source manifest must attest reviewed runtime and capacity diagnostic bundle"
        )
    if a.arm != "reference" and (a.selection is None or a.receipt is None):
        raise ValueError(
            "Max-pool arms require real selection and qualification receipt"
        )
    selection = json.loads(a.selection.read_text()) if a.selection else {}
    receipt = json.loads(a.receipt.read_text()) if a.receipt else {}
    shared, ids, budget = selected_pool(a.arm, selection, receipt, a.image)
    root = Path(__file__).resolve().parents[2]
    spec = root / "workflows/model_specs/dev/llm.yaml"
    original = spec.read_text()
    start = original.index("# Goal 12:")
    end = original.index("\n# IFM/K2-Horizon", start)
    section = original[start:end]
    if "QWEN38_SHARED_MLP_WEIGHTS:" in section:
        raise ValueError(
            "Generate each arm against the unchanged probe base, not an already-patched arm"
        )
    if "max_concurrency: 32" not in section or "max_num_seqs: 32" not in section:
        raise ValueError(
            "Matrix requires fixed B32/model admission32, including A40 HTTP stage"
        )
    section = replace_once(
        section,
        r'(        QWEN38_CACHE_BUDGET_BYTES_PER_DEVICE: )"[0-9]+"',
        rf'\g<1>"{budget}"',
    )
    section = replace_once(
        section,
        r'        QWEN38_FLAT_PREFILL: "1"',
        '        QWEN38_FLAT_PREFILL: "1"\n'
        f'        QWEN38_SHARED_MLP_WEIGHTS: "{shared}"\n'
        '        QWEN38_SHARED_MLP_READERS: "1"\n'
        '        QWEN38_SHARED_MLP_CORES: "80"\n'
        '        QWEN38_MEMORY_ATTESTATION: "1"\n'
        '        QWEN38_MEMORY_CHECKPOINTS: "1"',
    )
    updated = original[:start] + section + original[end:]
    control = root / "reference_config/evals/qwen38_goal12.json"
    original_control = control.read_text()
    config = json.loads(original_control)
    probe = config["prefix_retention_probe"]
    if probe["block_size"] != 512:
        raise ValueError(
            "Primary causal pair fixes checkpoint512;2048 is a separate factor"
        )
    probe.update(streams=a.streams, concurrency=a.streams)
    updated_control = json.dumps(config, indent=2) + "\n"
    a.output_dir.mkdir(parents=True, exist_ok=True)
    patch = ""
    for path, before, after in [
        (spec, original, updated),
        (control, original_control, updated_control),
    ]:
        rel = path.relative_to(root).as_posix()
        patch += "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/" + rel,
                tofile="b/" + rel,
            )
        )
    (a.output_dir / "arm.patch").write_text(patch)
    report = dict(
        status="prepared_not_dispatched_not_serving_qualified",
        arm=a.arm,
        shared=shared,
        model_B=32,
        model_max_num_seqs=32,
        HTTP_streams=a.streams,
        checkpoint_tokens=512,
        global_ids=ids,
        selected_actual_cache_bytes_per_device=(
            receipt.get("physical_cache_pool") or {}
        ).get("actual_bytes_per_device"),
        configured_cache_budget_bytes_per_device=budget,
        image=a.image,
        metal_ref=a.metal_ref,
        acceptance="Apply/review/commit patch in an isolated TTI arm; serving startup and memory/cached-prefix/quality gates still required.",
    )
    (a.output_dir / "arm.json").write_text(json.dumps(report, indent=2) + "\n")
    if a.tti_arm_ref:
        if not re.fullmatch(r"[0-9a-f]{40}", a.tti_arm_ref):
            raise ValueError("Use reviewed full TTI arm SHA")
        inputs = {
            "build-qwen38-overlay": "false",
            "build-only": "false",
            "model": "Qwen3.8-27B",
            "runner-label": "bh-qb-ge",
            "device-type": "p300x2",
            "workflow": "agentic",
            "docker-image": a.image,
            "tt-metal-git-ref": a.metal_ref,
            "inference-server-git-ref": a.tti_arm_ref,
            "vllm-git-ref": "1badbd02bb6911dfdb59c206bdc00c6c1e8dc0bf",
            "impl-of-model": "qwen38-autoport",
            "run-ai-summary": "false",
            "create-issue-comment": "false",
        }
        (a.output_dir / "dispatch-inputs.json").write_text(
            json.dumps(inputs, indent=2) + "\n"
        )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
