# SPDX-License-Identifier: Apache-2.0
"""Materialize reviewable checkpoint512 prefix-matrix patches; never deploys.

Max-pool arms require closed actual0.30 native receipts and a measured boundary
selection. Local0.26 tooling-calibration receipts cannot select serving pools.
"""

import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path


def replace_once(text, pattern, replacement):
    result, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected one source-owned setting: {pattern}; got{count}")
    return result


def receipt_digest(row):
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def campaign_identity(row):
    keys = (
        "source_sha256",
        "runtime",
        "capacity_image_digest",
        "vllm_distribution_version",
        "checkpoint_file_sha256",
        "effective_precision",
        "capacity_study_id",
        "capacity_guards",
        "environment",
        "shared_mlp_readers",
        "shared_mlp_cores",
        "layers",
        "batch",
        "native_context",
        "requested_outputs",
        "configured_trace_region_bytes",
        "throw_exception_on_fallback",
        "layer_types",
    )
    return {key: row.get(key) for key in keys}


def validate_boundary_contract(row):
    expected = dict(
        layers="all64",
        batch=32,
        native_context=262144,
        requested_outputs=128,
        configured_trace_region_bytes=128 * 1024**2,
        shared_mlp_readers=1,
        shared_mlp_cores=80,
        throw_exception_on_fallback=True,
    )
    if any(row.get(k) != value for k, value in expected.items()):
        raise ValueError("Unsafe boundary changed the registered workload/geometry")
    if row.get("capacity_guards") != dict(
        min_normal_free_bytes=512 * 1024**2,
        fragment_guard_bytes_per_bank=8 * 1024**2,
        min_trace_free_bytes=8 * 1024**2,
    ):
        raise ValueError("Receipt changed registered capacity guards")
    if row.get("stress_cycles") not in (1, 3):
        raise ValueError("Unexpected stress cycle contract")
    pool = row.get("physical_cache_pool", {})
    n = row["capacity_pool_ids"]
    actual = n * 17285120 + ((-3 * n) % 8) * 5120 * 16
    if (
        pool.get("actual_bytes_per_device") != actual
        or pool.get("global_ids") != n
        or pool.get("checkpoint_tokens") != 512
        or pool.get("unique_snapshot_pools") != 16
    ):
        raise ValueError("Unsafe boundary lacks exact physical pool ownership")
    for phase in ("after_construction", "after_preparation"):
        allocation = row.get(phase, {})
        if not allocation.get("exact_allocator_reconciliation") or allocation.get(
            "independently_reconciled_allocator_domains"
        ) != ["DRAM", "TRACE"]:
            raise ValueError("Unsafe boundary lacks reconciled allocator domains")
    records = row.get("records", [])
    expected_lengths = [513, 4096, 16384] * row["stress_cycles"]
    if [r["length"] for r in records] != expected_lengths[: len(records)] or len(
        records
    ) > len(expected_lengths):
        raise ValueError("Unsafe boundary changed completed workload prefix")
    if row.get("status") == "pass" and len(records) != len(expected_lengths):
        raise ValueError("Passing receipt must complete every registered stress case")
    if row.get("status") != "pass" and not records:
        raise ValueError("Unsafe boundary must include a completed stress-case witness")
    for r in records:
        if (
            len(r.get("tokens", [])) != 32
            or any(len(t) != 128 for t in r["tokens"])
            or len(r.get("logits", [])) != 4
            or not r.get("allocation", {}).get("exact_allocator_reconciliation")
        ):
            raise ValueError(
                "Unsafe boundary has incomplete completed-request evidence"
            )


def selected_pool(arm, selection, receipt, image, unsafe=None, study=None):
    shared = int(arm == "shared_max")
    if (receipt.get("vllm_distribution_version") or "").split("+")[0] != "0.30.0":
        raise ValueError("Every arm requires an actual0.30 CI receipt")
    if receipt.get("capacity_image_digest") != image:
        raise ValueError(
            "Native receipt and serving matrix must use the same immutable image"
        )
    if not study or receipt.get("capacity_study_id") != study:
        raise ValueError(
            "Native receipt and HTTP salts must belong to the same fixed study"
        )
    if not (
        receipt.get("status") == "pass"
        and receipt.get("close_mesh_device_returned")
        and receipt.get("shared") == str(shared)
    ):
        raise ValueError("Expected a closed qualifying receipt for this arm")
    actual = receipt["physical_cache_pool"]["actual_bytes_per_device"]
    validate_boundary_contract(receipt)
    if arm == "reference":
        if receipt["capacity_pool_ids"] != 1368:
            raise ValueError("Reference requires the matched1368-ID control receipt")
        return 0, 1368, 23661297664
    row = selection["arms"][str(shared)]
    if row["status"] != "native_max_safe_verified_at_one_ID_boundary":
        raise ValueError("Capacity search has not established an adjacent-ID boundary")
    if (
        not unsafe
        or receipt_digest(receipt) != row.get("selected_receipt_sha256")
        or receipt_digest(unsafe) != row.get("unsafe_receipt_sha256")
    ):
        raise ValueError(
            "Selection must bind both original qualifying and unsafe receipts"
        )
    identity = campaign_identity(receipt)
    if identity != campaign_identity(unsafe) or identity != row.get(
        "campaign_identity"
    ):
        raise ValueError("Pass and unsafe boundary have different campaign identities")
    if not (
        receipt["stress_cycles"] == 3
        and receipt["capacity_pool_ids"] == row["ids"]
        and unsafe.get("shared") == str(shared)
        and unsafe.get("status") != "pass"
        and unsafe.get("close_mesh_device_returned") is True
        and not unsafe.get("cleanup_error")
        and not unsafe.get("close_error")
        and unsafe.get("error_type") == "AssertionError"
        and unsafe.get("capacity_pool_ids") == row["ids"] + 1
        and any(
            t in unsafe.get("error", "")
            for t in (
                "Transient headroom failed",
                "TRACE headroom failed",
                "Fragmentation guard failed",
            )
        )
    ):
        raise ValueError(
            "Expected closed three-cycle pass and adjacent known guard failure"
        )
    validate_boundary_contract(unsafe)
    witnesses = {
        r["length"]: (r["input_sha256"], r["tokens"], r["logits"])
        for r in receipt["records"]
    }
    for r in unsafe.get("records", []):
        if all(
            k in r for k in ("length", "input_sha256", "tokens", "logits")
        ) and witnesses[r["length"]] != (r["input_sha256"], r["tokens"], r["logits"]):
            raise ValueError(
                "Completed unsafe-trial output differs from qualifying witness"
            )
    if actual != row["actual_physical_cache_bytes_per_device"]:
        raise ValueError("Selected IDs and actual physical bytes disagree")
    return shared, row["ids"], actual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arm", choices=["reference", "control_max", "shared_max"], required=True
    )
    p.add_argument("--selection", type=Path)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--unsafe-receipt", type=Path)
    p.add_argument("--study-namespace", default="qwen38-maxkv-512-v1")
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
    unsafe = json.loads(a.unsafe_receipt.read_text()) if a.unsafe_receipt else None
    shared, ids, budget = selected_pool(
        a.arm, selection, receipt, a.image, unsafe, a.study_namespace
    )
    for path, digest in receipt["source_sha256"].items():
        namespace, name = path.split("/", 1)
        expected = manifest.get(
            "sha256" if namespace == "tt" else "diagnostics_sha256", {}
        ).get(name)
        if expected != digest:
            raise ValueError(f"Native receipt does not match image manifest: {path}")
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
        '        QWEN38_SHARED_MLP_CORES: "80"',
    )
    for flag in ("QWEN38_MEMORY_ATTESTATION", "QWEN38_MEMORY_CHECKPOINTS"):
        if f"        {flag}:" not in section:
            section = section.replace(
                '        QWEN38_FLAT_PREFILL: "1"',
                f'        {flag}: "1"\n        QWEN38_FLAT_PREFILL: "1"',
            )
    # Qualified serving arms reuse the closed native receipts; they must not
    # repeat the full capacity search before each fresh server.
    section = section.replace('QWEN38_RUN_CAPACITY_SCREEN: "1"', 'QWEN38_RUN_CAPACITY_SCREEN: "0"')
    updated = original[:start] + section + original[end:]
    control = root / "reference_config/evals/qwen38_goal12.json"
    original_control = control.read_text()
    config = json.loads(original_control)
    probe = config["prefix_retention_probe"]
    if probe["block_size"] != 512:
        raise ValueError(
            "Primary causal pair fixes checkpoint512;2048 is a separate factor"
        )
    probe.update(
        streams=a.streams, concurrency=a.streams, study_namespace=a.study_namespace
    )
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
        study_namespace=a.study_namespace,
        fresh_server_required=True,
        selected_receipt_sha256=receipt_digest(receipt),
        unsafe_receipt_sha256=receipt_digest(unsafe) if unsafe else None,
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
