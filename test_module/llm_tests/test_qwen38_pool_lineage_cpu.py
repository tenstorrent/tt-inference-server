# SPDX-License-Identifier: Apache-2.0
"""Pure lineage rejection tests; these do not qualify any real cache pool."""

import importlib.util
import unittest
from pathlib import Path


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


arm = load("prepare_qwen38_prefix_pool_arm")
work = load("reconcile_qwen38_prefill_work")


class LineageTest(unittest.TestCase):
    def test_reference_cannot_bypass_actual_environment(self):
        with self.assertRaisesRegex(ValueError, "actual0.30"):
            arm.selected_pool("reference", {}, {}, "image", study="fixed")

    def test_reference_cannot_mix_study(self):
        with self.assertRaisesRegex(ValueError, "same fixed study"):
            arm.selected_pool(
                "reference",
                {},
                dict(vllm_distribution_version="0.30.0", capacity_image_digest="image"),
                "image",
                study="fixed",
            )

    def test_hash_binds_content_and_ignores_dictionary_order(self):
        self.assertEqual(
            arm.receipt_digest(dict(a=1, b=2)), arm.receipt_digest(dict(b=2, a=1))
        )
        self.assertNotEqual(
            arm.receipt_digest(dict(a=1)), arm.receipt_digest(dict(a=2))
        )

    def test_missing_boundary_receipt_cannot_qualify(self):
        receipt = dict(
            vllm_distribution_version="0.30.0",
            capacity_image_digest="image",
            capacity_study_id="fixed",
            status="pass",
            close_mesh_device_returned=True,
            shared="0",
            physical_cache_pool=dict(actual_bytes_per_device=1),
        )
        selection = dict(
            arms={"0": dict(status="native_max_safe_verified_at_one_ID_boundary")}
        )
        # Missing workload attestation must fail before even considering the boundary hash.
        with self.assertRaisesRegex(ValueError, "registered workload"):
            arm.selected_pool("control_max", selection, receipt, "image", study="fixed")

    def test_passing_partial_stress_receipt_is_rejected(self):
        n = 1368
        row = dict(
            layers="all64",
            batch=32,
            native_context=262144,
            requested_outputs=128,
            configured_trace_region_bytes=128 * 1024**2,
            shared_mlp_readers=1,
            shared_mlp_cores=80,
            throw_exception_on_fallback=True,
            stress_cycles=1,
            status="pass",
            capacity_guards=dict(
                min_normal_free_bytes=512 * 1024**2,
                fragment_guard_bytes_per_bank=8 * 1024**2,
                min_trace_free_bytes=8 * 1024**2,
            ),
            capacity_pool_ids=n,
            records=[],
            physical_cache_pool=dict(
                actual_bytes_per_device=n * 17285120,
                global_ids=n,
                checkpoint_tokens=512,
                unique_snapshot_pools=16,
            ),
        )
        for phase in ("after_construction", "after_preparation"):
            row[phase] = dict(
                exact_allocator_reconciliation=True,
                independently_reconciled_allocator_domains=["DRAM", "TRACE"],
            )
        with self.assertRaisesRegex(ValueError, "every registered"):
            arm.validate_boundary_contract(row)
        row["status"] = "fail"
        with self.assertRaisesRegex(ValueError, "completed stress-case"):
            arm.validate_boundary_contract(row)

    def test_actual_repeated_work_and_warmup_exclusion(self):
        calls = [
            dict(
                arrival_wall=10,
                end_wall=20,
                uncached_prompt_tokens=512,
                eligible_prefix_missed_tokens=256,
            )
        ]
        events = [
            dict(event="prefill_chunk", time=t, pid=1, start=0, length=512, slots=[0])
            for t in [1, 11, 12]
        ]
        result = work.reconcile(calls, events, 0)
        self.assertEqual(result["worker_completed_logical_prefill_tokens"], 1024)
        self.assertEqual(result["additional_executed_logical_prefill_tokens"], 512)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            work.reconcile(calls, events + [events[-1]], 0)


if __name__ == "__main__":
    unittest.main()
