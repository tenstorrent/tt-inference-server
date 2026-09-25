# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Validate SDXL matrix expansion in v2's image.json and video.json."""

from __future__ import annotations

from pathlib import Path

from test_module.test_categorization_system.suite_loader import (
    load_server_tests_config,
    load_suite_files_by_category,
)


def test_diffusiongemma_llm_suite_uses_model_specific_conformance_test():
    suites = load_suite_files_by_category("llm")
    suite = next(
        suite for suite in suites if suite["id"] == "diffusiongemma-26b-a4b-it-p300x2"
    )

    assert suite["weights"] == ["google/diffusiongemma-26B-A4B-it"]
    assert suite["device"] == "p300x2"
    assert [case["template"] for case in suite["test_cases"]] == [
        "VLLMDiffusionGemmaParamConformanceTest"
    ]

    templates = load_server_tests_config()["test_templates"]
    template = templates["VLLMDiffusionGemmaParamConformanceTest"]
    assert template["module"] == "test_module.llm_tests.vllm_param_conformance_test"
    assert {"param", "e2e", "slow", "heavy"} <= set(template["markers"])


def test_llama_3_2_1b_p300x2_uses_vllm_param_conformance_suite():
    suites = load_suite_files_by_category("llm")
    matching = [
        suite
        for suite in suites
        if suite["weights"] == ["meta-llama/Llama-3.2-1B-Instruct"]
    ]

    assert [(suite["id"], suite["device"]) for suite in matching] == [
        ("llama-3.2-1b-p300x2", "p300x2")
    ]
    assert [case["template"] for case in matching[0]["test_cases"]] == [
        "VLLMParamConformanceTest"
    ]


class TestImageMatrixExpansionSDXL:
    SDXL_SUITE_IDS = {
        "sdxl-n150",
        "sdxl-t3k",
        "sdxl-galaxy",
        "sdxl-n300",
        "sdxl-p150x8",
        "sdxl-p300x2",
        "sdxl-img2img-n150",
        "sdxl-img2img-t3k",
        "sdxl-img2img-galaxy",
        "sdxl-inpaint-n150",
        "sdxl-inpaint-t3k",
        "sdxl-inpaint-galaxy",
    }

    def test_sdxl_suite_ids_present(self):
        suites = load_suite_files_by_category("image")
        ids = {s["id"] for s in suites}
        missing = self.SDXL_SUITE_IDS - ids
        assert not missing, f"SDXL suites missing from v2 image.json: {missing}"

    def test_sdxl_full_lora_suites(self):
        """n150, t3k, galaxy should have 6 test cases including LoRA tests."""
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        for suite_id in ["sdxl-n150", "sdxl-t3k", "sdxl-galaxy"]:
            suite = suite_map[suite_id]
            assert len(suite["test_cases"]) == 6, f"{suite_id}: expected 6 test cases"
            templates = [tc["template"] for tc in suite["test_cases"]]
            assert "ImageGenerationEvalsTest" in templates
            assert "ImageGenerationLoraLoadTest" in templates

    def test_sdxl_galaxy_timing_differs(self):
        """Per-device LoadTest timing: galaxy 20/28/45, n150 12/16/23 (Forge
        path ~11-15s/img), t3k 10/14/23."""
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        galaxy = suite_map["sdxl-galaxy"]
        load_tests = [
            tc
            for tc in galaxy["test_cases"]
            if tc["template"] == "ImageGenerationLoadTest"
        ]
        times = [lt["targets"]["image_generation_time"] for lt in load_tests]
        assert times == [20, 28, 45]

        n150 = suite_map["sdxl-n150"]
        load_tests = [
            tc
            for tc in n150["test_cases"]
            if tc["template"] == "ImageGenerationLoadTest"
        ]
        times = [lt["targets"]["image_generation_time"] for lt in load_tests]
        assert times == [12, 16, 23]

        t3k = suite_map["sdxl-t3k"]
        load_tests = [
            tc
            for tc in t3k["test_cases"]
            if tc["template"] == "ImageGenerationLoadTest"
        ]
        times = [lt["targets"]["image_generation_time"] for lt in load_tests]
        assert times == [10, 14, 23]

    def test_sdxl_reduced_suites(self):
        """n300, p150x8, p300x2 should have 4 test cases (no LoRA)."""
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        for suite_id in ["sdxl-n300", "sdxl-p150x8", "sdxl-p300x2"]:
            suite = suite_map[suite_id]
            assert len(suite["test_cases"]) == 4, f"{suite_id}: expected 4 test cases"

    def test_sdxl_reduced_num_devices(self):
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        assert suite_map["sdxl-n300"]["num_of_devices"] == 1
        assert suite_map["sdxl-p150x8"]["num_of_devices"] == 4
        assert suite_map["sdxl-p300x2"]["num_of_devices"] == 2

    def test_sdxl_img2img_suites_present(self):
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        for suite_id in [
            "sdxl-img2img-n150",
            "sdxl-img2img-t3k",
            "sdxl-img2img-galaxy",
        ]:
            suite = suite_map[suite_id]
            templates = [tc["template"] for tc in suite["test_cases"]]
            assert templates == ["Img2ImgGenerationParamTest"]

    def test_sdxl_inpaint_suites_present(self):
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        for suite_id in [
            "sdxl-inpaint-n150",
            "sdxl-inpaint-t3k",
            "sdxl-inpaint-galaxy",
        ]:
            suite = suite_map[suite_id]
            templates = [tc["template"] for tc in suite["test_cases"]]
            assert templates == ["InpaintingGenerationParamTest"]


class TestVideoMatrixExpansion:
    """Regression coverage for video.json matrix expansion + per-device targets.

    VIDEO is served exclusively by v2 (routed via workflows/workflow_dispatch), so this
    suite is the only expansion/target guard for live video config — the v1
    server_tests video.json and its TestVideoMatrixExpansion were removed.
    """

    # Expected expanded per-device suite ids (one per model x device).
    EXPECTED_SUITE_IDS = {
        "wan-t3k",
        "wan-galaxy",
        "wan-p150x4",
        "wan-p150x8",
        "wan-p300x2",
        "wan-i2v-t3k",
        "wan-i2v-galaxy",
        "wan-i2v-p150x4",
        "wan-i2v-p150x8",
        "wan-i2v-p300x2",
        "mochi-t3k",
        "mochi-galaxy",
        "mochi-p150x4",
        "mochi-p150x8",
        "mochi-p300x2",
        "minimax-h3-blackhole_galaxy",
    }

    # Expected VideoGenerationLoadTest targets per expanded suite: the base
    # template targets merged with per-device model_targets overrides. p150x4
    # has no override, so it keeps the bare base target (no poll_timeout).
    WAN_LOAD_TARGETS = {
        "wan-t3k": {"video_generation_target_time": 1200, "poll_timeout": 1500},
        "wan-galaxy": {"video_generation_target_time": 250, "poll_timeout": 550},
        "wan-p150x4": {"video_generation_target_time": 370},
        "wan-p150x8": {"video_generation_target_time": 600, "poll_timeout": 900},
        "wan-p300x2": {"video_generation_target_time": 500, "poll_timeout": 800},
    }
    MOCHI_LOAD_TARGETS = {
        "mochi-p150x4": {
            "video_generation_target_time": 480,
            "num_inference_steps": 50,
        },
        "mochi-p300x2": {
            "video_generation_target_time": 900,
            "num_inference_steps": 50,
            "poll_timeout": 1100,
        },
        "mochi-t3k": {
            "video_generation_target_time": 600,
            "num_inference_steps": 50,
            "poll_timeout": 900,
        },
        "mochi-galaxy": {
            "video_generation_target_time": 650,
            "num_inference_steps": 50,
            "poll_timeout": 800,
        },
        "mochi-p150x8": {
            "video_generation_target_time": 900,
            "num_inference_steps": 50,
            "poll_timeout": 1000,
        },
    }
    # Expected VideoGenerationI2VTest targets per expanded suite. p150x4 keeps
    # the base poll_timeout (no per-device override).
    WAN_I2V_TARGETS = {
        "wan-i2v-t3k": {
            "num_inference_steps": 40,
            "poll_timeout": 1500,
            "poll_interval": 5,
        },
        "wan-i2v-galaxy": {
            "num_inference_steps": 40,
            "poll_timeout": 550,
            "poll_interval": 5,
        },
        "wan-i2v-p150x4": {
            "num_inference_steps": 40,
            "poll_timeout": 1200,
            "poll_interval": 5,
        },
        "wan-i2v-p150x8": {
            "num_inference_steps": 40,
            "poll_timeout": 900,
            "poll_interval": 5,
        },
        "wan-i2v-p300x2": {
            "num_inference_steps": 40,
            "poll_timeout": 800,
            "poll_interval": 5,
        },
    }

    @staticmethod
    def _suite_map():
        suites = load_suite_files_by_category("video")
        return {s["id"]: s for s in suites}

    def _case_targets(self, suite_id, template):
        cases = [
            tc
            for tc in self._suite_map()[suite_id]["test_cases"]
            if tc["template"] == template
        ]
        assert len(cases) == 1, f"{suite_id}: expected exactly one {template}"
        return cases[0]["targets"]

    def test_video_suite_ids_match(self):
        assert set(self._suite_map()) == self.EXPECTED_SUITE_IDS

    def test_t2v_suites_have_load_and_param_cases(self):
        suite_map = self._suite_map()
        for suite_id in ["wan-t3k", "wan-p150x4", "mochi-t3k", "mochi-p300x2"]:
            templates = [tc["template"] for tc in suite_map[suite_id]["test_cases"]]
            assert templates == [
                "VideoGenerationLoadTest",
                "VideoGenerationParamTest",
            ], f"{suite_id}: unexpected templates {templates}"

    def test_i2v_suites_include_i2v_param_test(self):
        # The I2V param sweep is a live test case; guard that every I2V suite
        # still carries both the happy-path and the param sweep after expansion.
        suite_map = self._suite_map()
        for suite_id in self.WAN_I2V_TARGETS:
            templates = [tc["template"] for tc in suite_map[suite_id]["test_cases"]]
            assert templates == [
                "VideoGenerationI2VTest",
                "VideoGenerationI2VParamTest",
            ], f"{suite_id}: unexpected templates {templates}"

    def test_all_video_suites_single_device(self):
        for suite in self._suite_map().values():
            assert suite["num_of_devices"] == 1

    def test_minimax_h3_suite_contract_lifecycle_and_benchmark(self):
        # On this dispatch branch the single-host BH Galaxy serves ref2va; the suite drives the
        # V1 contract, one judged lifecycle, then every REF2VA / SIZE h3-benchmark case. Cancel stays disabled until a
        # single-host cancel is verified not to take the deployment down.
        suite = self._suite_map()["minimax-h3-blackhole_galaxy"]
        enabled = [
            tc["template"] for tc in suite["test_cases"] if tc.get("enabled", True)
        ]
        assert (
            enabled
            == [
                "MiniMaxH3CreateContractTest",
                "MiniMaxH3LifecycleDownloadTest",
            ]
            + ["MiniMaxH3BenchmarkTest"] * 5
        )
        disabled = [
            tc["template"] for tc in suite["test_cases"] if not tc.get("enabled", True)
        ]
        assert disabled == ["MiniMaxH3CancelLifecycleTest"]
        # Every REF2VA / SIZE case exactly once, split over five entries (one pass is ~5.5 h
        # and an entry has 14400 s), rising risk, smoke only in the first.
        from test_module._test_common.minimax_h3_bench import models as M

        benches = [
            tc["targets"]
            for tc in suite["test_cases"]
            if tc["template"] == "MiniMaxH3BenchmarkTest"
        ]
        ref2va = [c["id"] for c in M.load_cases()["cases"] if c["task"] == "ref2va"]
        for key, runs in (("plan_ci", 1), ("plan_full", 3)):
            planned = [cid for b in benches for item in b[key] for cid in item["cases"]]
            assert sorted(planned) == sorted(ref2va), key
            assert {item["runs"] for b in benches for item in b[key]} == {runs}, key
        assert [b["task"] for b in benches] == ["ref2va"] * 5
        assert [b["timeout_table"] for b in benches] == ["BH1X"] * 5
        assert [b["skip_smoke"] for b in benches] == [False] + [True] * 4
        assert len({b["out_subdir"] for b in benches}) == 5
        assert benches[-1]["plan_ci"] == [{"cases": ["REF2VA-H"], "runs": 1}]
        # The expanded case carries only template/targets; its budget is the
        # template's test_config, which BaseTest reads as config["timeout"]
        # ("test_timeout" is a dead key there).
        templates = load_server_tests_config()["test_templates"]
        bench_config = templates["MiniMaxH3BenchmarkTest"]["test_config"]
        assert bench_config["timeout"] == 14400
        assert "test_timeout" not in bench_config
        for template in (
            "MiniMaxH3CreateContractTest",
            "MiniMaxH3LifecycleDownloadTest",
            "MiniMaxH3CancelLifecycleTest",
            "MiniMaxH3VideoQualityTest",
        ):
            assert "timeout" in templates[template]["test_config"], template
            assert "test_timeout" not in templates[template]["test_config"], template

    @staticmethod
    def _minimax_h3_spec_runner():
        """MODEL_RUNNER of the MiniMax-H3 BLACKHOLE_GALAXY dev spec, device env on top, as
        the benchmark reads it from ctx.model_spec.env_vars at run time."""
        import yaml

        path = (
            Path(__file__).resolve().parents[2] / "workflows/model_specs/dev/video.yaml"
        )
        (spec,) = [
            t
            for t in yaml.safe_load(path.read_text())["templates"]
            if "MiniMaxAI/MiniMax-H3" in t["weights"]
        ]
        (device,) = [
            d for d in spec["device_model_specs"] if d["device"] == "BLACKHOLE_GALAXY"
        ]
        env = {**spec.get("env_vars", {}), **device.get("env_vars", {})}
        return env["MODEL_RUNNER"]

    def test_minimax_h3_benchmark_entries_serve_the_spec_task(self, monkeypatch):
        # The benchmark takes its task from the spec's MODEL_RUNNER and fails a suite entry
        # whose targets.task contradicts it; every planned case must belong to that task. In
        # both modes (--ci-mode and run-full-evals) the entries of one suite must run
        # different cases into different directories, and each plan must fit its entry's
        # deadline at the pace the budgets assume: the cases it never starts are skipped and
        # fail the test on a healthy deployment.
        from types import SimpleNamespace

        from test_module._test_common import TestConfig
        from test_module._test_common.minimax_h3_bench import models as M
        from test_module.load_param_tests import minimax_h3_benchmark_test as T

        runner = self._minimax_h3_spec_runner()
        assert runner in T.RUNNER_TASKS, runner
        by_id = {c["id"]: c for c in M.load_cases()["cases"]}
        templates = load_server_tests_config()["test_templates"]
        template_config = templates["MiniMaxH3BenchmarkTest"]["test_config"]
        checked = 0
        for suite_id, suite in self._suite_map().items():
            entries = [
                tc
                for tc in suite["test_cases"]
                if tc["template"] == "MiniMaxH3BenchmarkTest"
                and tc.get("enabled", True)
            ]
            for ci in (True, False):
                ctx = SimpleNamespace(
                    model_spec=SimpleNamespace(env_vars={"MODEL_RUNNER": runner}),
                    runtime_config=SimpleNamespace(ci_mode=ci, limit_samples_mode=None),
                    output_path="/output",
                    service_port=8000,
                    base_url="http://127.0.0.1:8000",
                )
                monkeypatch.setattr(T, "_OUT_DIR_OWNERS", {})
                plans, dirs = set(), set()
                for tc in entries:
                    targets = tc.get("targets") or {}
                    config = {**template_config, **(tc.get("test_config") or {})}
                    test = T.MiniMaxH3BenchmarkTest(
                        TestConfig(config), targets, ctx=ctx
                    )
                    task = test._task()  # raises on a mismatch
                    plan = test._plan(task)
                    ids = tuple(cid for item in plan for cid in item["cases"])
                    where = (suite_id, "ci" if ci else "full", ids)
                    assert {by_id[cid]["task"] for cid in ids} == {task}, where
                    assert ids not in plans, f"{where}: another entry runs these cases"
                    plans.add(ids)
                    out = test._out_dir(plan)  # raises on a shared out_subdir
                    assert out not in dirs, (where, out)
                    dirs.add(out)
                    table = str(targets.get("timeout_table", M.DEFAULT_TIMEOUT_TABLE))
                    skip = bool(targets.get("skip_smoke", False))
                    need, deadline = (
                        T.plan_estimate_s(task, plan, table, skip),
                        test._deadline_s(),
                    )
                    assert need <= deadline, (
                        f"{where}: ~{need}s of generation at the {table} pace, deadline "
                        f"{deadline:.0f}s: raise the entry's test_config.timeout or split the plan"
                    )
                    checked += 1
        assert checked, "no enabled MiniMaxH3BenchmarkTest entry"

    def test_wan_load_targets_merge_per_device(self):
        for suite_id, expected in self.WAN_LOAD_TARGETS.items():
            assert (
                self._case_targets(suite_id, "VideoGenerationLoadTest") == expected
            ), suite_id

    def test_mochi_load_targets_merge_per_device(self):
        for suite_id, expected in self.MOCHI_LOAD_TARGETS.items():
            assert (
                self._case_targets(suite_id, "VideoGenerationLoadTest") == expected
            ), suite_id

    def test_wan_i2v_targets_merge_per_device(self):
        for suite_id, expected in self.WAN_I2V_TARGETS.items():
            assert self._case_targets(suite_id, "VideoGenerationI2VTest") == expected, (
                suite_id
            )
