# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Tests for the model identity <-> name token contract.

The contract is shared with tt-shield, which builds the CI artifact and job
names this repo parses, so these tests are the executable definition of it.
See ``docs/model_id_naming.md``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from utils.model_naming import (
    MODEL_ID_SEP,
    ci_job_name,
    device_from_ci_job_name,
    is_artifact_name_safe,
    model_name_variants,
    slugify_model_id,
    slugify_name_parts,
    split_workflow_logs_artifact_name,
    unslugify_model_id,
    workflow_logs_artifact_prefix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "utils" / "model_naming.py"
CI_CONFIG = REPO_ROOT / ".github" / "workflows" / "models-ci-config.json"

# Every shape a real model id takes: org-prefixed, bare, and carrying the
# single underscores that force `__` as the separator.
MODEL_IDS = [
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-0528",
    "microsoft/phi-1_5",
    "resnet-50",
    "yolox_nano",
]


class TestSlugifyRoundTrip:
    @pytest.mark.parametrize("model_id", MODEL_IDS)
    def test_round_trips_exactly(self, model_id):
        assert unslugify_model_id(slugify_model_id(model_id)) == model_id

    @pytest.mark.parametrize("model_id", MODEL_IDS)
    def test_slug_is_a_single_path_component(self, model_id):
        slug = slugify_model_id(model_id)
        assert "/" not in slug
        assert len(Path(slug).parts) == 1

    @pytest.mark.parametrize("model_id", MODEL_IDS)
    def test_slug_is_a_legal_artifact_name(self, model_id):
        assert is_artifact_name_safe(slugify_model_id(model_id))

    def test_org_prefix_is_escaped_not_stripped(self):
        # Two orgs publishing the same basename must stay distinguishable.
        assert slugify_model_id("Qwen/Qwen3-32B") == f"Qwen{MODEL_ID_SEP}Qwen3-32B"
        assert slugify_model_id("a/model") != slugify_model_id("b/model")

    def test_single_underscore_survives(self):
        # The whole reason the separator is "__": a single "_" is ambiguous.
        assert unslugify_model_id(slugify_model_id("microsoft/phi-1_5")) == (
            "microsoft/phi-1_5"
        )

    def test_bare_ids_are_untouched_in_both_directions(self):
        assert slugify_model_id("resnet-50") == "resnet-50"
        assert unslugify_model_id("resnet-50") == "resnet-50"

    def test_empty_is_empty(self):
        assert slugify_model_id("") == ""
        assert unslugify_model_id("") == ""

    def test_windows_separator_and_space(self):
        assert slugify_model_id("Qwen\\Qwen3-32B") == "Qwen__Qwen3-32B"
        assert slugify_model_id("some model") == "some_model"

    def test_all_ci_config_ids_round_trip(self):
        """The contract must hold for every model the CI actually runs."""
        models = json.loads(CI_CONFIG.read_text())["models"]
        assert models, "models-ci-config.json has no models"
        for model_id in models:
            slug = slugify_model_id(model_id)
            assert is_artifact_name_safe(slug), model_id
            assert unslugify_model_id(slug) == model_id, model_id


class TestSlugifyNameParts:
    def test_joins_and_escapes(self):
        assert slugify_name_parts("Qwen/Qwen3-32B", "p150") == "Qwen__Qwen3-32B_p150"

    def test_drops_empty_parts_without_leaving_a_separator(self):
        assert slugify_name_parts("Qwen/Qwen3-32B", None) == "Qwen__Qwen3-32B"
        assert slugify_name_parts("", "p150") == "p150"
        assert slugify_name_parts(None, "") == ""


class TestModelNameVariants:
    def test_canonical_form_comes_first(self):
        assert model_name_variants("Qwen/Qwen3-32B")[0] == "Qwen__Qwen3-32B"

    def test_covers_the_forms_producers_have_actually_used(self):
        variants = model_name_variants("Qwen/Qwen3-32B")
        assert set(variants) == {
            "Qwen__Qwen3-32B",  # canonical
            "Qwen/Qwen3-32B",  # unescaped (job names)
            "Qwen_Qwen3-32B",  # tt-shield's old single-underscore step
            "Qwen3-32B",  # pre-HF-prefix model id
        }

    def test_deduplicates_for_bare_ids(self):
        assert model_name_variants("resnet-50") == ("resnet-50",)

    def test_empty(self):
        assert model_name_variants("") == ()


class TestWorkflowLogsArtifactName:
    def test_prefix_ends_with_a_boundary(self):
        # Without the trailing "_", "foo" would also match "foo-turbo".
        prefix = workflow_logs_artifact_prefix("release", "Qwen/Qwen3-32B")
        assert prefix == "workflow_logs_release_Qwen__Qwen3-32B_"

    def test_splits_canonical_name(self):
        assert split_workflow_logs_artifact_name(
            "workflow_logs_release_Qwen__Qwen3-32B_p150_default",
            "release",
            "Qwen/Qwen3-32B",
        ) == ("p150", "default")

    def test_runner_label_may_contain_underscores(self):
        assert split_workflow_logs_artifact_name(
            "workflow_logs_release_Qwen__Qwen3-32B_tt_beta_p150_vllm",
            "release",
            "Qwen/Qwen3-32B",
        ) == ("tt_beta_p150", "vllm")

    def test_prefix_sharing_models_are_unambiguous(self):
        name = "workflow_logs_release_org__foo-turbo_p150_default"
        assert split_workflow_logs_artifact_name(name, "release", "org/foo") is None
        assert split_workflow_logs_artifact_name(name, "release", "org/foo-turbo") == (
            "p150",
            "default",
        )

    def test_rejects_other_models_and_other_workflows(self):
        name = "workflow_logs_release_Qwen__Qwen3-32B_p150_default"
        assert split_workflow_logs_artifact_name(name, "release", "org/other") is None
        assert (
            split_workflow_logs_artifact_name(name, "benchmarks", "Qwen/Qwen3-32B")
            is None
        )

    def test_tolerates_producers_that_predate_the_contract(self):
        """A bundle still resolves if it was named the old way."""
        for name in (
            "workflow_logs_release_Qwen_Qwen3-32B_p150_default",  # single "_"
            "workflow_logs_release_Qwen3-32B_p150_default",  # no org prefix
        ):
            assert split_workflow_logs_artifact_name(
                name, "release", "Qwen/Qwen3-32B"
            ) == ("p150", "default")

    def test_missing_suffix_is_not_a_match(self):
        assert (
            split_workflow_logs_artifact_name(
                "workflow_logs_release_Qwen__Qwen3-32B_p150",
                "release",
                "Qwen/Qwen3-32B",
            )
            is None
        )
        assert (
            split_workflow_logs_artifact_name("", "release", "Qwen/Qwen3-32B") is None
        )


class TestCiJobName:
    def test_build_and_parse_round_trip(self):
        name = ci_job_name("release", "Qwen/Qwen3-32B", "p150", "P150")
        assert name == "run-release-Qwen__Qwen3-32B-p150-P150"
        assert device_from_ci_job_name(name, "release", "Qwen/Qwen3-32B", "p150") == (
            "P150"
        )

    def test_reusable_workflow_prefix_is_ignored(self):
        name = "call-release / run-release-Qwen__Qwen3-32B-p150-P150"
        assert (
            device_from_ci_job_name(name, "release", "Qwen/Qwen3-32B", "p150") == "P150"
        )

    def test_unescaped_model_id_still_resolves(self):
        """The bug this contract exists to prevent.

        A producer that interpolates the raw id puts a "/" in the job name.
        Stripping the caller prefix by splitting on "/" would eat the org
        prefix along with it and lose the device.
        """
        name = "call-release / run-release-Qwen/Qwen3-32B-p150-P150"
        assert (
            device_from_ci_job_name(name, "release", "Qwen/Qwen3-32B", "p150") == "P150"
        )

    def test_trailing_punctuation_is_trimmed(self):
        name = "run-tests (run-release-Qwen__Qwen3-32B-p150-P150)"
        assert (
            device_from_ci_job_name(name, "release", "Qwen/Qwen3-32B", "p150") == "P150"
        )

    def test_wrong_model_runner_or_workflow_returns_none(self):
        name = "run-release-Qwen__Qwen3-32B-p150-P150"
        assert device_from_ci_job_name(name, "release", "org/other", "p150") is None
        assert (
            device_from_ci_job_name(name, "release", "Qwen/Qwen3-32B", "n300") is None
        )
        assert (
            device_from_ci_job_name(name, "benchmarks", "Qwen/Qwen3-32B", "p150")
            is None
        )
        assert device_from_ci_job_name("", "release", "Qwen/Qwen3-32B", "p150") is None

    def test_missing_device_returns_none(self):
        assert (
            device_from_ci_job_name(
                "run-release-Qwen__Qwen3-32B-p150-", "release", "Qwen/Qwen3-32B", "p150"
            )
            is None
        )


class TestStandaloneUsability:
    """tt-shield consumes this from a plain checkout, from shell, with no
    ``pip install`` and no ``PYTHONPATH``. Keep both of those working."""

    def test_module_imports_nothing_from_this_repo(self):
        source = MODULE_PATH.read_text()
        repo_packages = (
            "workflows",
            "workflow_module",
            "report_module",
            "llm_module",
            "test_module",
            "utils.",
        )
        for line in source.splitlines():
            if line.startswith(("import ", "from ")):
                assert not any(pkg in line for pkg in repo_packages), line

    def test_runs_as_a_script_by_path(self, tmp_path):
        # cwd deliberately outside the repo: no implicit package resolution.
        out = subprocess.run(
            [sys.executable, str(MODULE_PATH), "slugify", "Qwen/Qwen3-32B"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "Qwen__Qwen3-32B"

    @pytest.mark.parametrize(
        "args,expected",
        [
            (["slugify", "Qwen/Qwen3-32B"], "Qwen__Qwen3-32B"),
            (["unslugify", "Qwen__Qwen3-32B"], "Qwen/Qwen3-32B"),
            (
                ["artifact-prefix", "release", "Qwen/Qwen3-32B"],
                "workflow_logs_release_Qwen__Qwen3-32B_",
            ),
            (
                ["job-name", "release", "Qwen/Qwen3-32B", "p150", "P150"],
                "run-release-Qwen__Qwen3-32B-p150-P150",
            ),
        ],
    )
    def test_cli_commands(self, args, expected, capsys):
        from utils.model_naming import main

        assert main(args) == 0
        assert capsys.readouterr().out.strip() == expected

    def test_cli_rejects_bad_usage(self, capsys):
        from utils.model_naming import main

        assert main(["nope", "x"]) == 2
        assert main(["slugify"]) == 2
        assert main([]) == 0  # help
