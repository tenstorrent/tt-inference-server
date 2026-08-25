# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Tests for the model identity <-> name token contract.

The contract is shared with tt-shield, which builds the CI artifact and job
names this repo parses, so these tests are the executable definition of it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from utils.model_naming import (
    MODEL_ID_SEP,
    ci_job_matches_device,
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
VECTORS_PATH = Path(__file__).with_name("model_naming_vectors.json")
VECTORS = json.loads(VECTORS_PATH.read_text())["escape"]

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


class TestGoldenVectors:
    """The escape must match `model_naming_vectors.json`, byte for byte.

    That file is the cross-repo contract. tt-shield carries the identical copy at
    `.github/scripts/model_naming_vectors.json` and asserts its own escape
    against it; this class is the tt-inference-server half. Pinning the format
    with data rather than sharing code is deliberate -- sharing code would mean
    one repo fetching and executing the other's module at CI time, which is what
    this replaced. A vector added on one side and not the other fails here, on
    the pull request that introduced the drift.

    Add a vector to both copies in the same change.
    """

    @pytest.mark.parametrize(
        "vector", VECTORS, ids=[v["identity"] for v in VECTORS]
    )
    def test_slugify_matches_vector(self, vector):
        assert slugify_model_id(vector["identity"]) == vector["token"], vector["why"]

    @pytest.mark.parametrize(
        "vector", VECTORS, ids=[v["identity"] for v in VECTORS]
    )
    def test_round_trips(self, vector):
        assert unslugify_model_id(vector["token"]) == vector["identity"], vector["why"]

    def test_no_vector_collides(self):
        """Two identities must never escape to the same token.

        A collision makes a name ambiguous -- two models writing one artifact
        name -- which is what the `__` escape exists to prevent. The pair
        differing only by case is the sharpest instance.
        """
        tokens = [v["token"] for v in VECTORS]
        assert len(set(tokens)) == len(tokens)

    @pytest.mark.parametrize(
        "vector", VECTORS, ids=[v["identity"] for v in VECTORS]
    )
    def test_every_token_is_artifact_name_safe(self, vector):
        assert is_artifact_name_safe(vector["token"])

    def test_the_copy_in_tt_shield_is_named_in_the_file(self):
        """A reader who finds one copy must be able to find the other."""
        comment = " ".join(json.loads(VECTORS_PATH.read_text())["comment"])
        assert ".github/scripts/model_naming_vectors.json" in comment
        assert "tests/model_naming_vectors.json" in comment


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
        assert model_name_variants("Qwen/Qwen3-32B") == (
            "Qwen__Qwen3-32B",  # canonical
            "Qwen3-32B",  # pre-migration model id, i.e. every older run
        )

    def test_unwritten_spellings_are_absent(self):
        """Two spellings a tolerant reader might expect, and why they are not here.

        `Qwen/Qwen3-32B` needs full config keys *and* a name built from the
        identity instead of the token -- only reachable if the two halves of the
        migration land apart. `Qwen_Qwen3-32B` came from tt-shield's old
        `Sanitize model name` step, which fed the `report_` artifact name only,
        never a job name, and was a no-op on the bare ids of the time.
        """
        variants = model_name_variants("Qwen/Qwen3-32B")
        assert "Qwen/Qwen3-32B" not in variants
        assert "Qwen_Qwen3-32B" not in variants

    def test_a_native_separator_is_not_mistaken_for_an_escape(self):
        """`microsoft/phi-1_5` is exactly why the single-`_` spelling is unsafe."""
        assert model_name_variants("microsoft/phi-1_5") == (
            "microsoft__phi-1_5",
            "phi-1_5",
        )

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

    def test_a_pre_migration_bundle_still_resolves(self):
        """Every release before the migration named its bundles bare."""
        assert split_workflow_logs_artifact_name(
            "workflow_logs_release_Qwen3-32B_p150_default", "release", "Qwen/Qwen3-32B"
        ) == ("p150", "default")

    def test_a_single_underscore_bundle_is_not_a_match(self):
        """No bundle was ever named that way, and `_` is the field separator of
        this very grammar -- accepting it would make the name ambiguous."""
        assert (
            split_workflow_logs_artifact_name(
                "workflow_logs_release_Qwen_Qwen3-32B_p150_default",
                "release",
                "Qwen/Qwen3-32B",
            )
            is None
        )

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

    def test_a_caller_prefix_with_extra_separators_is_ignored(self):
        """The marker is searched for, not reached by splitting: the caller prefix
        the jobs API prepends can carry more separators than expected."""
        name = "a / b / run-release-Qwen__Qwen3-32B-p150-P150"
        assert (
            device_from_ci_job_name(name, "release", "Qwen/Qwen3-32B", "p150") == "P150"
        )

    def test_an_unescaped_model_id_is_not_a_match(self):
        """A job name can only hold a raw `/` if the tt-inference-server and
        tt-shield halves of the migration landed apart. They land together, so
        this spelling is not accepted -- and matching it would require splitting
        logic that eats the org prefix along with the caller prefix."""
        name = "call-release / run-release-Qwen/Qwen3-32B-p150-P150"
        assert device_from_ci_job_name(name, "release", "Qwen/Qwen3-32B", "p150") is None

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


class TestCiJobMatchesDevice:
    """Matching with the device known and the runner label not."""

    def test_matches_canonical_job_name(self):
        name = ci_job_name("release", "Qwen/Qwen3-32B", "p150", "P150")
        assert ci_job_matches_device(name, "release", "Qwen/Qwen3-32B", "P150")

    def test_runner_label_containing_hyphens(self):
        # The real shape from tt-shield run 6035.
        name = (
            "run-tests / run-release-meta-llama__Llama-3.3-70B-Instruct-bh-qb-ge-p300x2"
        )
        assert ci_job_matches_device(
            name, "release", "meta-llama/Llama-3.3-70B-Instruct", "P300X2"
        )

    def test_device_comparison_is_case_insensitive(self):
        name = "run-release-Qwen__Qwen3-32B-p150-p150"
        assert ci_job_matches_device(name, "release", "Qwen/Qwen3-32B", "P150")
        assert ci_job_matches_device(name, "release", "Qwen/Qwen3-32B", "p150")

    def test_a_pre_migration_job_name_still_matches(self):
        name = "run-tests / run-release-Qwen3-32B-p150-p150"
        assert ci_job_matches_device(name, "release", "Qwen/Qwen3-32B", "P150")

    def test_unwritten_spellings_do_not_match(self):
        for token in ("Qwen/Qwen3-32B", "Qwen_Qwen3-32B"):
            name = f"run-tests / run-release-{token}-p150-p150"
            assert not ci_job_matches_device(
                name, "release", "Qwen/Qwen3-32B", "P150"
            ), token

    def test_a_missing_runner_label_is_not_a_match(self):
        assert not ci_job_matches_device(
            "run-release-Qwen__Qwen3-32B-p150", "release", "Qwen/Qwen3-32B", "p150"
        )


class TestCiJobPrefixSiblings:
    """``-`` separates fields and also occurs inside model names, so a sibling's
    job is only attributable once the other models in scope are known."""

    SIBLING_JOB = "run-release-Qwen__Qwen3-32B-FP8-p150-P150"
    OWN_JOB = "run-release-Qwen__Qwen3-32B-p150-P150"

    def test_sibling_job_is_claimed_when_scope_is_unknown(self):
        # Documents the residual ambiguity: with nothing to compare against,
        # the shorter model still explains the name.
        assert ci_job_matches_device(
            self.SIBLING_JOB, "release", "Qwen/Qwen3-32B", "P150"
        )

    def test_sibling_job_is_left_to_the_sibling(self):
        assert not ci_job_matches_device(
            self.SIBLING_JOB,
            "release",
            "Qwen/Qwen3-32B",
            "P150",
            ["Qwen/Qwen3-32B-FP8"],
        )

    def test_sibling_still_claims_its_own_job(self):
        assert ci_job_matches_device(
            self.SIBLING_JOB,
            "release",
            "Qwen/Qwen3-32B-FP8",
            "P150",
            ["Qwen/Qwen3-32B"],
        )

    def test_own_job_survives_a_longer_sibling_in_scope(self):
        assert ci_job_matches_device(
            self.OWN_JOB, "release", "Qwen/Qwen3-32B", "P150", ["Qwen/Qwen3-32B-FP8"]
        )

    def test_self_in_scope_is_ignored(self):
        assert ci_job_matches_device(
            self.OWN_JOB, "release", "Qwen/Qwen3-32B", "P150", ["Qwen/Qwen3-32B"]
        )

    def test_unrelated_models_in_scope_do_not_interfere(self):
        assert ci_job_matches_device(
            self.OWN_JOB,
            "release",
            "Qwen/Qwen3-32B",
            "P150",
            ["meta-llama/Llama-3.1-8B-Instruct", "openai/whisper-large-v3"],
        )

    def test_device_must_be_a_whole_trailing_token(self):
        name = "run-release-Qwen__Qwen3-32B-p150-p300x2"
        assert not ci_job_matches_device(name, "release", "Qwen/Qwen3-32B", "x2")

    def test_wrong_model_device_or_workflow_returns_false(self):
        name = "run-release-Qwen__Qwen3-32B-p150-P150"
        assert not ci_job_matches_device(name, "release", "org/other", "P150")
        assert not ci_job_matches_device(name, "release", "Qwen/Qwen3-32B", "N300")
        assert not ci_job_matches_device(name, "benchmarks", "Qwen/Qwen3-32B", "P150")
        assert not ci_job_matches_device("", "release", "Qwen/Qwen3-32B", "P150")
        assert not ci_job_matches_device(name, "release", "Qwen/Qwen3-32B", "")

    def test_trailing_punctuation_is_trimmed(self):
        name = "run-tests (run-release-Qwen__Qwen3-32B-p150-P150)"
        assert ci_job_matches_device(name, "release", "Qwen/Qwen3-32B", "P150")

    def test_agrees_with_device_from_ci_job_name_on_every_model(self):
        # The two directions must not disagree.
        for model_id in MODEL_IDS:
            name = ci_job_name("release", model_id, "some-runner", "P300X2")
            device = device_from_ci_job_name(name, "release", model_id, "some-runner")
            assert device == "P300X2"
            assert ci_job_matches_device(name, "release", model_id, device)


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
