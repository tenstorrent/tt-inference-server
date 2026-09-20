# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Guard that every enrolled spec-test suite is actually selectable at runtime.

``test_module/dispatch.py`` resolves spec-test suites with
``TestFilter().filter_by_model(ctx.model_spec.hf_model_repo)``, and
``filter_by_model`` is an exact-membership test against ``suite["weights"]``.
``run_spec_tests`` treats an empty selection as a clean ``rc=0`` no-op, so a
``weights`` value that does not equal the model's ``hf_model_repo`` makes the
whole enrollment silently pass without running anything.

That is a fail-open, so it needs a test at the selection layer rather than at
the matrix-expansion layer: expansion only crosses ``models x devices`` from
JSON and cannot see the mismatch.
"""

from __future__ import annotations

import pytest

from test_module.test_categorization_system.suite_loader import (
    load_server_tests_config,
)
from test_module.test_categorization_system.test_filter import (
    TestFilter as SuiteTestFilter,  # aliased so pytest does not collect it
)

# Rows enrolled in model_configs that no test_matrices entry references, so they
# expand to zero suites regardless of naming. Fixing these means giving them
# real coverage, which is a per-model decision -- not a naming change.
UNREFERENCED_MODEL_CONFIGS = {"mistral_small_24b"}


def _enrolled_pairs():
    """Yield (config_key, weights_value, device) for every enrolled combination."""
    model_configs = load_server_tests_config()["model_configs"]
    for key, config in sorted(model_configs.items()):
        if key in UNREFERENCED_MODEL_CONFIGS:
            continue
        for weights_value in config.get("weights", []):
            for device in config.get("compatible_devices", []):
                yield key, weights_value, device


@pytest.mark.parametrize(
    "config_key,weights_value,device",
    list(_enrolled_pairs()),
    ids=lambda v: str(v),
)
def test_enrolled_model_device_pair_selects_at_least_one_suite(
    config_key, weights_value, device
):
    """Every enrolled pair must select >= 1 suite.

    This catches rows that expand to nothing -- typically enrolled in
    ``model_configs`` but referenced by no ``test_matrices`` entry, or given a
    ``compatible_devices`` entry no matrix covers.

    It deliberately does NOT catch a ``weights``/``hf_model_repo`` mismatch:
    expansion derives each suite's ``weights`` from this same string, so the two
    always agree with each other no matter which form is used. That divergence
    is what ``test_model_configs_weights_are_full_hf_repo_ids`` is for.
    """
    suites = (
        SuiteTestFilter()
        .filter_by_model(weights_value)
        .filter_by_device(device)
        .get_tests()
    )

    assert suites, (
        f"model_configs[{config_key!r}] enrolls weights={weights_value!r} on "
        f"device={device!r} but TestFilter selects 0 suites, so "
        f"run_spec_tests would skip it and return rc=0. If weights is not the "
        f"full HF repo id (org/name, matching model_spec.hf_model_repo), that "
        f"is the cause."
    )


def test_model_configs_weights_are_full_hf_repo_ids():
    """weights must be full HF repo ids, since dispatch passes hf_model_repo.

    A handful of CNN models legitimately have no org prefix in the catalog, so
    the check is that weights matches some known hf_model_repo -- not that it
    merely contains a slash.
    """
    from workflows.model_spec import MODEL_SPECS

    known_repos = {
        spec.hf_model_repo
        for spec in MODEL_SPECS.values()
        if getattr(spec, "hf_model_repo", None)
    }
    # Models served outside the catalog still follow the org/name convention.
    tails = {repo.rsplit("/", 1)[-1] for repo in known_repos}

    offenders = {}
    for key, config in load_server_tests_config()["model_configs"].items():
        for weights_value in config.get("weights", []):
            if weights_value in known_repos or "/" in weights_value:
                continue
            if weights_value in tails:
                offenders[key] = weights_value

    assert not offenders, (
        "model_configs weights must be full HF repo ids to match "
        f"model_spec.hf_model_repo; bare names found: {offenders}"
    )


def test_model_categories_agree_with_weights():
    """model_categories must use the same strings as weights.

    ``_get_category_marker`` is looked up with each value from
    ``suite["weights"]``, so a category entry in a different form silently
    yields no category marker for that model's suites.
    """
    config = load_server_tests_config()
    categorized = {
        model for models in config["model_categories"].values() for model in models
    }
    enrolled = {
        weights_value
        for model_config in config["model_configs"].values()
        for weights_value in model_config.get("weights", [])
    }

    # Every enrolled model should carry a category marker.
    missing = sorted(enrolled - categorized)
    assert not missing, (
        "enrolled models absent from model_categories, so their suites get no "
        f"category marker: {missing}"
    )
