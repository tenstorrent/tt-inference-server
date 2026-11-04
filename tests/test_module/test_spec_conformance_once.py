# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Each served (model, impl, device) gets at most one VLLMParamConformanceTest.

One weights+device pair can be served by several implementations, and test
matrices key on weights and device. A row covered by both a generic matrix and
its own matrix runs the test twice, with different settings. For example, a
quetzal Qwen3-32B row got the generic thinking-on run next to its own
enable_thinking=false run. Selection goes through the same TestFilter chain
that dispatch.py uses, filter_by_impl included.
"""

from __future__ import annotations

from collections import defaultdict

import pytest
import yaml

from test_module.test_categorization_system.suite_loader import load_suite_files
from test_module.test_categorization_system.test_filter import (
    TestFilter as SuiteTestFilter,  # aliased so pytest does not collect it
)

TEMPLATE = "VLLMParamConformanceTest"
CATALOG = "workflows/model_specs/dev/llm.yaml"


def _served_rows():
    """(weights, device, impl_id) for every row in the dev LLM catalog."""
    with open(CATALOG) as handle:
        templates = yaml.safe_load(handle)["templates"]
    rows = set()
    for template in templates:
        impl = template.get("impl")
        for spec in template.get("device_model_specs") or []:
            for weights in template.get("weights") or []:
                rows.add((weights, str(spec["device"]).lower(), impl))
    return sorted(rows, key=str)


def _conformance_matrices(suites, weights, device, impl):
    selected = (
        SuiteTestFilter(suites)
        .filter_by_model(weights)
        .filter_by_device(device)
        .filter_by_impl(impl)
        .get_tests()
    )
    return [
        suite["id"]
        for suite in selected
        for case in suite["test_cases"]
        if case.get("name") == TEMPLATE and case.get("enabled", True)
    ]


def _duplicates(only_impl=None, skip_impl=None):
    suites = load_suite_files()
    duplicates = {}
    for weights, device, impl in _served_rows():
        if only_impl is not None and impl != only_impl:
            continue
        if skip_impl is not None and impl == skip_impl:
            continue
        found = _conformance_matrices(suites, weights, device, impl)
        if len(found) > 1:
            duplicates[(weights, device, impl)] = found
    return duplicates


def test_every_quetzal_row_with_an_enrollment_gets_exactly_one():
    suites = load_suite_files()
    counts = defaultdict(list)
    for weights, device, impl in _served_rows():
        if impl != "quetzal":
            continue
        counts[(weights, device)] = _conformance_matrices(suites, weights, device, impl)
    enrolled = {k: v for k, v in counts.items() if v}
    assert enrolled, "no quetzal row selects a param-conformance matrix at all"
    duplicates = {k: v for k, v in enrolled.items() if len(v) != 1}
    assert not duplicates, (
        f"{TEMPLATE} selected more than once for a quetzal row (scope the "
        f"generic matrix with impls/exclude_impls): {duplicates}"
    )


def test_every_other_impl_gets_at_most_one():
    # Quetzal-generated matrices carry impls=["quetzal"] (quetzal #254), so
    # they no longer select for other impls serving the same weights.
    duplicates = _duplicates(skip_impl="quetzal")
    assert not duplicates, f"{TEMPLATE} selected more than once: {duplicates}"


@pytest.mark.parametrize(
    "suite,impl,selected",
    [
        ({}, "quetzal", True),
        ({"exclude_impls": ["quetzal"]}, "quetzal", False),
        ({"exclude_impls": ["quetzal"]}, "tt_transformers", True),
        ({"impls": ["quetzal"]}, "quetzal", True),
        ({"impls": ["quetzal"]}, "tt_transformers", False),
        ({"exclude_impls": ["quetzal"]}, None, True),
    ],
)
def test_filter_by_impl(suite, impl, selected):
    suites = [{"id": "s", "weights": ["org/m"], "device": "p300x2", **suite}]
    assert bool(SuiteTestFilter(suites).filter_by_impl(impl).filtered_suites) is (
        selected
    )


def test_matrix_impl_scope_reaches_the_expanded_suite():
    from test_module.test_categorization_system.suite_loader import (
        expand_test_matrices,
    )

    (suite,) = expand_test_matrices(
        [
            {
                "models": ["m"],
                "devices": ["p300x2"],
                "exclude_impls": ["quetzal"],
                "test_cases": [{"template": TEMPLATE}],
            }
        ],
        {"m": {"weights": ["org/m"]}},
    )
    assert suite["exclude_impls"] == ["quetzal"]
    assert "impls" not in suite


def test_dispatch_selects_by_the_spec_impl():
    from types import SimpleNamespace

    from test_module.dispatch import _resolve_spec_test_suites

    def names(impl_id):
        ctx = SimpleNamespace(
            model_spec=SimpleNamespace(
                hf_model_repo="meta-llama/Llama-3.1-8B-Instruct",
                inference_engine="vLLM",
                impl=SimpleNamespace(impl_id=impl_id),
            ),
            device=SimpleNamespace(name="P300X2"),
        )
        return [s["id"] for s in _resolve_spec_test_suites(ctx)]

    assert "llama-3.1-8b-p300x2" not in names("quetzal")
    assert "meta-llama--Llama-3.1-8B-Instruct-p300x2" in names("quetzal")
    assert "llama-3.1-8b-p300x2" in names("tt_transformers")
