# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import contextlib
import io
import json
import shutil
import textwrap

import pytest
import yaml

from scripts.release.promote_dev_spec_to_prod import (
    DEFAULT_CI_CONFIG,
    DEFAULT_DEV_DIR,
    DEFAULT_PROD_DIR,
    main,
    promote,
)
from scripts.release.generate_model_support_docs import (
    coalesce_catalog_groups_for_docs,
    generate_doc_pages,
)
from workflows.model_spec import (
    MODEL_SPEC_CATALOG_FILES,
    get_model_spec_map,
    load_templates_from_yaml,
)
from workflows.workflow_types import DeviceTypes, InferenceEngine


PINS = {
    "tt_metal_commit": "new-metal",
    "version": "9.9.9",
    "vllm_commit": "new-vllm",
}


def _write_catalogs(tmp_path, *, dev_llm, prod_llm):
    dev_dir = tmp_path / "dev"
    prod_dir = tmp_path / "prod"
    dev_dir.mkdir()
    prod_dir.mkdir()
    for filename in MODEL_SPEC_CATALOG_FILES:
        (dev_dir / filename).write_text("templates: []\n")
        (prod_dir / filename).write_text("templates: []\n")
    (dev_dir / "llm.yaml").write_text(textwrap.dedent(dev_llm))
    (prod_dir / "llm.yaml").write_text(textwrap.dedent(prod_llm))
    return dev_dir, prod_dir


def _write_ci(tmp_path, models):
    path = tmp_path / "ci.json"
    path.write_text(json.dumps({"models": models}))
    return path


def _vllm_release_entry(*devices):
    return {
        "inference_engine": "vLLM",
        "ci": {"release": {"devices": list(devices)}},
    }


def _template_identity(template):
    return (
        template["weights"][0],
        DeviceTypes.from_string(
            template["device_model_specs"][0]["device"]
        ).to_string(),
        InferenceEngine.from_string(template["inference_engine"]).value,
        template["impl"],
    )


def _load_leaf_templates(prod_dir):
    templates = []
    for filename in MODEL_SPEC_CATALOG_FILES:
        data = yaml.safe_load((prod_dir / filename).read_text())
        for template in data["templates"]:
            assert len(template["weights"]) == 1
            assert len(template["device_model_specs"]) == 1
            templates.append(template)
    return {_template_identity(template): template for template in templates}


def _raw_templates(prod_dir):
    return [
        template
        for filename in MODEL_SPEC_CATALOG_FILES
        for template in yaml.safe_load((prod_dir / filename).read_text())["templates"]
    ]


def _file_tree(root):
    return {
        str(path.relative_to(root)): path.read_text()
        for path in root.rglob("*")
        if path.is_file()
    }


def _serialized_catalog(prod_dir):
    templates = [
        template
        for filename in MODEL_SPEC_CATALOG_FILES
        for template in load_templates_from_yaml(prod_dir / filename)
    ]
    return {
        model_id: spec.get_serialized_dict()
        for model_id, spec in get_model_spec_map(templates).items()
    }


def _grouped_fixture(tmp_path):
    dev_dir, prod_dir = _write_catalogs(
        tmp_path,
        dev_llm="""
            templates:
            - weights:
                - org/model-A
                - org/model-B
              impl: tt_transformers
              inference_engine: VLLM
              device_model_specs:
                - device: N150
                  max_concurrency: 16
                  max_context: 65536
                  default_impl: true
                - device: N300
                  max_concurrency: 8
                  max_context: 32768
                  default_impl: true
              metadata:
                org/model-A:
                  note: new-a
                org/model-B:
                  note: new-b
        """,
        prod_llm="""
            templates:
            # This unrelated leaf must retain its released semantics.
            - weights:
                - org/unrelated
              impl: tt_transformers
              inference_engine: VLLM
              version: "0.5.0"
              tt_metal_commit: "unrelated-metal"
              vllm_commit: "unrelated-vllm"
              device_model_specs:
                - device: N150
                  max_concurrency: 1
                  max_context: 1024

            - weights:
                - org/model-A
                - org/model-B
              impl: tt_transformers
              inference_engine: VLLM
              version: "1.0.0"
              tt_metal_commit: "old-metal"
              vllm_commit: "old-vllm"
              device_model_specs:
                - device: N150
                  max_concurrency: 2
                  max_context: 4096
                  default_impl: true
                - device: N300
                  max_concurrency: 4
                  max_context: 8192
                  default_impl: true
              metadata:
                org/model-A:
                  note: old-a
                org/model-B:
                  note: old-b
        """,
    )
    ci_path = _write_ci(
        tmp_path,
        {"model-A": _vllm_release_entry("N150")},
    )
    return ci_path, dev_dir, prod_dir


def test_grouped_owner_is_leafized_and_only_target_semantics_change(tmp_path):
    ci_path, dev_dir, prod_dir = _grouped_fixture(tmp_path)

    report = promote(ci_path, dev_dir, prod_dir, **PINS)

    target = ("org/model-A", "N150", "vLLM", "tt_transformers")
    assert report["actions"] == {target: "updated"}
    assert report["leaf_count_before"] == 5
    assert report["leaf_count_after"] == 5
    assert report["changed_files"] == ["llm.yaml"]

    text = (prod_dir / "llm.yaml").read_text()
    assert "# This unrelated leaf must retain its released semantics." in text
    leaves = _load_leaf_templates(prod_dir)
    assert len(leaves) == 5

    promoted = leaves[target]
    assert promoted["version"] == "9.9.9"
    assert promoted["tt_metal_commit"] == "new-metal"
    assert promoted["vllm_commit"] == "new-vllm"
    assert promoted["device_model_specs"][0]["max_context"] == 65536
    assert promoted["metadata"] == {"org/model-A": {"note": "new-a"}}
    unrelated = leaves[("org/unrelated", "N150", "vLLM", "tt_transformers")]
    assert unrelated["version"] == "0.5.0"
    assert unrelated["tt_metal_commit"] == "unrelated-metal"
    assert unrelated["vllm_commit"] == "unrelated-vllm"

    for identity in (
        ("org/model-A", "N300", "vLLM", "tt_transformers"),
        ("org/model-B", "N150", "vLLM", "tt_transformers"),
        ("org/model-B", "N300", "vLLM", "tt_transformers"),
    ):
        sibling = leaves[identity]
        assert sibling["version"] == "1.0.0"
        assert sibling["tt_metal_commit"] == "old-metal"
        assert sibling["vllm_commit"] == "old-vllm"
        assert sibling["metadata"] == {
            identity[0]: {"note": f"old-{identity[0][-1].lower()}"}
        }


def test_new_target_is_appended_as_exact_leaf(tmp_path):
    dev_dir, prod_dir = _write_catalogs(
        tmp_path,
        dev_llm="""
            templates:
            - weights:
                - org/new-model
              impl: tt_transformers
              inference_engine: VLLM
              device_model_specs:
                - device: N150
                  max_concurrency: 1
                  max_context: 2048
                  default_impl: true
        """,
        prod_llm="templates: []\n",
    )
    ci_path = _write_ci(
        tmp_path,
        {"new-model": _vllm_release_entry("N150")},
    )

    report = promote(ci_path, dev_dir, prod_dir, **PINS)

    identity = ("org/new-model", "N150", "vLLM", "tt_transformers")
    assert report["actions"] == {identity: "added"}
    assert report["leaf_count_before"] == 0
    assert report["leaf_count_after"] == 1
    assert identity in _load_leaf_templates(prod_dir)


def test_resolution_failure_does_not_write_any_prod_file(tmp_path):
    ci_path, dev_dir, prod_dir = _grouped_fixture(tmp_path)
    ci_path = _write_ci(
        tmp_path,
        {
            "model-A": _vllm_release_entry("N150"),
            "missing-model": _vllm_release_entry("N150"),
        },
    )
    before = {path.name: path.read_bytes() for path in prod_dir.glob("*.yaml")}

    with pytest.raises(ValueError, match="No model spec matches"):
        promote(ci_path, dev_dir, prod_dir, **PINS)

    assert {path.name: path.read_bytes() for path in prod_dir.glob("*.yaml")} == before


def test_duplicate_prod_identity_fails_before_write(tmp_path):
    dev_dir, prod_dir = _write_catalogs(
        tmp_path,
        dev_llm="""
            templates:
            - weights: [org/model-A]
              impl: tt_transformers
              inference_engine: VLLM
              device_model_specs:
                - {device: N150, max_concurrency: 1, max_context: 1024, default_impl: true}
        """,
        prod_llm="""
            templates:
            - weights: [org/model-A]
              impl: tt_transformers
              inference_engine: VLLM
              version: "1.0.0"
              tt_metal_commit: old
              vllm_commit: old
              device_model_specs:
                - {device: N150, max_concurrency: 1, max_context: 1024}
            - weights: [org/model-A]
              impl: tt_transformers
              inference_engine: VLLM
              version: "2.0.0"
              tt_metal_commit: newer
              vllm_commit: newer
              device_model_specs:
                - {device: N150, max_concurrency: 1, max_context: 1024}
        """,
    )
    ci_path = _write_ci(tmp_path, {"model-A": _vllm_release_entry("N150")})
    before = (prod_dir / "llm.yaml").read_bytes()

    with pytest.raises(ValueError, match="Duplicate model spec leaf identity"):
        promote(ci_path, dev_dir, prod_dir, **PINS)

    assert (prod_dir / "llm.yaml").read_bytes() == before


@pytest.mark.parametrize("vllm_commit", [None, "", "   "])
def test_vllm_target_requires_non_empty_vllm_commit(tmp_path, vllm_commit):
    ci_path, dev_dir, prod_dir = _grouped_fixture(tmp_path)
    before = (prod_dir / "llm.yaml").read_bytes()

    with pytest.raises(ValueError, match="vllm_commit"):
        promote(
            ci_path,
            dev_dir,
            prod_dir,
            version=PINS["version"],
            tt_metal_commit=PINS["tt_metal_commit"],
            vllm_commit=vllm_commit,
        )

    assert (prod_dir / "llm.yaml").read_bytes() == before


def test_dry_run_json_reports_changes_without_writing(tmp_path, capsys):
    ci_path, dev_dir, prod_dir = _grouped_fixture(tmp_path)
    before = (prod_dir / "llm.yaml").read_bytes()

    result = main(
        [
            "--ci-config",
            str(ci_path),
            "--dev-dir",
            str(dev_dir),
            "--prod-dir",
            str(prod_dir),
            "--version",
            PINS["version"],
            "--tt-metal-commit",
            PINS["tt_metal_commit"],
            "--vllm-commit",
            PINS["vllm_commit"],
            "--dry-run",
            "--json",
        ]
    )

    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["dry_run"] is True
    assert report["actions"][0]["action"] == "updated"
    assert report["resolved"][0]["source_path"].endswith("llm.yaml")
    assert report["changed_files"] == ["llm.yaml"]
    assert (prod_dir / "llm.yaml").read_bytes() == before


def test_json_error_is_machine_readable_and_does_not_write(tmp_path, capsys):
    _, dev_dir, prod_dir = _grouped_fixture(tmp_path)
    ci_path = _write_ci(
        tmp_path,
        {"missing-model": _vllm_release_entry("N150")},
    )
    before = (prod_dir / "llm.yaml").read_bytes()

    result = main(
        [
            "--ci-config",
            str(ci_path),
            "--dev-dir",
            str(dev_dir),
            "--prod-dir",
            str(prod_dir),
            "--version",
            PINS["version"],
            "--tt-metal-commit",
            PINS["tt_metal_commit"],
            "--vllm-commit",
            PINS["vllm_commit"],
            "--dry-run",
            "--json",
        ]
    )

    assert result == 2
    error = json.loads(capsys.readouterr().out)
    assert error["ok"] is False
    assert error["error_type"] == "ValueError"
    assert "No model spec matches" in error["error"]
    assert (prod_dir / "llm.yaml").read_bytes() == before


@pytest.mark.parametrize(
    "invalid_prod",
    [
        "templates:\n- weights: [\n",
        """
        templates:
        - weights: [org/model-A]
          impl: tt_transformers
          impl: tt_transformers
          inference_engine: VLLM
          version: "1.0.0"
          tt_metal_commit: old
          vllm_commit: old
          device_model_specs:
            - {device: N150, max_concurrency: 1, max_context: 1024}
        """,
    ],
)
def test_yaml_errors_are_machine_readable_and_do_not_write(
    tmp_path, capsys, invalid_prod
):
    ci_path, dev_dir, prod_dir = _grouped_fixture(tmp_path)
    (prod_dir / "llm.yaml").write_text(textwrap.dedent(invalid_prod))
    before = (prod_dir / "llm.yaml").read_bytes()

    result = main(
        [
            "--ci-config",
            str(ci_path),
            "--dev-dir",
            str(dev_dir),
            "--prod-dir",
            str(prod_dir),
            "--version",
            PINS["version"],
            "--tt-metal-commit",
            PINS["tt_metal_commit"],
            "--vllm-commit",
            PINS["vllm_commit"],
            "--dry-run",
            "--json",
        ]
    )

    assert result == 2
    error = json.loads(capsys.readouterr().out)
    assert error["ok"] is False
    assert error["error"]
    assert (prod_dir / "llm.yaml").read_bytes() == before


def test_second_promotion_is_byte_idempotent(tmp_path):
    ci_path, dev_dir, prod_dir = _grouped_fixture(tmp_path)
    promote(ci_path, dev_dir, prod_dir, **PINS)
    after_first = {path.name: path.read_bytes() for path in prod_dir.glob("*.yaml")}

    report = promote(ci_path, dev_dir, prod_dir, **PINS)

    assert report["changed_files"] == []
    assert set(report["unchanged_identities"]) == set(report["requested_identities"])
    assert {
        path.name: path.read_bytes() for path in prod_dir.glob("*.yaml")
    } == after_first


def test_full_prod_flattening_has_no_expanded_semantic_diff(tmp_path):
    prod_copy = tmp_path / "prod"
    shutil.copytree(DEFAULT_PROD_DIR, prod_copy)
    before = _serialized_catalog(prod_copy)
    ci_path = _write_ci(tmp_path, {})

    report = promote(
        ci_path,
        DEFAULT_DEV_DIR,
        prod_copy,
        version=PINS["version"],
        tt_metal_commit=PINS["tt_metal_commit"],
    )

    assert report["leaf_count_before"] == 225
    assert report["leaf_count_after"] == 225
    assert _serialized_catalog(prod_copy) == before
    assert len(_load_leaf_templates(prod_copy)) == 225
    for template in _raw_templates(prod_copy):
        assert "perf_reference" in template["device_model_specs"][0]
        assert template["model_display_name"]
        assert template["catalog_group"]


def test_full_prod_flattening_preserves_generated_documentation(tmp_path):
    prod_copy = tmp_path / "prod"
    shutil.copytree(DEFAULT_PROD_DIR, prod_copy)
    ci_path = _write_ci(tmp_path, {})
    before_docs = tmp_path / "docs-before"
    after_docs = tmp_path / "docs-after"

    with contextlib.redirect_stdout(io.StringIO()):
        generate_doc_pages(
            [
                template
                for filename in MODEL_SPEC_CATALOG_FILES
                for template in load_templates_from_yaml(prod_copy / filename)
            ],
            str(before_docs),
        )

    promote(
        ci_path,
        DEFAULT_DEV_DIR,
        prod_copy,
        version=PINS["version"],
        tt_metal_commit=PINS["tt_metal_commit"],
    )

    with contextlib.redirect_stdout(io.StringIO()):
        generate_doc_pages(
            [
                template
                for filename in MODEL_SPEC_CATALOG_FILES
                for template in load_templates_from_yaml(prod_copy / filename)
            ],
            str(after_docs),
        )

    assert _file_tree(after_docs) == _file_tree(before_docs)


def test_docs_do_not_coalesce_heterogeneous_release_pins(tmp_path):
    _, prod_dir = _write_catalogs(
        tmp_path,
        dev_llm="templates: []\n",
        prod_llm="""
            templates:
            - weights: [org/model-A]
              impl: tt_transformers
              inference_engine: VLLM
              version: "1.0.0"
              tt_metal_commit: old
              vllm_commit: old
              catalog_group: shared
              model_display_name: model
              device_model_specs:
                - {device: N150, max_concurrency: 1, max_context: 1024}
            - weights: [org/model-B]
              impl: tt_transformers
              inference_engine: VLLM
              version: "2.0.0"
              tt_metal_commit: new
              vllm_commit: new
              catalog_group: shared
              model_display_name: model
              device_model_specs:
                - {device: N150, max_concurrency: 1, max_context: 1024}
        """,
    )
    templates = load_templates_from_yaml(prod_dir / "llm.yaml")

    assert coalesce_catalog_groups_for_docs(templates) == templates


def test_real_catalog_promotion_is_exact_leaf_granular_and_idempotent(tmp_path):
    prod_copy = tmp_path / "prod"
    shutil.copytree(DEFAULT_PROD_DIR, prod_copy)

    report = promote(
        DEFAULT_CI_CONFIG,
        DEFAULT_DEV_DIR,
        prod_copy,
        **PINS,
    )

    expected = {
        (
            "google/diffusiongemma-26B-A4B-it",
            "P300X2",
            "vLLM",
            "diffusion_gemma",
        ),
        ("Qwen/Qwen3-32B", "GALAXY", "vLLM", "qwen3_32b_galaxy"),
    }
    assert set(report["requested_identities"]) == expected
    assert set(report["added_identities"]) == {
        (
            "google/diffusiongemma-26B-A4B-it",
            "P300X2",
            "vLLM",
            "diffusion_gemma",
        )
    }
    assert set(report["updated_identities"]) == {
        ("Qwen/Qwen3-32B", "GALAXY", "vLLM", "qwen3_32b_galaxy")
    }
    assert report["leaf_count_before"] == 225
    assert report["leaf_count_after"] == 226
    assert len(_load_leaf_templates(prod_copy)) == 226
    assert all(
        "perf_reference" in template["device_model_specs"][0]
        for template in _raw_templates(prod_copy)
    )

    snapshot = {path.name: path.read_bytes() for path in prod_copy.glob("*.yaml")}
    second = promote(
        DEFAULT_CI_CONFIG,
        DEFAULT_DEV_DIR,
        prod_copy,
        **PINS,
    )
    assert second["changed_files"] == []
    assert {
        path.name: path.read_bytes() for path in prod_copy.glob("*.yaml")
    } == snapshot


def test_partial_multiweight_release_keeps_sibling_docs_and_stable_group(tmp_path):
    prod_copy = tmp_path / "prod"
    shutil.copytree(DEFAULT_PROD_DIR, prod_copy)
    empty_config = _write_ci(tmp_path, {})
    promote(
        empty_config,
        DEFAULT_DEV_DIR,
        prod_copy,
        version="base",
        tt_metal_commit="base-metal",
    )
    release_config = _write_ci(
        tmp_path,
        {"Llama-3.1-8B": _vllm_release_entry("N150")},
    )

    promote(
        release_config,
        DEFAULT_DEV_DIR,
        prod_copy,
        version="next",
        tt_metal_commit="next-metal",
        vllm_commit="next-vllm",
    )
    docs = tmp_path / "docs"
    with contextlib.redirect_stdout(io.StringIO()):
        generate_doc_pages(
            [
                template
                for filename in MODEL_SPEC_CATALOG_FILES
                for template in load_templates_from_yaml(prod_copy / filename)
            ],
            str(docs),
        )

    n150 = (docs / "llm" / "Llama-3.1-8B_n150.md").read_text()
    n300 = (docs / "llm" / "Llama-3.1-8B_n300.md").read_text()
    assert "Llama-3.1-8B-Instruct" in n150
    assert "Additional released configurations" in n150
    assert "Llama-3.1-8B-Instruct" in n300

    llama_groups = {
        template["catalog_group"]
        for template in _raw_templates(prod_copy)
        if template.get("model_display_name") == "Llama-3.1-8B"
        and template["impl"] == "tt_transformers"
        and any(
            weight
            in {
                "meta-llama/Llama-3.1-8B",
                "meta-llama/Llama-3.1-8B-Instruct",
            }
            for weight in template["weights"]
        )
    }
    assert len(llama_groups) == 1
    assert next(iter(llama_groups)).startswith("doc:")
