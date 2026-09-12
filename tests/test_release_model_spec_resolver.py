# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.release.model_spec_resolver import (
    ReleaseCombo,
    collect_release_combos,
    load_dev_model_spec_sources,
    resolve_release_combo,
    resolve_release_combos,
)
from workflows.model_spec import MODEL_SPEC_CATALOG_FILES, resolve_model_spec
from workflows.workflow_types import DeviceTypes, InferenceEngine


REPO_ROOT = Path(__file__).resolve().parents[1]
CI_CONFIG = REPO_ROOT / ".github" / "workflows" / "models-ci-config.json"
DEV_CATALOG = REPO_ROOT / "workflows" / "model_specs" / "dev"


def _write_dev_catalog(tmp_path, *, default_impl=True):
    dev_dir = tmp_path / "dev"
    dev_dir.mkdir()
    (dev_dir / "llm.yaml").write_text(
        textwrap.dedent(
            f"""\
            templates:
            - weights:
                - Qwen/Qwen3-32B
                - Qwen/Qwen3-14B
              impl: qwen3_32b_galaxy
              inference_engine: VLLM
              device_model_specs:
                - device: GALAXY
                  max_concurrency: 32
                  max_context: 131072
                  default_impl: {str(default_impl).lower()}
                - device: BLACKHOLE_GALAXY
                  max_concurrency: 8
                  max_context: 32768
                  default_impl: {str(default_impl).lower()}

            - weights:
                - Qwen/Qwen3-32B
              impl: tt_transformers
              inference_engine: VLLM
              device_model_specs:
                - device: GALAXY
                  max_concurrency: 1
                  max_context: 32768
                  default_impl: false
            """
        )
    )
    for filename in MODEL_SPEC_CATALOG_FILES:
        if filename != "llm.yaml":
            (dev_dir / filename).write_text("templates: []\n")
    return dev_dir


def test_collect_release_combos_is_ordered_and_normalizes_aliases():
    ci_config = {
        "models": {
            "model-a": {
                "implementations": [
                    {
                        "inference_engine": "vLLM",
                        "ci": {
                            "release": {"devices": ["GALAXY", "BH_GALAXY", "GALAXY"]}
                        },
                    },
                    {
                        "inference_engine": "FORGE",
                        "ci": {"nightly": {"devices": ["P150"]}},
                    },
                ]
            },
            "model-b": {
                "inference_engine": "MEDIA",
                "ci": {"release": {"devices": ["P150"]}},
            },
        }
    }

    assert collect_release_combos(ci_config) == [
        ReleaseCombo("model-a", InferenceEngine.VLLM, DeviceTypes.GALAXY),
        ReleaseCombo(
            "model-a",
            InferenceEngine.VLLM,
            DeviceTypes.BLACKHOLE_GALAXY,
        ),
        ReleaseCombo("model-b", InferenceEngine.MEDIA, DeviceTypes.P150),
    ]


def test_collect_release_combos_rejects_all_devices():
    ci_config = {
        "models": {
            "model-a": {
                "inference_engine": "vLLM",
                "ci": {"release": {"devices": "ALL"}},
            }
        }
    }

    with pytest.raises(ValueError, match="cannot be 'ALL'"):
        collect_release_combos(ci_config)


def test_resolve_release_combo_returns_exact_leaf_and_source(tmp_path):
    dev_dir = _write_dev_catalog(tmp_path)
    sources = load_dev_model_spec_sources(dev_dir)
    combo = ReleaseCombo(
        "Qwen3-32B",
        InferenceEngine.VLLM,
        DeviceTypes.GALAXY,
    )

    resolved = resolve_release_combo(combo, sources)

    assert resolved.identity == (
        "Qwen/Qwen3-32B",
        "GALAXY",
        "vLLM",
        "qwen3_32b_galaxy",
    )
    assert resolved.source_path == dev_dir / "llm.yaml"
    assert resolved.template_index == 0
    assert resolved.weight_index == 0
    assert resolved.device_index == 0
    assert resolved.source_template.weights == [
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-14B",
    ]
    assert len(resolved.source_template.device_model_specs) == 2


def test_dev_source_loading_ignores_yaml_files_runtime_does_not_load(tmp_path):
    dev_dir = _write_dev_catalog(tmp_path)
    (dev_dir / "extra.yaml").write_text("not: valid: yaml\n")

    sources = load_dev_model_spec_sources(dev_dir)

    assert sources
    assert {source.path.name for source in sources} == {"llm.yaml"}


def test_resolve_release_combo_requires_explicit_default(tmp_path):
    sources = load_dev_model_spec_sources(
        _write_dev_catalog(tmp_path, default_impl=False)
    )
    combo = ReleaseCombo(
        "Qwen3-14B",
        InferenceEngine.VLLM,
        DeviceTypes.GALAXY,
    )

    with pytest.raises(ValueError, match="explicit default implementation"):
        resolve_release_combo(combo, sources)


@pytest.mark.parametrize(
    "ci_config,error",
    [
        ({"models": []}, "'models' must be an object"),
        (
            {"models": {"model-a": {"implementations": ["bad"]}}},
            "implementation must be an object",
        ),
        (
            {
                "models": {
                    "model-a": {
                        "inference_engine": "vLLM",
                        "ci": {"release": {"devices": [123]}},
                    }
                }
            },
            "device must be a string",
        ),
    ],
)
def test_collect_release_combos_rejects_malformed_shapes(ci_config, error):
    with pytest.raises(ValueError, match=error):
        collect_release_combos(ci_config)


def test_real_release_scope_resolves_to_runtime_equivalent_dev_leaves():
    combos = collect_release_combos(json.loads(CI_CONFIG.read_text()))
    sources = load_dev_model_spec_sources(DEV_CATALOG)

    resolved = resolve_release_combos(combos, sources)

    assert combos
    assert len(resolved) == len(combos)
    specs = [source.spec for source in sources]
    for item in resolved:
        runtime_spec = resolve_model_spec(
            specs,
            model=item.combo.model_name,
            device=item.combo.device,
            engine=item.combo.engine,
            catalog_name="test dev catalog",
        )
        assert runtime_spec is item.model_spec

    qwen = next(
        item
        for item in resolved
        if item.combo.model_name == "Qwen/Qwen3-32B"
        and item.combo.device == DeviceTypes.GALAXY
        and item.combo.engine == InferenceEngine.VLLM
    )
    assert qwen.identity == (
        "Qwen/Qwen3-32B",
        "GALAXY",
        "vLLM",
        "qwen3_32b_galaxy",
    )


def test_real_release_scope_matches_dev_runtime_subprocess():
    script = textwrap.dedent(
        """\
        import json
        from pathlib import Path
        from scripts.release.model_spec_resolver import collect_release_combos
        from workflows.model_spec import get_runtime_model_spec, model_spec_leaf_identity

        config = json.loads(Path(".github/workflows/models-ci-config.json").read_text())
        identities = []
        for combo in collect_release_combos(config):
            spec, _, _ = get_runtime_model_spec(
                model=combo.model_name,
                device=combo.device.to_string(),
                engine=combo.engine.value,
            )
            identities.append(model_spec_leaf_identity(spec))
        print(json.dumps(identities))
        """
    )
    env = {**os.environ, "MODEL_SPECS_ENV": "dev", "PYTHONDONTWRITEBYTECODE": "1"}

    output = subprocess.check_output(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        text=True,
    )

    combos = collect_release_combos(json.loads(CI_CONFIG.read_text()))
    sources = load_dev_model_spec_sources(DEV_CATALOG)
    expected = [list(item.identity) for item in resolve_release_combos(combos, sources)]
    assert json.loads(output) == expected
