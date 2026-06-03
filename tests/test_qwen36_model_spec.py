# SPDX-License-Identifier: Apache-2.0
import os
os.environ.setdefault("MODEL_SPECS_ENV", "dev")

from workflows.model_spec import MODEL_SPECS
from workflows.workflow_types import DeviceTypes, ModelStatusTypes


def _find_qwen36_spec():
    for spec in MODEL_SPECS.values():
        if getattr(spec, "hf_model_repo", "") == "Qwen/Qwen3.6-27B" \
                and spec.device_type == DeviceTypes.BLACKHOLE_GALAXY:
            return spec
    return None


def test_qwen36_spec_present_for_blackhole_galaxy():
    spec = _find_qwen36_spec()
    assert spec is not None, "Qwen3.6-27B BLACKHOLE_GALAXY spec not found"
    assert spec.device_model_spec.max_context == 262144
    assert spec.impl.code_path == "models/demos/qwen3_6_galaxy_v2"
    assert spec.device_model_spec.max_concurrency == 32
    assert spec.status == ModelStatusTypes.EXPERIMENTAL
