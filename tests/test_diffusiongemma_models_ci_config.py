# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_CI_CONFIG = REPO_ROOT / ".github" / "workflows" / "models-ci-config.json"


def test_diffusiongemma_runs_nightly_and_release_on_p300x2():
    models = json.loads(MODELS_CI_CONFIG.read_text())["models"]
    config = models["diffusiongemma-26B-A4B-it"]

    assert config["inference_engine"] == "vLLM"
    assert config["ci"]["nightly"]["devices"] == ["P300X2"]
    assert config["ci"]["release"]["devices"] == ["P300X2"]
