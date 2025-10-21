# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from domain.base_image_request import BaseImageRequest
from tt_model_runners.sd35_runner import TTSD35Runner
import ttnn
import pytest

#Test components of sd35 device runner
def test_sd35_runner(monkeypatch) -> None:
    #configure the settings to use the correct model and device
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "")
    if ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY:
        runner_device = "galaxy"
    elif ttnn.cluster.get_cluster_type() ==ttnn.cluster.ClusterType.T3K:
        runner_device = "quietbox"
    else:
        pytest.skip("Unsupported device. Skipping test")

    monkeypatch.setenv("MODEL", "stable-diffusion-3.5-large")
    monkeypatch.setenv("DEVICE", runner_device)
    
    from ..config.settings import get_settings

    settings = get_settings()
    print(settings)
    prompt = "Happy Robot in a park"

    runner = TTSD35Runner(device_id=settings.device_ids[0])
    runner.load_model(runner.get_device())
    tt_out = runner.run_inference([BaseImageRequest.model_construct(
        prompt=prompt,
        negative_prompt=""
    )])

    tt_out[0].save("test_sd35_runner.png")