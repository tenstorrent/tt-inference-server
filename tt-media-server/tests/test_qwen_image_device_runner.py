# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn


def test_qwen_image_runner(monkeypatch) -> None:
    """Test QwenImage model runner on supported devices."""
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "")
    if ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY:
        runner_device = "galaxy"
    elif ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.T3K:
        runner_device = "t3k"
    else:
        pytest.skip("Unsupported device. Skipping test")

    monkeypatch.setenv("MODEL", "qwen-image")
    monkeypatch.setenv("MODEL_RUNNER", "tt-qwen-image")
    monkeypatch.setenv("DEVICE", runner_device)

    from domain.image_generate_request import ImageGenerateRequest
    from tt_model_runners.dit_runners import TTQwenImageRunner

    from config.settings import get_settings

    settings = get_settings()
    print(settings)
    prompt = "A beautiful sunset over the mountains"

    runner = TTQwenImageRunner(device_id=settings.device_ids[0])
    runner.warmup()
    tt_out = runner.run(
        [ImageGenerateRequest.model_construct(prompt=prompt, negative_prompt="")]
    )

    tt_out[0].save("test_qwen_image_runner.png")
