# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Published native closure must be applied before compiling TTNN."""
from pathlib import Path


def test_native_patch_runs_after_checkout_before_compile():
    dockerfile = Path(__file__).resolve().parents[2] / 'vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile'
    text = dockerfile.read_text()
    hook = 'python3 ${TT_METAL_HOME}/models/autoports/qwen_qwen3_8_27b/ci/bootstrap_metal.py ${TT_METAL_HOME}'
    assert hook in text
    assert text.index('git checkout ${TT_METAL_COMMIT_SHA_OR_TAG}') < text.index(hook)
    assert text.index('git submodule update --init --recursive') < text.index(hook)
    assert text.index(hook) < text.index('bash ./build_metal.sh')
    assert text.index(hook) < text.index('rm -rf ${TT_METAL_HOME}/.git')
