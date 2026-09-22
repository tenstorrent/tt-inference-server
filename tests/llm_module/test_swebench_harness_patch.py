# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from llm_module.agentic.swebench import _write_swebench_harness_patch


def test_matplotlib_harness_patch_is_narrow_and_valid_python(tmp_path):
    patch_path = _write_swebench_harness_patch(tmp_path) / "sitecustomize.py"
    source = patch_path.read_text(encoding="utf-8")

    compile(source, str(patch_path), "exec")
    assert 'self.instance_id != "matplotlib__matplotlib-25332"' in source
    assert "export CONDA_SOLVER=classic" in source
    assert "release-branch-semver" in source
    assert "TestSpec.setup_env_script = property" in source
    assert "TestSpec.eval_script = property" in source
