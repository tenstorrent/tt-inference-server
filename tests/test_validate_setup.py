# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Tests for bind mount permission validation in workflows/validate_setup.py."""

import os
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from reference_config.agentic_traces.agentic_traces_config import (
    AGENTIC_TRACES_CONFIGS,
    TraceSource,
)
from reference_config.evals.eval_config import (
    EvalConfig,
    EvalTask,
    SWEbenchEvalConfig,
    TerminalBenchEvalConfig,
)
from workflows.runtime_config import RuntimeConfig
from workflows.utils import check_path_permissions_for_uid
from workflows.validate_setup import (
    _check_image_version_supported,
    _try_fix_path_permissions_for_uid,  # noqa: F401
    validate_agentic_task_capabilities,
    validate_bind_mount_permissions,
    validate_local_setup,
    validate_local_server_paths,
    validate_runtime_args,
    validate_setup,
)
from workflows.workflow_types import WorkflowVenvType


class TestAgenticTaskCapabilityAdmission:
    @staticmethod
    def _runtime(workflow="release", **overrides):
        values = {
            "workflow": workflow,
            "agentic_benchmark": None,
            "external_agentic_contract": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _spec(model_name, max_context, *, impl="quetzal"):
        return SimpleNamespace(
            model_id=f"id_{impl}_{model_name}_p300x2",
            model_name=model_name,
            hf_model_repo=f"test/{model_name}",
            impl=SimpleNamespace(impl_id=impl),
            device_model_spec=SimpleNamespace(
                device="P300X2",
                max_context=max_context,
                # Deliberately smaller than the harness concurrency. The
                # admission contract is per-request context, not client queueing.
                max_concurrency=1,
            ),
        )

    @staticmethod
    def _terminal_task(*, max_input=128, max_output=64, model_info=True):
        agent_kwargs = {}
        if model_info:
            agent_kwargs["model_info"] = {
                "max_input_tokens": max_input,
                "max_output_tokens": max_output,
            }
        return EvalTask(
            task_name="terminal_bench_2",
            workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
            agentic_eval_config=TerminalBenchEvalConfig(
                dataset="terminal-bench/terminal-bench-2",
                agent="terminus-2",
                n_concurrent_trials=99,
                agent_kwargs=agent_kwargs,
            ),
        )

    def test_exact_qwen_quetzal_release_omits_explicitly_ineligible_tb2(self):
        # The generated row lives in the dev catalog while unit tests load the
        # prod catalog by default. Bind its exact advertised C1/S8192 profile
        # to the authoritative shared Qwen eval config.
        spec = self._spec("Qwen3.6-27B", 8192)

        validate_agentic_task_capabilities(spec, self._runtime())

    def test_exact_qwen_native_release_omits_same_ineligible_tb2(self):
        # The shared task contract is implementation-independent and also
        # exceeds the native row. Both default release paths omit it through
        # the task's explicit 344064-token catalogue floor.
        spec = self._spec("Qwen3.6-27B", 262144, impl="qwen36_blackhole")
        validate_agentic_task_capabilities(spec, self._runtime())

    def test_explicit_qwen_tb2_selection_still_fails_closed(self):
        spec = self._spec("Qwen3.6-27B", 8192)
        with pytest.raises(ValueError) as exc:
            validate_agentic_task_capabilities(
                spec,
                self._runtime(agentic_benchmark="tb2.0"),
            )

        message = str(exc.value)
        assert "before host/server/device setup" in message
        assert "implementation='quetzal'" in message
        assert "available_context=8192" in message
        assert "required_context=344064" in message
        assert "max_input_tokens=262144 + max_output_tokens=81920" in message

    def test_unmarked_oversized_release_task_still_fails_closed(self):
        task = self._terminal_task(max_input=128, max_output=64)
        spec = self._spec("unmarked-model", 191)
        with patch.dict(
            "workflows.validate_setup.EVAL_CONFIGS",
            {spec.model_name: EvalConfig(spec.hf_model_repo, [task])},
        ), pytest.raises(ValueError, match="required_context=192"):
            validate_agentic_task_capabilities(spec, self._runtime())

    def test_exact_gpt_quetzal_release_admits_bounded_swe_with_headroom(self):
        spec = self._spec("gpt-oss-120b", 8192)

        validate_agentic_task_capabilities(spec, self._runtime())

    def test_exact_gpt_quetzal_release_omits_task_below_s8192(self):
        spec = self._spec("gpt-oss-120b", 8191)

        validate_agentic_task_capabilities(spec, self._runtime())

    def test_explicit_gpt_swe_selection_requires_declared_s8192_envelope(self):
        spec = self._spec("gpt-oss-120b", 8191)

        with pytest.raises(ValueError) as exc:
            validate_agentic_task_capabilities(
                spec,
                self._runtime(agentic_benchmark="swe_bench_verified"),
            )

        message = str(exc.value)
        assert "task='swe_bench_verified'" in message
        assert "available_context=8191" in message
        assert "required_context=8192" in message
        assert "max_input_tokens=5120 + max_output_tokens=2048" in message
        assert "min_context_required=8192" in message

    def test_explicit_gpt_swe_selection_accepts_s8192(self):
        spec = self._spec("gpt-oss-120b", 8192)

        validate_agentic_task_capabilities(
            spec,
            self._runtime(agentic_benchmark="swe_bench_verified"),
        )

    def test_adequate_context_passes_without_concurrency_gate(self):
        task = self._terminal_task(max_input=128, max_output=64)
        spec = self._spec("adequate-model", 192, impl="native")
        with patch.dict(
            "workflows.validate_setup.EVAL_CONFIGS",
            {spec.model_name: EvalConfig(spec.hf_model_repo, [task])},
        ):
            validate_agentic_task_capabilities(spec, self._runtime())

    def test_swebench_uses_explicit_input_plus_output_budget(self):
        task = EvalTask(
            task_name="swe_bench_verified",
            workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
            swebench_eval_config=SWEbenchEvalConfig(
                dataset_name="SWE-bench/SWE-bench_Verified",
                max_input_tokens=256,
                max_output_tokens=128,
                n_concurrent_trials=50,
            ),
        )
        spec = self._spec("swe-model", 383, impl="native")
        with patch.dict(
            "workflows.validate_setup.EVAL_CONFIGS",
            {spec.model_name: EvalConfig(spec.hf_model_repo, [task])},
        ), pytest.raises(ValueError) as exc:
            validate_agentic_task_capabilities(spec, self._runtime())

        message = str(exc.value)
        assert "task='swe_bench_verified'" in message
        assert "available_context=383" in message
        assert "required_context=384" in message
        assert "max_input_tokens=256 + max_output_tokens=128" in message

    def test_standalone_agentic_validates_only_cli_selected_task(self):
        malformed_unselected = self._terminal_task(model_info=False)
        selected = EvalTask(
            task_name="swe_bench_verified",
            workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
            swebench_eval_config=SWEbenchEvalConfig(
                dataset_name="SWE-bench/SWE-bench_Verified",
                max_input_tokens=256,
                max_output_tokens=128,
            ),
        )
        spec = self._spec("selected-model", 384, impl="native")
        with patch.dict(
            "workflows.validate_setup.EVAL_CONFIGS",
            {
                spec.model_name: EvalConfig(
                    spec.hf_model_repo, [malformed_unselected, selected]
                )
            },
        ):
            validate_agentic_task_capabilities(
                spec,
                self._runtime(workflow="agentic", agentic_benchmark="swebench"),
            )

    @pytest.mark.parametrize(
        "max_input,max_output,model_info,expected",
        [
            (128, 64, False, "model_info must be an object"),
            (128, None, True, "max_output_tokens must be a positive integer"),
            (True, 64, True, "max_input_tokens must be a positive integer"),
        ],
    )
    def test_missing_or_malformed_task_budget_fails_closed(
        self, max_input, max_output, model_info, expected
    ):
        task = self._terminal_task(
            max_input=max_input,
            max_output=max_output,
            model_info=model_info,
        )
        spec = self._spec("malformed-model", 4096)
        with patch.dict(
            "workflows.validate_setup.EVAL_CONFIGS",
            {spec.model_name: EvalConfig(spec.hf_model_repo, [task])},
        ), pytest.raises(ValueError) as exc:
            validate_agentic_task_capabilities(spec, self._runtime())

        message = str(exc.value)
        assert "model='malformed-model'" in message
        assert "implementation='quetzal'" in message
        assert "available_context=4096" in message
        assert "required_context=undeclared" in message
        assert expected in message

    def test_rejection_precedes_every_later_setup_stage(self):
        spec = self._spec("Qwen3.6-27B", 8192)
        runtime = RuntimeConfig(
            model="Qwen3.6-27B",
            workflow="release",
            device="p300x2",
            impl="quetzal",
            docker_server=True,
            agentic_benchmark="tb2.0",
        )
        later_stages = (
            "validate_custom_weights",
            "validate_local_setup",
            "validate_bind_mount_permissions",
            "validate_local_server_paths",
        )
        mocks = [patch(f"workflows.validate_setup.{name}") for name in later_stages]
        started = [mock.start() for mock in mocks]
        try:
            with patch.dict(
                "workflows.validate_setup.MODEL_SPECS", {spec.model_id: spec}
            ), pytest.raises(ValueError, match="required_context=344064"):
                validate_setup(spec, runtime, "/unused/runtime-model-spec.json")
        finally:
            for mock in reversed(mocks):
                mock.stop()

        assert all(not mocked.called for mocked in started)


class TestCheckPathPermissionsForUid:
    """Tests for check_path_permissions_for_uid helper."""

    def test_nonexistent_path(self, tmp_path):
        ok, reason = check_path_permissions_for_uid(tmp_path / "nonexistent", uid=1000)
        assert not ok
        assert "does not exist" in reason

    def test_owner_has_read(self, tmp_path):
        """Owner UID matches, read bit set."""
        d = tmp_path / "owned"
        d.mkdir()
        uid = os.getuid()
        ok, reason = check_path_permissions_for_uid(d, uid=uid)
        assert ok
        assert reason == ""

    def test_owner_lacks_read(self, tmp_path):
        """Owner UID matches but read bit is cleared."""
        d = tmp_path / "no_read"
        d.mkdir()
        os.chmod(d, 0o300)
        try:
            uid = os.getuid()
            ok, reason = check_path_permissions_for_uid(d, uid=uid)
            assert not ok
            assert "lacks read permission" in reason
            assert "owner" in reason
        finally:
            os.chmod(d, 0o700)

    def test_owner_has_write(self, tmp_path):
        d = tmp_path / "writable"
        d.mkdir()
        uid = os.getuid()
        ok, reason = check_path_permissions_for_uid(d, uid=uid, need_write=True)
        assert ok

    def test_owner_lacks_write(self, tmp_path):
        d = tmp_path / "no_write"
        d.mkdir()
        os.chmod(d, 0o500)
        try:
            uid = os.getuid()
            ok, reason = check_path_permissions_for_uid(d, uid=uid, need_write=True)
            assert not ok
            assert "lacks write permission" in reason
        finally:
            os.chmod(d, 0o700)

    def test_directory_lacks_execute(self, tmp_path):
        """Directory without execute bit blocks traversal."""
        d = tmp_path / "no_exec"
        d.mkdir()
        os.chmod(d, 0o600)
        try:
            uid = os.getuid()
            ok, reason = check_path_permissions_for_uid(d, uid=uid)
            assert not ok
            assert "traverse" in reason
        finally:
            os.chmod(d, 0o700)

    def test_other_uid_world_readable(self, tmp_path):
        """Non-owner, non-group UID can read if world-readable."""
        d = tmp_path / "world_read"
        d.mkdir()
        os.chmod(d, 0o755)
        # UID 0 is root; use a UID that is not the owner and not in the group.
        # We use a mock to force the "other" code path.
        fake_uid = 99999
        with patch("workflows.utils.get_groups_for_uid", return_value=set()):
            ok, reason = check_path_permissions_for_uid(d, uid=fake_uid)
        assert ok

    def test_other_uid_not_world_readable(self, tmp_path):
        """Non-owner, non-group UID cannot read without world-read bit."""
        d = tmp_path / "no_world_read"
        d.mkdir()
        os.chmod(d, 0o750)
        fake_uid = 99999
        with patch("workflows.utils.get_groups_for_uid", return_value=set()):
            ok, reason = check_path_permissions_for_uid(d, uid=fake_uid)
        assert not ok
        assert "lacks read permission" in reason
        assert "other" in reason

    def test_other_uid_world_writable(self, tmp_path):
        d = tmp_path / "world_write"
        d.mkdir()
        os.chmod(d, 0o757)
        fake_uid = 99999
        with patch("workflows.utils.get_groups_for_uid", return_value=set()):
            ok, reason = check_path_permissions_for_uid(
                d, uid=fake_uid, need_write=True
            )
        assert ok

    def test_other_uid_not_world_writable(self, tmp_path):
        d = tmp_path / "no_world_write"
        d.mkdir()
        os.chmod(d, 0o755)
        fake_uid = 99999
        with patch("workflows.utils.get_groups_for_uid", return_value=set()):
            ok, reason = check_path_permissions_for_uid(
                d, uid=fake_uid, need_write=True
            )
        assert not ok
        assert "lacks write permission" in reason

    def test_group_member_can_read(self, tmp_path):
        """UID in the file's group can read with group-read bit."""
        d = tmp_path / "group_read"
        d.mkdir()
        st = d.stat()
        os.chmod(d, 0o750)
        fake_uid = 99999
        with patch(
            "workflows.utils.get_groups_for_uid",
            return_value={st.st_gid},
        ):
            ok, reason = check_path_permissions_for_uid(d, uid=fake_uid)
        assert ok

    def test_file_permissions(self, tmp_path):
        """Regular file (not directory) does not require execute bit."""
        f = tmp_path / "readable_file.txt"
        f.write_text("data")
        os.chmod(f, 0o644)
        uid = os.getuid()
        ok, reason = check_path_permissions_for_uid(f, uid=uid)
        assert ok


class TestValidateBindMountPermissions:
    """Tests for validate_bind_mount_permissions."""

    def _make_args(self, **overrides):
        defaults = {
            "image_user": str(os.getuid()),
            "host_volume": None,
            "host_hf_cache": None,
            "host_weights_dir": None,
        }
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_no_bind_mounts_passes(self):
        """No host paths set -- nothing to validate."""
        args = self._make_args()
        validate_bind_mount_permissions(args)

    def test_host_volume_writable_passes(self, tmp_path):
        d = tmp_path / "volume"
        d.mkdir()
        args = self._make_args(host_volume=str(d))
        validate_bind_mount_permissions(args)

    def test_host_volume_not_writable_auto_fixed(self, tmp_path):
        """Auto-fix adds write permission when current user owns the directory."""
        d = tmp_path / "ro_volume"
        d.mkdir()
        os.chmod(d, 0o500)
        try:
            args = self._make_args(host_volume=str(d))
            validate_bind_mount_permissions(args)
            assert os.access(d, os.W_OK)
        finally:
            os.chmod(d, 0o700)

    def test_host_volume_not_writable_raises_when_fix_fails(self, tmp_path):
        d = tmp_path / "ro_volume"
        d.mkdir()
        os.chmod(d, 0o500)
        try:
            args = self._make_args(host_volume=str(d))
            with patch(
                "workflows.validate_setup._try_fix_path_permissions_for_uid",
                return_value=False,
            ):
                with pytest.raises(
                    ValueError, match="Bind mount permission check failed"
                ):
                    validate_bind_mount_permissions(args)
        finally:
            os.chmod(d, 0o700)

    def test_host_hf_cache_readable_passes(self, tmp_path):
        d = tmp_path / "hf_cache"
        d.mkdir()
        args = self._make_args(host_hf_cache=str(d))
        validate_bind_mount_permissions(args)

    def test_host_hf_cache_not_readable_auto_fixed(self, tmp_path):
        """Auto-fix adds read+execute permission for read-only mounts."""
        d = tmp_path / "hf_cache_noperm"
        d.mkdir()
        os.chmod(d, 0o300)
        try:
            args = self._make_args(host_hf_cache=str(d))
            validate_bind_mount_permissions(args)
            assert os.access(d, os.R_OK)
        finally:
            os.chmod(d, 0o700)

    def test_host_hf_cache_not_readable_raises_when_fix_fails(self, tmp_path):
        d = tmp_path / "hf_cache_noperm"
        d.mkdir()
        os.chmod(d, 0o300)
        try:
            args = self._make_args(host_hf_cache=str(d))
            with patch(
                "workflows.validate_setup._try_fix_path_permissions_for_uid",
                return_value=False,
            ):
                with pytest.raises(
                    ValueError, match="Bind mount permission check failed"
                ):
                    validate_bind_mount_permissions(args)
        finally:
            os.chmod(d, 0o700)

    def test_host_weights_dir_readable_passes(self, tmp_path):
        d = tmp_path / "weights"
        d.mkdir()
        args = self._make_args(host_weights_dir=str(d))
        validate_bind_mount_permissions(args)

    def test_host_weights_dir_not_readable_raises_when_fix_fails(self, tmp_path):
        d = tmp_path / "weights_noperm"
        d.mkdir()
        os.chmod(d, 0o300)
        try:
            args = self._make_args(host_weights_dir=str(d))
            with patch(
                "workflows.validate_setup._try_fix_path_permissions_for_uid",
                return_value=False,
            ):
                with pytest.raises(
                    ValueError, match="Bind mount permission check failed"
                ):
                    validate_bind_mount_permissions(args)
        finally:
            os.chmod(d, 0o700)

    def test_nonexistent_host_volume_is_created(self, tmp_path):
        missing = tmp_path / "missing"
        args = self._make_args(host_volume=str(missing))
        validate_bind_mount_permissions(args)
        assert missing.is_dir()

    def test_nonexistent_nested_host_volume_is_created(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        args = self._make_args(host_volume=str(nested))
        validate_bind_mount_permissions(args)
        assert nested.is_dir()

    def test_other_uid_volume_auto_fixed(self, tmp_path):
        """Auto-fix adds other rwx bits when container UID is not owner/group."""
        d = tmp_path / "other_fix"
        d.mkdir()
        os.chmod(d, 0o700)
        fake_uid = 99999
        args = self._make_args(image_user=str(fake_uid), host_volume=str(d))
        with patch("workflows.utils.get_groups_for_uid", return_value=set()):
            validate_bind_mount_permissions(args)
        mode = os.stat(d).st_mode
        assert mode & 0o007 == 0o007

    def test_error_message_includes_fix_guidance(self, tmp_path):
        d = tmp_path / "noperm"
        d.mkdir()
        os.chmod(d, 0o500)
        try:
            args = self._make_args(host_volume=str(d))
            with patch(
                "workflows.validate_setup._try_fix_path_permissions_for_uid",
                return_value=False,
            ):
                with pytest.raises(ValueError, match="chmod/chown"):
                    validate_bind_mount_permissions(args)
        finally:
            os.chmod(d, 0o700)


class TestLocalServerValidation:
    def _make_model_spec(self):
        model_spec = MagicMock()
        model_spec.model_id = "id_tt-transformers_Mistral-7B-Instruct-v0.3_n150"
        model_spec.model_name = "Mistral-7B-Instruct-v0.3"
        model_spec.inference_engine = "vLLM"
        return model_spec

    def _make_runtime_config(self):
        runtime_config = RuntimeConfig(
            model="Mistral-7B-Instruct-v0.3",
            workflow="server",
            device="n150",
            local_server=True,
        )
        runtime_config.runtime_model_spec = {
            "hf_weights_repo": "mistralai/Mistral-7B-Instruct-v0.3"
        }
        return runtime_config

    def test_runtime_args_require_tt_metal_home_for_local_server(self):
        model_spec = self._make_model_spec()
        runtime_config = self._make_runtime_config()

        with patch.dict(
            "workflows.validate_setup.MODEL_SPECS",
            {model_spec.model_id: model_spec},
        ):
            with pytest.raises(
                ValueError, match="requires --tt-metal-home or TT_METAL_HOME"
            ):
                validate_runtime_args(model_spec, runtime_config)

    def test_runtime_args_allow_tt_metal_home_from_env(self):
        model_spec = self._make_model_spec()
        runtime_config = self._make_runtime_config()
        runtime_config.tt_metal_home = "/env/tt-metal"

        with patch.dict(
            "workflows.validate_setup.MODEL_SPECS",
            {model_spec.model_id: model_spec},
        ):
            validate_runtime_args(model_spec, runtime_config)

    def test_validate_local_server_paths_passes(self, tmp_path):
        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        build_lib_dir = tt_metal_home / "build" / "lib"
        vllm_dir = tt_metal_home / "vllm"
        python_bin_dir.mkdir(parents=True)
        build_lib_dir.mkdir(parents=True)
        vllm_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        args = Namespace(
            local_server=True,
            tt_metal_home=str(tt_metal_home),
            tt_metal_python_venv_dir=None,
            vllm_dir=None,
            host_hf_cache=None,
            host_weights_dir=None,
            runtime_model_spec={
                "hf_weights_repo": "mistralai/Mistral-7B-Instruct-v0.3"
            },
        )

        validate_local_server_paths(args)

    def test_validate_local_server_paths_requires_python(self, tmp_path):
        tt_metal_home = tmp_path / "tt-metal"
        (tt_metal_home / "vllm").mkdir(parents=True)
        (tt_metal_home / "build" / "lib").mkdir(parents=True)

        args = Namespace(
            local_server=True,
            tt_metal_home=str(tt_metal_home),
            tt_metal_python_venv_dir=None,
            vllm_dir=None,
            host_hf_cache=None,
            host_weights_dir=None,
            runtime_model_spec={
                "hf_weights_repo": "mistralai/Mistral-7B-Instruct-v0.3"
            },
        )

        with pytest.raises(ValueError, match="python venv interpreter"):
            validate_local_server_paths(args)

    def test_validate_local_server_paths_requires_cached_hf_snapshot(self, tmp_path):
        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        build_lib_dir = tt_metal_home / "build" / "lib"
        vllm_dir = tt_metal_home / "vllm"
        python_bin_dir.mkdir(parents=True)
        build_lib_dir.mkdir(parents=True)
        vllm_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        hf_home = tmp_path / "hf_home"
        hf_home.mkdir()
        args = Namespace(
            local_server=True,
            tt_metal_home=str(tt_metal_home),
            tt_metal_python_venv_dir=None,
            vllm_dir=None,
            host_hf_cache=str(hf_home),
            host_weights_dir=None,
            runtime_model_spec={
                "hf_weights_repo": "mistralai/Mistral-7B-Instruct-v0.3"
            },
        )

        with pytest.raises(ValueError, match="did not contain a cached snapshot"):
            validate_local_server_paths(args)

    def test_validate_local_server_paths_accepts_cached_hf_snapshot(self, tmp_path):
        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        build_lib_dir = tt_metal_home / "build" / "lib"
        vllm_dir = tt_metal_home / "vllm"
        python_bin_dir.mkdir(parents=True)
        build_lib_dir.mkdir(parents=True)
        vllm_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        snapshot_dir = (
            tmp_path
            / "hf_home"
            / "hub"
            / "models--mistralai--Mistral-7B-Instruct-v0.3"
            / "snapshots"
            / "abc123"
        )
        snapshot_dir.mkdir(parents=True)

        args = Namespace(
            local_server=True,
            tt_metal_home=str(tt_metal_home),
            tt_metal_python_venv_dir=None,
            vllm_dir=None,
            host_hf_cache=str(tmp_path / "hf_home"),
            host_weights_dir=None,
            runtime_model_spec={
                "hf_weights_repo": "mistralai/Mistral-7B-Instruct-v0.3"
            },
        )

        validate_local_server_paths(args)

    def test_validate_local_server_paths_accepts_explicit_vllm_dir(self, tmp_path):
        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        build_lib_dir = tt_metal_home / "build" / "lib"
        python_bin_dir.mkdir(parents=True)
        build_lib_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        explicit_vllm_dir = tmp_path / "custom-vllm"
        explicit_vllm_dir.mkdir()
        args = Namespace(
            local_server=True,
            tt_metal_home=str(tt_metal_home),
            tt_metal_python_venv_dir=None,
            vllm_dir=str(explicit_vllm_dir),
            host_hf_cache=None,
            host_weights_dir=None,
            runtime_model_spec={
                "hf_weights_repo": "mistralai/Mistral-7B-Instruct-v0.3"
            },
        )

        validate_local_server_paths(args)

    @patch("workflows.validate_setup.run_command", return_value=0)
    @patch("workflows.validate_setup.ensure_readwriteable_dir")
    @patch("workflows.validate_setup.get_default_workflow_root_log_dir")
    def test_validate_local_setup_checks_vllm_installation(
        self,
        mock_get_log_dir,
        mock_ensure_dir,
        mock_run_command,
        tmp_path,
    ):
        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        python_bin_dir.mkdir(parents=True)
        venv_python = python_bin_dir / "python"
        venv_python.write_text("")

        model_spec = self._make_model_spec()
        runtime_config = self._make_runtime_config()
        runtime_config.tt_metal_home = str(tt_metal_home)
        runtime_config.skip_system_sw_validation = True

        mock_get_log_dir.return_value = tmp_path / "logs"

        validate_local_setup(model_spec, runtime_config, tmp_path / "runtime.json")

        mock_ensure_dir.assert_called_once_with(tmp_path / "logs")
        # Two probes, both unconditional: vLLM importable, then the vllm-tt-plugin
        # `tt` platform entry point registered. Neither needs a source checkout.
        assert mock_run_command.call_count == 2
        calls = mock_run_command.call_args_list
        assert calls[0].args[0] == [str(venv_python), "-c", "import vllm"]
        probe_cmd = calls[1].args[0]
        assert probe_cmd[0] == str(venv_python)
        assert probe_cmd[1] == "-c"
        assert "import vllm_tt_plugin" in probe_cmd[2]
        assert "vllm.platform_plugins" in probe_cmd[2]
        assert "'tt' in eps" in probe_cmd[2]

    @patch("workflows.validate_setup.run_command", return_value=1)
    @patch("workflows.validate_setup.ensure_readwriteable_dir")
    @patch("workflows.validate_setup.get_default_workflow_root_log_dir")
    def test_validate_local_setup_raises_when_vllm_not_installed(
        self,
        mock_get_log_dir,
        mock_ensure_dir,
        mock_run_command,
        tmp_path,
    ):
        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        python_bin_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        model_spec = self._make_model_spec()
        runtime_config = self._make_runtime_config()
        runtime_config.tt_metal_home = str(tt_metal_home)
        runtime_config.skip_system_sw_validation = True

        mock_get_log_dir.return_value = tmp_path / "logs"

        with pytest.raises(ValueError, match="requires the `vllm` Python package"):
            validate_local_setup(model_spec, runtime_config, tmp_path / "runtime.json")

        mock_ensure_dir.assert_called_once_with(tmp_path / "logs")
        mock_run_command.assert_called_once()

    @patch("workflows.validate_setup.ensure_readwriteable_dir")
    @patch("workflows.validate_setup.get_default_workflow_root_log_dir")
    def test_validate_local_setup_probes_tt_plugin_without_a_vllm_checkout(
        self,
        mock_get_log_dir,
        mock_ensure_dir,
        tmp_path,
    ):
        """The `tt` entry-point probe must run unconditionally.

        vllm-tt-plugin is its own repository now, installed into the tt-metal
        venv by its docs/install-vllm-tt.sh. There is no plugin source tree
        under the vLLM checkout to gate the probe on, and gating on one meant
        validation was silently skipped in exactly the case it was needed --
        letting `vllm serve` fail later with no TT platform registered.
        """
        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        python_bin_dir.mkdir(parents=True)
        venv_python = python_bin_dir / "python"
        venv_python.write_text("")

        model_spec = self._make_model_spec()
        runtime_config = self._make_runtime_config()
        runtime_config.tt_metal_home = str(tt_metal_home)
        runtime_config.skip_system_sw_validation = True

        mock_get_log_dir.return_value = tmp_path / "logs"

        # Deliberately no vLLM checkout staged anywhere. `import vllm` succeeds
        # (rc=0); the plugin probe fails (rc=1) and must surface as an error.
        with patch(
            "workflows.validate_setup.run_command", side_effect=[0, 1]
        ) as mock_run_command:
            with pytest.raises(ValueError, match=r"`vllm_tt_plugin` Python package"):
                validate_local_setup(
                    model_spec, runtime_config, tmp_path / "runtime.json"
                )

            assert mock_run_command.call_count == 2
            probe_script = mock_run_command.call_args_list[1].args[0][2]
            assert "import vllm_tt_plugin" in probe_script
            assert "vllm.platform_plugins" in probe_script


class TestAgenticTracesRegistration:
    """A model with no AGENTIC_TRACES_CONFIGS entry must be refused up front.

    The agentic-traces venv setup clones and installs InferenceX, which takes
    minutes, and a release child runs after the evals and benchmarks. Both would
    be wasted before the missing config surfaced, so validation has to fail here
    instead.
    """

    UNREGISTERED_ID = "id_tt-transformers_Llama-3.1-8B-Instruct_n150"

    def _spec(self, model_id, model_name="Llama-3.1-8B-Instruct"):
        spec = MagicMock()
        spec.model_id = model_id
        spec.model_name = model_name
        spec.inference_engine = "vLLM"
        return spec

    def _runtime_config(self, workflow, **overrides):
        config = RuntimeConfig(
            model="Llama-3.1-8B-Instruct",
            workflow=workflow,
            device="n150",
            **overrides,
        )
        return config

    def _validate(self, spec, runtime_config):
        # No license, deliberately: these paths must not depend on one. The
        # host running the suite may happen to have a key, so pin it to absent.
        with patch.dict(
            "workflows.validate_setup.MODEL_SPECS", {spec.model_id: spec}
        ), patch(
            "workflows.validate_setup._swarmone_license_available",
            return_value=False,
        ):
            validate_runtime_args(spec, runtime_config)

    def test_unregistered_model_is_rejected(self):
        spec = self._spec(self.UNREGISTERED_ID)
        with pytest.raises(AssertionError, match="no AGENTIC_TRACES_CONFIGS entry"):
            self._validate(spec, self._runtime_config("agentic_traces"))

    def test_the_error_names_the_model_and_where_to_register_it(self):
        """The message is the only guidance a new model's onboarder gets."""
        spec = self._spec(self.UNREGISTERED_ID)
        with pytest.raises(AssertionError) as exc:
            self._validate(spec, self._runtime_config("agentic_traces"))
        message = str(exc.value)
        assert "Llama-3.1-8B-Instruct" in message
        assert self.UNREGISTERED_ID in message
        assert "reference_config/agentic_traces/agentic_traces_config.py" in message
        # Registering without pinning a ref makes results incomparable.
        assert "git ref" in message

    def test_registered_model_passes(self):
        """Guards against the assertion firing on an onboarded model.

        Runs without a SwarmOne license on purpose: the plain sweep must stay
        usable for a model that merely has an opt-in SwarmOne run configured.
        """
        registered_id = next(iter(AGENTIC_TRACES_CONFIGS))
        spec = self._spec(registered_id, model_name="Kimi-K2.7-Code")
        config = RuntimeConfig(
            model="Kimi-K2.7-Code", workflow="agentic_traces", device="super_cluster"
        )
        self._validate(spec, config)

    def test_release_opted_into_agentic_traces_is_also_gated(self, monkeypatch):
        """The release child runs the same sweep, so it needs the same entry."""
        spec = self._spec(self.UNREGISTERED_ID)
        config = self._runtime_config("release", agentic_traces=True)
        with pytest.raises(AssertionError, match="no AGENTIC_TRACES_CONFIGS entry"):
            self._validate(spec, config)

    def test_plain_release_is_not_gated(self, monkeypatch):
        """Without the opt-in there is no agentic-traces child to protect."""
        spec = self._spec(self.UNREGISTERED_ID)
        monkeypatch.setattr(
            "reference_config.evals.eval_config.EVAL_CONFIGS",
            {spec.model_name: object()},
        )
        monkeypatch.setattr(
            "workflows.validate_setup.can_dispatch_to_engine", lambda *a, **k: True
        )
        self._validate(spec, self._runtime_config("release"))


class TestSwarmOneLicenseGate:
    """A swarmone run needs a swo-bench license, checked up front in run.py.

    The check fires only when swarmone will actually run, so a multi-minute
    venv build is never wasted on a sweep that will fail the moment the
    swo-bench driver looks for its key -- while a plain sweep of a model that
    merely *has* a swarmone run configured stays license-free, because swarmone
    is opt-in.
    """

    def _spec(self):
        registered_id = next(iter(AGENTIC_TRACES_CONFIGS))
        spec = MagicMock()
        spec.model_id = registered_id
        spec.model_name = "Kimi-K2.7-Code"
        spec.inference_engine = "vLLM"
        return spec

    def _config(self, **overrides):
        return RuntimeConfig(
            model="Kimi-K2.7-Code",
            workflow="agentic_traces",
            device="super_cluster",
            **overrides,
        )

    def _validate(self, spec, config, *, license_available):
        with patch.dict(
            "workflows.validate_setup.MODEL_SPECS", {spec.model_id: spec}
        ), patch(
            "workflows.validate_setup._swarmone_license_available",
            return_value=license_available,
        ):
            validate_runtime_args(spec, config)

    def test_configured_but_unselected_swarmone_needs_no_license(self):
        """The regression this gate once caused: Kimi gaining a swarmone run
        made ``--workflow agentic_traces`` demand a license from everyone."""
        spec = self._spec()
        assert any(
            run.trace_source is TraceSource.SWARMONE
            for run in AGENTIC_TRACES_CONFIGS[spec.model_id].runs
        ), "fixture no longer covers the mixed-source case this test guards"
        self._validate(spec, self._config(), license_available=False)

    def test_missing_license_is_rejected_when_swarmone_is_selected(self):
        spec = self._spec()
        with pytest.raises(ValueError, match="SwarmOne license"):
            self._validate(
                spec,
                self._config(agentic_traces_sources="swarmone"),
                license_available=False,
            )

    def test_the_error_says_how_to_proceed_without_swarmone(self):
        """Someone who just wants the InferenceX sweep needs a way out."""
        spec = self._spec()
        with pytest.raises(ValueError) as exc:
            self._validate(
                spec,
                self._config(agentic_traces_sources="swarmone"),
                license_available=False,
            )
        message = str(exc.value)
        assert "SWO_LICENSE_KEY" in message
        assert "--agentic-traces-sources swarmone" in message

    def test_present_license_allows_an_explicit_swarmone_sweep(self):
        spec = self._spec()
        self._validate(
            spec,
            self._config(agentic_traces_sources="swarmone"),
            license_available=True,
        )

    def test_selecting_swarmone_alongside_inferencex_still_needs_a_license(self):
        spec = self._spec()
        with pytest.raises(ValueError, match="SwarmOne license"):
            self._validate(
                spec,
                self._config(agentic_traces_sources="inferencex_agentx,swarmone"),
                license_available=False,
            )

    def test_selecting_only_inferencex_skips_the_license_check(self):
        """Narrowing away from swarmone means no license is needed."""
        spec = self._spec()
        self._validate(
            spec,
            self._config(agentic_traces_sources="inferencex_agentx"),
            license_available=False,
        )


class TestCheckImageVersionSupported:
    """run.py only emits the post-0.11 vLLM docker contract; pre-0.11 vLLM
    specs (or pre-0.11 override images) must be refused with a clear migration
    message rather than silently producing a broken docker run.

    Media-inference-server and forge images use a different Dockerfile that
    isn't affected by the 0.11.0 vLLM interface refactor, so the check must
    NOT fire for them.
    """

    def _spec(self, version, engine="vLLM"):
        s = MagicMock()
        s.version = version
        s.inference_engine = engine
        return s

    def test_post_0_11_vllm_versions_pass(self):
        # Boundary and above must not raise.
        _check_image_version_supported(self._spec("0.11.0"))
        _check_image_version_supported(self._spec("0.11.1"))
        _check_image_version_supported(self._spec("0.13.0"))
        _check_image_version_supported(self._spec("1.0.0"))

    def test_pre_0_11_vllm_versions_raise(self):
        for v in ("0.10.9", "0.10.1", "0.10.0", "0.9.0", "0.2.0"):
            with pytest.raises(RuntimeError, match="not supported"):
                _check_image_version_supported(self._spec(v))

    def test_error_names_exact_tag_to_checkout(self):
        # The matching tt-inference-server release tag is `v<spec.version>`.
        with pytest.raises(RuntimeError) as exc:
            _check_image_version_supported(self._spec("0.10.1"))
        msg = str(exc.value)
        assert "v0.10.1" in msg
        assert "git checkout v0.10.1" in msg

    def test_unparseable_versions_pass(self):
        # `dev`, `latest`, empty — let the runtime decide, matches main.
        _check_image_version_supported(self._spec("dev"))
        _check_image_version_supported(self._spec("latest"))
        _check_image_version_supported(self._spec(""))

    def test_media_engine_not_blocked_by_pre_0_11(self):
        # tt-media-inference-server images aren't affected by the vLLM
        # 0.11.0 interface change. Pre-0.11 media specs must still run.
        for v in ("0.2.0", "0.5.0", "0.9.0", "0.10.0", "0.10.1"):
            _check_image_version_supported(self._spec(v, engine="media"))

    def test_forge_engine_not_blocked_by_pre_0_11(self):
        # forge images are also outside the vLLM image-interface refactor.
        for v in ("0.2.0", "0.9.0", "0.10.1"):
            _check_image_version_supported(self._spec(v, engine="forge"))


class TestVersionParsers:
    """workflows.utils version helpers, used by _check_image_version_supported."""

    def test_parse_version_tuple(self):
        from workflows.utils import parse_version_tuple

        assert parse_version_tuple("0.10.0") == (0, 10, 0)
        assert parse_version_tuple("0.11.0") == (0, 11, 0)
        assert parse_version_tuple("0.9") == (0, 9, 0)
        assert parse_version_tuple("0.13.0-suffix") == (0, 13, 0)
        # Non-version / empty / non-string inputs return None.
        assert parse_version_tuple("dev") is None
        assert parse_version_tuple("") is None
        assert parse_version_tuple(None) is None  # type: ignore[arg-type]

    def test_parse_image_version(self):
        from workflows.utils import parse_image_version

        assert parse_image_version("ghcr.io/foo/bar:0.10.0-abc") == (0, 10, 0)
        assert parse_image_version("ghcr.io/foo/bar:0.11.0") == (0, 11, 0)
        assert parse_image_version("ghcr.io/foo/bar:0.9-abc") == (0, 9, 0)
        # Unparseable tags / no tag / no version-prefix / None.
        assert parse_image_version("ghcr.io/foo/bar:dev") is None
        assert parse_image_version("ghcr.io/foo/bar:latest") is None
        assert parse_image_version("ghcr.io/foo/bar") is None
        assert parse_image_version(None) is None  # type: ignore[arg-type]
