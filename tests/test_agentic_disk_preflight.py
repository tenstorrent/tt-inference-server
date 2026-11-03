import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from workflows import agentic_disk_preflight as gate
from llm_module.agentic import preflight_evidence as evidence


def test_early_gate_precedes_host_setup_and_server():
    tree = ast.parse(Path("run.py").read_text())
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    calls = {
        n.func.id: n.lineno
        for n in ast.walk(main)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in {"early_agentic_disk_check", "setup_host"}
    }
    assert calls["early_agentic_disk_check"] < calls["setup_host"]


def test_opt_in_preserves_other_models():
    with patch.object(gate, "collect_disk_evidence") as collect:
        assert (
            gate.early_agentic_disk_check(
                SimpleNamespace(impl=SimpleNamespace(impl_id="other")),
                SimpleNamespace(workflow="agentic"),
            )
            is None
        )
    collect.assert_not_called()


def test_selected_early_gate_uses_release_reserve_and_fails_closed():
    model = SimpleNamespace(
        hf_model_repo="google/gemma-4-31B-it",
        impl=SimpleNamespace(impl_id="gemma4_fresh_autoport"),
    )
    with patch.object(
        gate, "required_host_paths", return_value={"runner_diagnostics": "/diag"}
    ), patch.object(
        gate, "collect_disk_evidence", side_effect=RuntimeError("insufficient")
    ) as collect, pytest.raises(
        RuntimeError, match="insufficient"
    ):
        gate.early_agentic_disk_check(model, SimpleNamespace(workflow="agentic"))
    assert collect.call_args.args[2] == 512 * 1024**3
    assert collect.call_args.kwargs["required_paths"] == {"runner_diagnostics": "/diag"}
    assert collect.call_args.kwargs["min_free_inodes"] == 100_000


def test_runner_diagnostics_required_in_ci(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    with pytest.raises(RuntimeError, match="diagnostics"):
        gate.required_host_paths()
    (tmp_path / "_diag").mkdir()
    assert gate.required_host_paths()["runner_diagnostics"] == str(tmp_path / "_diag")


def test_nonexistent_cache_checks_existing_filesystem(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    paths = gate.required_host_paths(
        SimpleNamespace(host_volume=str(tmp_path / "new" / "cache"))
    )
    assert paths["host_volume"] == str(tmp_path)


@pytest.mark.parametrize(
    "free,inodes,missing,passed",
    [
        (512, 100_000, False, True),
        (511, 100_000, False, False),
        (512, 99_999, False, False),
        (512, 100_000, True, False),
    ],
)
def test_distinct_runner_volume_byte_inode_admission(
    tmp_path, free, inodes, missing, passed
):
    target = tmp_path / "jobs"
    diag = tmp_path / "diag"
    diag.mkdir()
    real_stat = evidence.os.stat

    def stat(path, *a, **kw):
        if str(path) == str(diag):
            if missing:
                raise FileNotFoundError()
            return SimpleNamespace(st_dev=2)
        if str(path) == str(target):
            return SimpleNamespace(st_dev=1)
        return real_stat(path, *a, **kw)

    def usage(path):
        return SimpleNamespace(
            total=2048, used=0, free=free if str(path) == str(diag) else 1024
        )

    def vfs(path):
        return SimpleNamespace(
            f_favail=inodes if str(path) == str(diag) else 200_000, f_files=300_000
        )

    target.mkdir()
    with patch.object(
        evidence, "_docker", return_value={"unavailable": "mock"}
    ), patch.object(evidence.shutil, "disk_usage", side_effect=usage), patch.object(
        evidence.os, "stat", side_effect=stat
    ), patch.object(
        evidence.os, "statvfs", side_effect=vfs
    ):
        if passed:
            result = evidence.collect_disk_evidence(
                target,
                "test",
                512,
                required_paths={"runner_diagnostics": str(diag)},
                min_free_inodes=100_000,
            )
            assert result["reservation_check"]["observed_free_bytes_by_device"] == {
                "1": 1024,
                "2": 512,
            }
        else:
            with pytest.raises(RuntimeError):
                evidence.collect_disk_evidence(
                    target,
                    "test",
                    512,
                    required_paths={"runner_diagnostics": str(diag)},
                    min_free_inodes=100_000,
                )


@pytest.mark.parametrize(
    "diagnostic_free,job_free,diagnostic_device,passed",
    [(8, 512, 2, True), (7, 512, 2, False), (8, 511, 2, False), (8, 512, 1, False)],
)
def test_role_floors_keep_data_reserve_and_max_on_aliases(
    tmp_path, diagnostic_free, job_free, diagnostic_device, passed
):
    target = tmp_path / "jobs"
    target.mkdir()
    diag = tmp_path / "diag"
    diag.mkdir()
    real_stat = evidence.os.stat

    def stat(path, *a, **kw):
        if str(path) == str(diag):
            return SimpleNamespace(st_dev=diagnostic_device)
        if str(path) == str(target):
            return SimpleNamespace(st_dev=1)
        return real_stat(path, *a, **kw)

    def usage(path):
        return SimpleNamespace(
            total=2048,
            used=0,
            free=diagnostic_free if str(path) == str(diag) else job_free,
        )

    with patch.object(
        evidence, "_docker", return_value={"unavailable": "mock"}
    ), patch.object(evidence.shutil, "disk_usage", side_effect=usage), patch.object(
        evidence.os, "stat", side_effect=stat
    ), patch.object(
        evidence.os,
        "statvfs",
        return_value=SimpleNamespace(f_favail=200_000, f_files=300_000),
    ):
        kwargs = dict(
            required_paths={"runner_diagnostics": str(diag)},
            required_path_min_free_bytes={"runner_diagnostics": 8},
            min_free_inodes=100_000,
        )
        if passed:
            result = evidence.collect_disk_evidence(target, "roles", 512, **kwargs)
            assert result["reservation_check"]["minimum_free_bytes_by_device"] == {
                "1": 512,
                "2": 8,
            }
        else:
            with pytest.raises(RuntimeError):
                evidence.collect_disk_evidence(target, "roles", 512, **kwargs)


def test_role_policy_separates_model_data_from_diagnostic_tmp():
    paths = {
        k: "unused"
        for k in [
            "host_root",
            "host_tmp",
            "runner_diagnostics",
            "host_volume",
            "HOST_HF_HOME",
        ]
    }
    floors = gate.host_path_floors(paths, 512 * 1024**3)
    assert floors == {
        "host_root": 8 * 1024**3,
        "host_tmp": 8 * 1024**3,
        "runner_diagnostics": 8 * 1024**3,
        "host_volume": 512 * 1024**3,
        "HOST_HF_HOME": 512 * 1024**3,
    }


def test_bootstrap_gate_avoids_eval_package_dependencies():
    import subprocess
    import sys

    script = """
import importlib.abc, sys
from types import SimpleNamespace
class NoEvalPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'requests','numpy','torch','transformers','datasets','scipy','PIL','llm_module'}:
            raise ModuleNotFoundError('eval package intentionally unavailable: ' + fullname)
sys.meta_path.insert(0, NoEvalPackages())
import run
import workflows.agentic_disk_preflight as gate
gate.collect_disk_evidence = lambda *a, **kw: {'reservation_check': {'passed': True}}
gate.required_host_paths = lambda runtime: {'host_root': '/'}
gate.early_agentic_disk_check(SimpleNamespace(hf_model_repo='google/gemma-4-31B-it', impl=SimpleNamespace(impl_id='gemma4_fresh_autoport')), SimpleNamespace(workflow='agentic'))
assert 'llm_module' not in sys.modules
print('MINIMAL_HOST_GATE_OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "MINIMAL_HOST_GATE_OK" in result.stdout


@pytest.mark.parametrize("dev_mode", [False, True])
def test_actual_production_cli_registry_reaches_gate_without_eval_deps(dev_mode):
    import subprocess
    import sys

    script = """
import importlib.abc, sys
class NoEvalPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'requests','numpy','torch','transformers','datasets','scipy','PIL','llm_module'}:
            raise ModuleNotFoundError('eval package intentionally unavailable: ' + fullname)
sys.meta_path.insert(0, NoEvalPackages())
sys.argv = ['run.py','--model','gemma-4-31B-it','--workflow','agentic','--device','p300x2','--docker-server','--dev-mode','--impl','gemma4-fresh-autoport','--ci-mode','--override-docker-image','ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:273153ff5fe93b8141320da86ba446f5ad73b8143e649778fa9cd533b6faca74']
import run
runtime, model = run.resolve_runtime(run.parse_arguments())
assert model.impl.impl_id == 'gemma4_fresh_autoport'
assert model.impl.impl_name == 'gemma4-fresh-autoport'
assert model.env_vars['TT_SERVER_BOOT_ATTEMPTS'] == '1'
assert model.device_model_spec.tensor_cache_timeout == 3600
import workflows.agentic_disk_preflight as gate
calls = []
def denied(*args, **kwargs):
    calls.append((args,kwargs))
    raise RuntimeError('actual admission rejected')
gate.collect_disk_evidence = denied
gate.required_host_paths = lambda runtime: {'runner_diagnostics':'/diag'}
try:
    gate.early_agentic_disk_check(model,runtime)
except RuntimeError as exc:
    assert str(exc) == 'actual admission rejected'
else:
    raise AssertionError('production CLI skipped required admission')
assert len(calls) == 1
assert calls[0][0][2] == 512*1024**3
assert calls[0][1]['min_free_inodes'] == 100000
assert calls[0][1]['required_path_min_free_bytes']['runner_diagnostics'] == 8*1024**3
assert 'llm_module' not in sys.modules
print('REAL_CLI_GATE_FAILS_CLOSED')
"""
    if not dev_mode:
        script = script.replace("'--dev-mode',", "")
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "REAL_CLI_GATE_FAILS_CLOSED" in result.stdout


@pytest.mark.parametrize(
    "thresholds", [[], [None], [0], [-1], [True], ["512"], [512, None]]
)
def test_actual_target_invalid_task_reserve_fails_closed(thresholds):
    from reference_config.evals.eval_config import EVAL_CONFIGS
    from workflows.model_spec import MODEL_SPECS

    model = MODEL_SPECS["id_gemma4-fresh-autoport_gemma-4-31B-it_p300x2"]
    tasks = [
        SimpleNamespace(agentic_eval_config=SimpleNamespace(preflight_min_free_bytes=t))
        for t in thresholds
    ]
    with patch.dict(
        EVAL_CONFIGS, {"google/gemma-4-31B-it": SimpleNamespace(tasks=tasks)}
    ), patch.object(gate, "collect_disk_evidence") as collect, pytest.raises(
        RuntimeError, match="positive disk reserve"
    ):
        gate.early_agentic_disk_check(model, SimpleNamespace(workflow="agentic"))
    collect.assert_not_called()


def test_actual_target_wrong_adapter_identity_cannot_skip_gate():
    with pytest.raises(RuntimeError, match="autoport identity"):
        gate.early_agentic_disk_check(
            SimpleNamespace(
                hf_model_repo="google/gemma-4-31B-it",
                impl=SimpleNamespace(impl_id="gemma4-fresh-autoport"),
            ),
            SimpleNamespace(workflow="agentic"),
        )


def test_actual_integer_workflow_enum_reaches_gate():
    from workflows.model_spec import MODEL_SPECS
    from workflows.workflow_types import WorkflowType

    model = MODEL_SPECS["id_gemma4-fresh-autoport_gemma-4-31B-it_p300x2"]
    assert isinstance(WorkflowType.AGENTIC.value, int)
    with patch.object(
        gate, "required_host_paths", return_value={"runner_diagnostics": "/diag"}
    ), patch.object(
        gate, "collect_disk_evidence", side_effect=RuntimeError("enum reached gate")
    ) as collect, pytest.raises(
        RuntimeError, match="enum reached gate"
    ):
        gate.early_agentic_disk_check(
            model, SimpleNamespace(workflow=WorkflowType.AGENTIC)
        )
    collect.assert_called_once()
