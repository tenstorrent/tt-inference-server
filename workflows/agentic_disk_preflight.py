"""Opt-in host admission checks for the generated Gemma agentic release route."""

import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile

# Loading this stdlib-only helper by filename avoids llm_module.__init__,
# whose eager driver imports require eval-venv packages before host setup.
_evidence_path = (
    Path(__file__).resolve().parents[1] / "llm_module/agentic/preflight_evidence.py"
)
_evidence_spec = importlib.util.spec_from_file_location(
    "_agentic_host_disk_evidence", _evidence_path
)
_evidence_module = importlib.util.module_from_spec(_evidence_spec)
_evidence_spec.loader.exec_module(_evidence_module)
collect_disk_evidence = _evidence_module.collect_disk_evidence

# Planning headroom, not measured peak consumption or a filesystem reservation.
MIN_FREE_INODES = 100_000


def _existing_parent(path):
    path = Path(path).expanduser().absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return str(path)


def required_host_paths(runtime=None):
    paths = {
        "host_root": "/",
        "host_checkout": str(Path.cwd()),
        "host_home": str(Path.home()),
        "host_tmp": tempfile.gettempdir(),
    }
    for name in ("host_volume", "host_hf_cache", "host_weights_dir"):
        value = getattr(runtime, name, None)
        if value:
            paths[name] = _existing_parent(value)
    for name in (
        "STORAGE_PATH",
        "PERSISTENT_VOLUME_ROOT",
        "HOST_HF_HOME",
        "RUNNER_TEMP",
    ):
        if os.environ.get(name):
            paths[name] = _existing_parent(os.environ[name])
    # The workflow executes this source on the runner host. Require its actual
    # diagnostics path rather than assuming it shares Docker's filesystem.
    if os.environ.get("GITHUB_ACTIONS") == "true":
        roots = [Path.cwd(), Path(os.environ.get("GITHUB_WORKSPACE", Path.cwd()))]
        candidates = [p / "_diag" for root in roots for p in (root, *root.parents)]
        diag = next((p for p in candidates if p.is_dir()), None)
        if diag is None:
            raise RuntimeError(
                "Runner diagnostics filesystem cannot be identified; capacity admission blocked"
            )
        paths["runner_diagnostics"] = str(diag)
    return paths


def host_path_floors(paths, data_reserve):
    """Small diagnostic/runtime volumes need headroom, not a model-data reserve."""
    diagnostic_roles = {
        "host_root",
        "host_checkout",
        "host_home",
        "host_tmp",
        "runner_diagnostics",
        "RUNNER_TEMP",
    }
    # 8GiB is a planning allowance for diagnostics/temp, not a measured peak.
    # A shared Docker/data filesystem inherits the maximum role floor below.
    return {
        name: 8 * 1024**3 if name in diagnostic_roles else data_reserve
        for name in paths
    }


def early_agentic_disk_check(model_spec, runtime):
    if getattr(model_spec, "hf_model_repo", None) != "google/gemma-4-31B-it":
        return None
    workflow = getattr(runtime, "workflow", None)
    # RuntimeConfig stores strings; WorkflowType uses integer enum values.
    workflow_name = (
        workflow if isinstance(workflow, str) else getattr(workflow, "name", None)
    )
    if not isinstance(workflow_name, str):
        raise RuntimeError("Gemma release workflow identity is unavailable")
    if workflow_name.lower() != "agentic":
        return None
    if (
        getattr(getattr(model_spec, "impl", None), "impl_id", None)
        != "gemma4_fresh_autoport"
    ):
        raise RuntimeError(
            "Gemma agentic release requires the generated autoport identity"
        )
    from reference_config.evals.eval_config import EVAL_CONFIGS

    tasks = EVAL_CONFIGS["google/gemma-4-31B-it"].tasks
    thresholds = [
        getattr(
            getattr(t, "agentic_eval_config", None), "preflight_min_free_bytes", None
        )
        for t in tasks
    ]
    if not thresholds or any(
        isinstance(t, bool)
        or not isinstance(t, (int, float))
        or not math.isfinite(t)
        or t <= 0
        for t in thresholds
    ):
        raise RuntimeError(
            "Gemma agentic release requires a positive disk reserve for every selected task"
        )
    threshold = max(thresholds)
    paths = required_host_paths(runtime)
    result = collect_disk_evidence(
        Path("workflow_logs/disk_preflight"),
        "before_host_setup",
        threshold,
        require_docker_root=True,
        required_paths=paths,
        required_path_min_free_bytes=host_path_floors(paths, threshold),
        min_free_inodes=MIN_FREE_INODES,
    )
    print(
        "EARLY_AGENTIC_DISK_PREFLIGHT " + json.dumps(result, sort_keys=True), flush=True
    )
    return result
