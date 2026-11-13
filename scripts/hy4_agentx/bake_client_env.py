#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build time: create the AGENTIC_TRACES client venv (python 3.12, InferenceX at the pinned ref, aiperf) inside the image tree, for the model in model.env.

usage (cwd = tt-inference-server tree):  bake_client_env.py <catalog json> [B=4] [--venvs A,B,..] [--dry-run]
--venvs: WorkflowVenvType names to bake (default AGENTIC_TRACES = the AgentX client; add BENCHMARKS_VLLM,EVALS_COMMON,EVALS_AGENTIC to make the grid and
the evals runnable offline inside the image: UNVERIFIED, DSV4 baked only AGENTIC_TRACES).
Unlike the DSV4 bake script (scripts of tt-bench-harness/inimage) this takes the spec from the release CATALOG JSON (no yaml catalog entry is needed), so the
image can be baked before the yaml catalog knows the model. --dry-run: load the entry, resolve the AgentX config (model_id -> pinned InferenceX ref, dataset,
concurrency) and print them; no network, no venv. Needs network (PyPI, github) without --dry-run; run twice (tt-release-packaging build-paths.md)."""
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness_model import HarnessModel  # noqa: E402


def main(argv) -> int:
    venvs = ["AGENTIC_TRACES"]
    if "--venvs" in argv:
        i = argv.index("--venvs")
        venvs = argv[i + 1].split(",")
        argv = argv[:i] + argv[i + 2:]
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    m = HarnessModel.load()
    b = int(args[1]) if len(args) > 1 else m.batch_sizes[0]
    entry = json.load(open(args[0]))["model_specs"][m.model_id][m.device_key]["vLLM"][m.impl_id(b)]
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(entry, tmp)
    tmp.close()
    sys.path.insert(0, os.getcwd())
    os.environ["MODEL_SPECS_ENV"] = "dev"
    from workflows.model_spec import ModelSpec

    spec = ModelSpec.from_json(tmp.name)
    from reference_config.agentic_traces.agentic_traces_config import get_agentic_traces_config

    cfg = get_agentic_traces_config(spec)
    if cfg is None:
        print("ERROR: no AgenticTracesConfig for model_id", spec.model_id)
        return 3
    print("venvs", venvs)
    print("model_id", spec.model_id, "inferencex_git_ref", cfg.inferencex_git_ref, "runs", [(r.public_dataset, r.concurrency) for r in cfg.runs])
    if "--dry-run" in argv:
        return 0
    from workflows.bootstrap_uv import bootstrap_uv

    bootstrap_uv()
    from workflows.workflow_types import WorkflowVenvType
    from workflows.workflow_venvs import VENV_CONFIGS

    for v in venvs:
        assert VENV_CONFIGS[WorkflowVenvType[v]].setup(model_spec=spec), v
    print("BAKE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
