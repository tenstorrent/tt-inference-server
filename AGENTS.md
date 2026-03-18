# Agent Guide

Read `docs/agent_memory.md` before broad changes or repo exploration.

## Mental Model
- `run.py` + `workflows/`: host-side control plane for runtime resolution, validation, Docker launch, workflow orchestration, and reporting.
- `vllm-tt-metal/`: primary LLM serving path via Dockerized Tenstorrent vLLM.
- `tt-media-server/`: non-LLM and some OpenAI-style serving surfaces, plus separate Python and C++ server paths.
- `benchmarking/`, `evals/`, `stress_tests/`, `tests/`, `scripts/release/`: model-readiness and release pipeline.

## Source Of Truth
- Runtime selection and CLI: `run.py`
- Workflow dispatch: `workflows/run_workflows.py`
- Model catalog and support matrix: `workflows/model_spec.py`
- vLLM container contract: `vllm-tt-metal/src/run_vllm_api_server.py`
- Media API routing: `tt-media-server/open_ai_api/__init__.py`
- Media config/env resolution: `tt-media-server/config/settings.py`

## Caveats
- Prefer code over docs when they differ.
- `--local-server` only supports vLLM-backed model specs and always uses host filesystem persistence.
- If no host storage flags are passed with `--local-server`, it defaults to `REPO_ROOT/persistent_volume/` for logs, weights, and TT caches.
- Non-`reports` workflows auto-run `reports`.
- `release` is a wrapper workflow; check `workflows/run_workflows.py` for the exact sequence.
- Media-server routes depend on `settings.model_service`; not every API exists in every mode.
