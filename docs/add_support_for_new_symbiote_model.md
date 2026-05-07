# How To Add Support for a New tt_symbiote Model

This guide is the tt_symbiote-specific complement to
[add_support_for_new_model.md](add_support_for_new_model.md). It covers the
five files a new tt_symbiote model needs.

The architecture and rationale live in
[tt_symbiote_integration_pipeline.md](tt_symbiote_integration_pipeline.md).
Read that first.

## Step 0: preconditions

1. The model already runs to completion in standalone pytest under
   `tt-metal/models/experimental/tt_symbiote/tests/test_<model>.py`.
2. You have read the hard rules in `CLAUDE.md` section 2: never edit
   upstream `vllm/`, never edit `tt-vllm-plugin/`, never modify a working
   `tt_transformers` entry in `model_spec.py`.
3. You know which device(s) you are targeting and have device access for
   the local server bring-up at the end of this guide.

## Step 1: scaffold the five files

Run the scaffolder to emit all five blocks at once:

```bash
cd ~/tt-inference-server
python3 scripts/add_symbiote_model.py \
    --hf-arch BailingMoeV2ForCausalLM \
    --weights inclusionAI/Ling-mini-2.0 \
    --short-name Ling-mini-2.0 \
    --device t3k \
    --max-context 2048 \
    --max-concurrency 1 \
    --tt-metal-commit 489d8b0 \
    --vllm-commit 6f6d817
```

The scaffolder is read-only: it prints five blocks (each marked with a
`# ===== FILE: ... =====` header) and never mutates the source repos.
Paste each block at its marked location.

The five files, in dependency order:

1. `~/tt-metal/models/experimental/tt_symbiote/vllm/generator_vllm_<model>.py`
   (NEW FILE) - the adapter subclass.
2. `~/tt-metal/models/experimental/tt_symbiote/vllm/__init__.py` - add the
   new entry to `SYMBIOTE_MODEL_REGISTRY`.
3. `~/tt-inference-server/workflows/model_spec.py` - add the
   `ModelSpecTemplate` near existing `impl=tt_symbiote_impl` entries.
4. `~/tt-inference-server/evals/eval_config.py` - add the `EvalConfig` to
   `EVAL_CONFIGS`.
5. `~/tt-inference-server/benchmarking/benchmark_targets/model_performance_reference.json`
   - add the perf-reference stub.

## Step 2: fill in the adapter

The scaffolded adapter has `# TODO` markers wherever model-specific code is
required. The full pattern is documented in
[tt_symbiote_integration_pipeline.md](tt_symbiote_integration_pipeline.md)
section 3. At minimum:

1. Set `MODEL_KEY` to a short uppercase identifier (e.g. `GEMMA4`, `LING`).
   It is used in the `[WATCHDOG] prefill_forward` log string as
   `TT_SYMBIOTE_<MODEL_KEY>_PREFILL_SYNC`.
2. Set `WATCHDOG_PREFILL_KERNEL_HINT` to the human-readable name of the
   model's TTNN attention-prefill kernel module.
3. Set `WARMUP_PREFILL_SEQ_LENS` to cover every ISL the benchmark sweep
   exercises against the spec's `max_context` cap.
4. Implement `_build_model_and_kv_cache`. Refer to
   [generator_vllm.py (Gemma-4)](../../tt-metal/models/experimental/tt_symbiote/vllm/generator_vllm.py)
   and `generator_vllm_ling.py` for working examples.

If the model ships HF custom code that references symbols removed in
transformers 5.x, apply the shims at MODULE TOP of the adapter file before
the `SymbioteAdapterBase` import. The shims must fire before
`AutoModelForCausalLM.from_pretrained` triggers HF dynamic-import. See
`generator_vllm_ling.py`.

## Step 3: validate the spec

Run the validator unit tests:

```bash
cd ~/tt-inference-server
PYTHONPATH=. ~/tt-metal/python_env/bin/python -m pytest tests/workflows/test_tt_symbiote_validator.py -v
```

`ModelSpecTemplate._validate_data` runs at module-import time and catches
common misconfigurations as `AssertionError`:

- `has_builtin_warmup` not `True`.
- `override_tt_config["enable_model_warmup"]` not `True`.
- `override_tt_config["trace_mode"]` not `none` (TRACED is incompatible).
- `env_vars["TT_SYMBIOTE_DISPATCHER"]` outside the allowed set.
- `env_vars["MESH_DEVICE"]` mismatch with the device.

Multi-chip devices without `fabric_config` only log a WARNING. The
recommendation is `FABRIC_1D_RING`; some attention kernels deadlock on
the linear `FABRIC_1D` default.

`ModelSpec.__post_init__` automatically injects defaults for tt_symbiote
specs (per-spec values win):

- `TT_SYMBIOTE_DISPATCHER=CPU`
- `TT_SYMBIOTE_DIAG=1`
- `TT_SYMBIOTE_PREFILL_WATCHDOG_SEC=60`
- `TT_SYMBIOTE_DECODE_WATCHDOG_SEC=30`
- `TT_SYMBIOTE_SYNC_EVERY_N_DECODES=32`
- `DISABLE_METAL_OP_TIMEOUT=1`

Override per-spec only when needed (e.g. Ling raises the prefill watchdog
to 180s for cold-boot JIT).

## Step 4: bring up the local server

Reset the device(s) and run the canonical bring-up command:

```bash
tt-smi -r 0,1,2,3                      # T3K example; adjust for your device

cd ~/tt-inference-server
export VLLM_USE_V1=1
export HF_TOKEN=hf_...                 # only if the HF repo is gated
python3 run.py \
    --model <YOUR-MODEL-ID> \
    --device <YOUR-DEVICE> \
    --workflow server \
    --local-server \
    --dev-mode \
    --tt-metal-home ~/tt-metal \
    --vllm-dir ~/vllm \
    --skip-system-sw-validation \
    --no-auth
```

The server is healthy when:

1. `Application startup complete` appears in the server log.
2. `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/v1/models`
   returns `200`.
3. A single short `/v1/chat/completions` request returns `200` with
   non-empty `choices[0].message.content`. Use the lightweight CLI tool:

```bash
python3 utils/local_server_prompt.py --no-auth --prompt "Hello, what is your name?"
```

When all three pass, the bring-up is done.

## Step 5: hand off

Per `CLAUDE.md` section 1, the agent's job ends at HTTP 200. After that,
the operator is responsible for the full benchmark / eval pass. For the
benchmark workflow, see
[tt_symbiote_integration_pipeline.md](tt_symbiote_integration_pipeline.md)
section 7 and
[add_support_for_new_model.md](add_support_for_new_model.md) Steps 3 and 4
(the same eval and perf-target patterns apply to tt_symbiote).

## Quick reference

- Design: [tt_symbiote_integration_pipeline.md](tt_symbiote_integration_pipeline.md).
- Local-server bring-up: [local_server_workflow.md](local_server_workflow.md).
- Workflow runner: [workflows_user_guide.md](workflows_user_guide.md).
- Hard rules: `CLAUDE.md` in the workspace root.
