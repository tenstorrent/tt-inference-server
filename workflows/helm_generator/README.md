# Helm Generator

Python package that generates entries in
[`charts/tt-inference-server/values.yaml`](../../charts/tt-inference-server/values.yaml)
from the `ModelSpec` catalog in
[`workflows/model_spec.py`](../model_spec.py).

The goal: keep `values.yaml` in lockstep with the source-of-truth model catalog
without hand-editing image tags, env vars, probe delays, or resource budgets.
Run the generator after a model spec changes; it produces an updated
`values.yaml` while preserving comments and any hand-tuned keys the operator
has added.

## Schema produced

```yaml
models:
  Llama-3.1-8B-Instruct:
    defaultEngine: vllm                     # picked when --set engine is omitted
    vllm:
      galaxy:
        defaultImpl: tt_transformers        # picked when --set impl is omitted
        impls:
          tt_transformers:
            image: { repository: ..., tag: ... }
            progressDeadlineSeconds: 5400
            resources: { requests: { memory: 20Gi } }
            probes:
              liveness:  { initialDelaySeconds: 2400 }
              readiness: { initialDelaySeconds: 2400 }
            env:
              - { name: ARCH_NAME,    value: "wormhole_b0" }
              - { name: MESH_DEVICE,  value: "TG" }
              # ...
```

## What lives where

| File | Role |
|---|---|
| [base_mapper.py](base_mapper.py) | Abstract `HelmValuesMapper` + the helpers each mapper uses to build an `HelmImplConfig` from a `ModelSpec`. |
| [vllm/](vllm), [media/](media), [forge/](forge) | One subclass per inference engine; each declares `engine`, probe paths, and owned leaf paths. |
| [schema.py](schema.py) | Dataclasses for the YAML shape: `HelmImage`, `HelmProbe`, `HelmResources`, `HelmImplConfig`, `HelmModelSpec`. |
| [device.py](device.py) | `DeviceTypes` → values.yaml device key (lowercase enum name) + multihost filter. |
| [yaml_io.py](yaml_io.py) | `ruamel.yaml` round-trip loader/dumper that preserves comments. |
| [merge.py](merge.py) | Inserts/updates entries inside a loaded `CommentedMap`; only overwrites the leaf paths a mapper claims to own. |
| [cli.py](cli.py) | `argparse` entry point; uniqueness + default-impl validation; defaultEngine computation. |
| [errors.py](errors.py) | `GenerateHelmValuesError` raised when upstream data can't be turned into a valid values.yaml. |

## How to run

The chart's [values.yaml](../../charts/tt-inference-server/values.yaml) is the
output; rerun the generator any time the catalog changes.

```bash
# Regenerate the whole file (idempotent — second run is a no-op)
python -m workflows.helm_generator

# Preview without writing
python -m workflows.helm_generator --dry-run

# Regenerate just one model or device
python -m workflows.helm_generator --model Qwen3-32B --device galaxy
python -m workflows.helm_generator --engine media
```

Flags:

| Flag | Effect |
|---|---|
| `--values-path PATH` | Override the default `charts/tt-inference-server/values.yaml`. |
| `--model NAME`, repeatable | Only emit specs whose `model_name` matches. |
| `--device NAME`, repeatable | Only emit specs whose lowercased device matches. |
| `--engine {vllm,media,forge}`, repeatable | Only emit specs for the given engine(s). |
| `--dry-run` | Print resulting YAML to stdout; do not write the file. |
| `--include-multihost` | Don't skip `DUAL_GALAXY` / `QUAD_GALAXY` (skipped by default). |
| `-v / --verbose` | Verbose logging. |

Exit code `2` on a `GenerateHelmValuesError`
(duplicate `(model, device, engine, impl)` tuple, or a multi-impl group missing
a unique `default_impl=True`).

## Running tests

The test suite lives in [tests/test_helm_generator/](../../tests/test_helm_generator).

```bash
# from repo root
python -m pytest tests/test_helm_generator/ -v
```

Coverage:

- Unit tests per module (schema, device, yaml_io, merge, each mapper).
- CLI integration: end-to-end mapping + merge, idempotency, dry-run.
- `tests/test_helm_generator/test_helpers_tpl.py` invokes `helm template`
  against the live chart for resolution / error scenarios; skipped
  automatically if `helm` is not on `PATH`.

## Trying the chart locally

The generator outputs a chart; render it with `helm template` to see what a
Helm install would produce:

```bash
# single-engine model
helm template charts/tt-inference-server \
  --set model=Llama-3.1-8B-Instruct \
  --set device=galaxy \
  --set hfToken=fake

# multi-engine model — defaultEngine resolves to vllm
helm template charts/tt-inference-server \
  --set model=Llama-3.1-70B \
  --set device=t3k \
  --set hfToken=fake

# same as above, but pick a non-default engine
helm template charts/tt-inference-server \
  --set model=Llama-3.1-70B \
  --set device=t3k \
  --set engine=media \
  --set hfToken=fake

# pick a non-default impl
helm template charts/tt-inference-server \
  --set model=Qwen3-32B \
  --set device=galaxy \
  --set impl=tt_transformers \
  --set hfToken=fake

# mount pre-downloaded weights (sets MODEL_WEIGHTS_DIR + MODEL_WEIGHTS_PATH)
helm template charts/tt-inference-server \
  --set model=Llama-3.1-8B-Instruct \
  --set device=galaxy \
  --set hfToken=fake \
  --set hfCacheDir=/data/weights
```

## Adding a new engine

1. Add the value to `InferenceEngine` in
   [workflows/workflow_types.py](../workflow_types.py).
2. Create a new subpackage `<engine>/mapper.py` with a subclass of
   `HelmValuesMapper` that declares `engine`, `liveness_path`,
   `readiness_path`, and `owned_leaf_paths()`. Use
   [vllm/mapper.py](vllm/mapper.py) as a template.
3. Register the mapper in [`__init__.py`](__init__.py)'s `MAPPERS` dict.
4. If the engine needs different branching, update
   [`charts/tt-inference-server/templates/configmap.yaml`](../../charts/tt-inference-server/templates/configmap.yaml)
   and
   [`charts/tt-inference-server/templates/deployment.yaml`](../../charts/tt-inference-server/templates/deployment.yaml)
   — both already read the engine via `tt-inference-server.resolvedEngine`.
5. Add an entry to `ENGINE_PRECEDENCE` so `defaultEngine` resolution stays
   deterministic.
