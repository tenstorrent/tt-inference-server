# Promote dev model specs to prod — design

**Date:** 2026-06-02
**Script:** `scripts/release/promote_dev_spec_to_prod.py`

## Purpose

A single-purpose script: take `.github/workflows/models-ci-config.json`, find every
model-device combination marked for `release`, locate those same model specs in the
**dev** catalogue (`workflows/model_specs/dev/`), and copy them into the **prod**
catalogue (`workflows/model_specs/prod/`).

## Background

- The model spec catalogue lives in two parallel trees, one per environment:
  `workflows/model_specs/dev/` and `workflows/model_specs/prod/`. Each holds 7
  category YAML files: `llm.yaml`, `vlm.yaml`, `cnn.yaml`, `embedding.yaml`,
  `image.yaml`, `video.yaml`, `audio_tts.yaml`.
- The runtime loader (`workflows/model_spec.py`) selects the catalogue via the
  `MODEL_SPECS_ENV` env var (default `prod`; `run.py --dev-mode` sets it to `dev`).
- A YAML **template** is identified by `weights` (list of HF repo paths),
  `impl` (implementation id string), `inference_engine` (enum name: `VLLM` / `MEDIA` /
  `FORGE`), and a list of `device_model_specs`, each carrying a `device` enum name
  (`GALAXY`, `P300X2`, `T3K`, ...).
- `models-ci-config.json` is keyed by **model name** (e.g. `Llama-3.1-8B-Instruct`),
  which corresponds to `Path(weight).name` of a spec's weights. Each model has either a
  flat `{inference_engine, ci}` shape or an `implementations: [...]` array. Under
  `ci.release.devices` is the list of release-marked devices. Engine strings are
  `vLLM` / `FORGE` / `MEDIA`.

## Decisions

- **Copy granularity:** copy the **whole matched template** (all devices, all weights)
  whenever it has at least one release-marked device. No pruning of non-release devices.
- **Write mode:** **upsert** (add/update). Matched templates replace the same-identity
  template in prod, or are appended if absent. Existing prod templates that are not
  matched are left untouched (no removal of stale entries).
- **Formatting:** copy via `ruamel.yaml` round-trip so inline comments and formatting in
  the dev block survive into prod. (`ruamel.yaml` 0.18.6 is available in the venv.)
- **Normalization:** engine/device strings are normalized through
  `workflows/workflow_types.py` enums (`InferenceEngine.from_string`,
  `DeviceTypes.from_string`) so dev (`VLLM`) and ci-config (`vLLM`) spellings can't
  drift. We use `workflow_types` (a pure types module) rather than importing
  `workflows.model_spec`, which loads the catalogues as an import side effect.

## Algorithm

### Step 1 — collect release combos
Walk `models-ci-config.json`. For each model, flatten the `implementations` array (or
treat the flat object as a single implementation). Emit the set of
`(model_name, inference_engine, device)` tuples found under `ci.release.devices`.

### Step 2 — match against dev
For each of the 7 dev catalog files, ruamel-load the `templates` list. A template
**matches** a release combo when ALL of:
- some `weight` in `template["weights"]` has `Path(weight).name == model_name`, AND
- `InferenceEngine.from_string(combo_engine) == InferenceEngine.from_string(template_engine)`, AND
- `DeviceTypes.from_string(combo_device)` appears among the template's
  `device_model_specs[].device` (normalized).

A matched template is collected **whole** and tagged with its source filename.

### Step 3 — upsert into prod
For each matched dev template, open the **same-named** prod file (an `llm.yaml` match
goes to `prod/llm.yaml`, preserving category placement). Template identity for upsert is
`(impl, normalized_inference_engine, frozenset(weights))`. If prod has a template with
that identity, replace it in place (same list index); otherwise append. The dev block is
inserted as its ruamel round-trip object so comments/formatting carry over. Write the
prod file back only if it changed.

### Step 4 — report
Print a summary mapping each release combo to the template/file it matched, with a
**warning** for any combo that matched nothing in dev, and per prod file whether a
template was updated vs. appended. `--dry-run` performs steps 1–2 and reports intended
changes without writing.

## CLI

```
promote_dev_spec_to_prod.py
  --ci-config PATH   (default: .github/workflows/models-ci-config.json)
  --dev-dir   PATH   (default: workflows/model_specs/dev)
  --prod-dir  PATH   (default: workflows/model_specs/prod)
  --dry-run          (report only, write nothing)
```

Exit non-zero if any release combo matched no dev template (surfaces config drift).

## Testing (TDD)

Tests are written first, in `tests/` alongside existing `test_model_specification.py`,
using `pytest` with `tmp_path` fixtures that build small dev/prod catalogue trees and a
minimal ci-config. Coverage:

1. **Collect release combos** — flat shape and `implementations` array shape both
   parsed; only `release` (not `nightly`/`weekly`) devices collected.
2. **Match by weights basename** — `Llama-3.1-8B-Instruct` matches a template whose
   weight is `meta-llama/Llama-3.1-8B-Instruct`.
3. **Engine/device normalization** — `vLLM`↔`VLLM`, device case/alias normalized.
4. **No match** — combo with no corresponding dev template is reported/warned and
   triggers non-zero exit.
5. **Append** — a matched template absent from prod is appended.
6. **Update in place** — a matched template already present in prod (same identity,
   different content/version) replaces the existing entry; surrounding prod templates
   are untouched.
7. **Comment preservation** — an inline comment in the dev template block survives into
   the written prod file.
8. **Whole-template copy** — non-release devices in the matched template are still
   present in prod (granularity decision).
9. **Idempotency** — running twice produces no second-run changes.
10. **Dry-run** — writes nothing; prod files unchanged on disk.

## Out of scope (YAGNI)

- No pruning of non-release devices from copied templates.
- No removal of prod templates that are no longer release-marked.
- No support for promoting `nightly`/`weekly` categories (hardcoded to `release`).
- No deep schema validation beyond what matching requires.
