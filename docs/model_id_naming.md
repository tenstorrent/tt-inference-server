# Model id naming contract

Since the model id became the full HF repo id (`Qwen/Qwen3-32B` rather than
`Qwen3-32B`), every name derived from a model has to escape the org separator,
and everything that reads such a name has to undo the escape.

The producer and the consumer live in different repositories — **tt-shield**
builds the GitHub artifact and job names, **tt-inference-server** parses them —
so this is a cross-repo contract. Both directions live in one place:

**[`utils/model_naming.py`](../utils/model_naming.py)** — stdlib only, imports
nothing else from this repo. Tests: [`tests/test_model_naming.py`](../tests/test_model_naming.py).

## The two representations

A model has two representations and they must not be confused.

| | example | used for |
|---|---|---|
| **data identity** | `Qwen/Qwen3-32B` | `models-ci-config.json` keys, report `metadata`, HTTP `model` params, performance-target lookup, DB columns. **Never escaped.** |
| **name token** | `Qwen__Qwen3-32B` | filenames, directory names, GitHub artifact names, CI job names. **Always escaped.** |

The org prefix is **escaped, never stripped**, so a token stays unique
(`a/model` and `b/model` do not collide) and round-trips exactly.

`__` rather than `_` is what makes the reversal exact. Model ids already contain
single underscores (`microsoft/phi-1_5`, `yolox_nano`), so a single-underscore
separator is ambiguous and cannot be undone — which is how the two sides drifted
apart in the first place.

## API

```python
from utils.model_naming import slugify_model_id, unslugify_model_id

slugify_model_id("Qwen/Qwen3-32B")     # "Qwen__Qwen3-32B"
unslugify_model_id("Qwen__Qwen3-32B")  # "Qwen/Qwen3-32B"
```

| function | purpose |
|---|---|
| `slugify_model_id(model_id)` | identity → token |
| `unslugify_model_id(slug)` | token → identity |
| `slugify_name_parts(*parts)` | `_`-join then escape; composite names (report ids, block ids) |
| `model_name_variants(model_id)` | every token a producer may plausibly have used, for tolerant reading — see [Transition](#transition) |
| `is_artifact_name_safe(name)` | assert on the producing side, before `upload-artifact` rejects it |
| `workflow_logs_artifact_prefix(workflow, model_id)` | `workflow_logs_<workflow>_<model>_` |
| `split_workflow_logs_artifact_name(name, workflow, model_id)` | → `(runner, suffix)` or `None` |
| `ci_job_name(workflow, model_id, runner_label, runner_type)` | `run-<workflow>-<model>-<label>-<type>` |
| `device_from_ci_job_name(name, workflow, model_id, runner_label)` | → device or `None` |

### From shell

The module is also a CLI, so a producer that builds names in YAML/bash needs no
`pip install` and no `PYTHONPATH`:

```bash
SLUG=$(python tt-inference-server/utils/model_naming.py slugify "$MODEL")
# also: unslugify <slug> | artifact-prefix <workflow> <model> | job-name <workflow> <model> <label> <type>
```

## Which side does what

**Producing a name** — escape once, at the point the name is built:

```python
artifact = workflow_logs_artifact_prefix("release", model) + device + ".zip"
```

**Reading an identity back** — prefer *data* over *names*. A report carries its
own identity in `metadata.model_repo` / `metadata.model_id`; read that rather
than reversing a filename. Reverse a name only when the name is all you have
(an artifact listing, a directory scan), and then use the tolerant helpers.

## Transition

Not every producer has adopted this contract yet, so the readers accept the
older forms too. `model_name_variants("Qwen/Qwen3-32B")` yields, most- to
least-canonical:

| token | origin |
|---|---|
| `Qwen__Qwen3-32B` | canonical |
| `Qwen/Qwen3-32B` | unescaped. Impossible in an artifact name (GitHub rejects `/`); **does** occur in job names, which allow it |
| `Qwen_Qwen3-32B` | tt-shield's single-underscore shell step, predating this contract |
| `Qwen3-32B` | the bare model id, predating the HF prefix |

The last one is the only ambiguous entry — two orgs can share a basename — so it
is tried last. No two models in `models-ci-config.json` currently collide, and
`tests/test_model_naming.py` checks the whole config round-trips.

One consequence worth knowing: a caller prefix must **not** be stripped by
splitting a job name on `/`. The GitHub jobs API returns
`"caller job / run-release-…"`, but an unescaped model id also contains `/`, so
splitting eats the org prefix along with the caller prefix and loses the device.
`device_from_ci_job_name` searches for the marker instead.

## tt-shield side (not yet applied)

As of the tt-shield checkout inspected on 2026-07-28, the producing side is
inconsistent: one artifact name is escaped with a single `_`, another is not
escaped at all. Steps are named below rather than only line-numbered, since
line numbers drift.

### 1. Emit the token from the matrix generator (preferred)

GitHub expressions have no `replace()`, so a token cannot be derived inside a
`${{ }}`. Emit it alongside the identity instead, and every consumer downstream
just interpolates it.

`.github/workflows/dynamic-workflow.yml`, job `generate-matrix`, step *"Fetch
model config files from tt-inference-server"* already pulls the config by
`gh api`, pinned to `inputs.inference-server-sha`. Pull the contract the same
way, at the same ref — no vendored copy, no drift:

```yaml
          gh api repos/tenstorrent/tt-inference-server/contents/utils/model_naming.py?ref=${{ inputs.inference-server-sha }} \
            -H "Accept: application/vnd.github.v3.raw" > "$DEST/model_naming.py"
```

`generate_ci_matrix.py` runs with that directory as its cwd, so it imports
directly:

```python
from model_naming import slugify_model_id

config = {
    "model": model_name,                        # identity — verbatim, unchanged
    "model_slug": slugify_model_id(model_name), # name token — new
    ...
}
```

### 2. Use the token in every name, the identity in every payload

`.github/workflows/workflow_run-tests-with-inference-server.yml`:

| step / field | current | change to |
|---|---|---|
| `⬆️ Upload workflow logs`, `name:` | `workflow_logs_…_${{ matrix.config.model }}_…` | `${{ matrix.config.model_slug }}` — **this is the actual breakage**: `upload-artifact` rejects `/` in a name |
| job `name:` (`run-<workflow>-<model>-<label>-<type>`) | `${{ matrix.config.model }}` | `${{ matrix.config.model_slug }}` |
| step *"Sanitize model name for artifacts"* (`${SANITIZED_MODEL//\//_}`) | single `_` | delete the step; use `${{ matrix.config.model_slug }}` in the `report_…` artifact name |
| `--model` argument to `run.py`, `generate-empty-report.sh`, report payloads | `${{ matrix.config.model }}` | **unchanged** — these take the identity |

If a name has to be built somewhere the matrix isn't reachable, call the CLI in
that step instead (tt-inference-server is already checked out at
`tt-inference-server/` in this workflow):

```yaml
      - id: slug
        run: |
          echo "model=$(python tt-inference-server/utils/model_naming.py \
            slugify '${{ matrix.config.model }}')" >> "$GITHUB_OUTPUT"
```

The reader tolerates the old forms during the transition, so these changes can
land independently of a tt-inference-server release.

## Where the prefix does and does not appear

Two different identifiers are easy to confuse, and only one of them ever
carries an org prefix.

| field | value | prefix? |
|---|---|---|
| `ModelSpec.hf_model_repo` → report `metadata.model_repo` | `Qwen/Qwen3-32B` | **yes** — this is the identity |
| `ModelSpec.model_id` → report `metadata.model_id` | `id_vllm_Qwen3-32B_p150` | **no, ever** |

`model_id` is `f"id_{impl_name}_{model_name}_{device}"` and `model_name` is
`Path(hf_model_repo).name` — the *basename*. So everything derived from
`model_id` is prefix-free by construction and contains no `/`: the `MODEL_SPECS`
key, `eval_<model_id>/` directories, `runtime_model_spec_*_<model_id>_*.json`,
and the media `report_id`. That is a deliberate design, not a gap — but it means
`hf_model_repo` is the only field to read when the org matters, and a reader
must not treat `model_id` as an escaped repo id.

## Known gaps

* **Eight models have no repo id at all.** The Forge CNN specs in
  `workflows/model_specs/{dev,prod}/cnn.yaml` carry bare `weights` —
  `resnet-50`, `vovnet`, `mobilenetv2`, `efficientnet`, `segformer`, `vit`,
  `unet`, `yolox_nano` — so their `hf_model_repo` is not an HF repo id and
  `models-ci-config.json` keys them bare, while all 45 other models are
  `org/name`. Giving them the prefix is a spec-data change (those YAMLs are
  generated by the release process) plus the matching config keys, and needs the
  real repo per model — several may not have a canonical one. Nothing in the
  code blocks it: `slugify_model_id` leaves a bare name untouched and escapes a
  prefixed one, so both forms work today and will keep working after.
* **Single-`_` escapes left alone**, because they name directories and files
  that already exist on disk: `test_module/eval_tests/whisper_eval_test.py`
  (HF cache dirs), `llm_module/drivers/aiperf_*.py`,
  `llm_module/prefix_cache/scenarios.py`, `test_fixtures/server_helper.py`,
  `test_module/load_param_tests/server_helper.py`.
