# Extract Model Catalog from model_spec.py to YAML — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the ~3,100-line model catalog (currently split across `llm_templates`, `vlm_templates`, …, `cnn_templates` Python lists) out of `workflows/model_spec.py` into a single YAML file: `workflows/model_specs/catalog.yaml`. Keep schema dataclasses and `ImplSpec` instances in Python. Preserve `MODEL_SPECS` and `spec_templates` as module-level constants so all 20+ consumers see no API change.

**Architecture:** Add a small YAML loader inside `workflows/model_spec.py` that builds `ModelSpecTemplate` / `DeviceModelSpec` / `SystemRequirements` / `KnownIssue` from dict data. The catalog YAML uses string names for enums (`"VLLM"`, `"T3K"`, `"EXPERIMENTAL"`) and `impl_id` strings (e.g. `"gpt_oss"`) that the loader resolves via a Python-side `_IMPL_REGISTRY`. Category boundaries (LLM, VLM, video, image, audio/TTS, embedding, CNN) are preserved as `# ====` comment dividers inside the single YAML file. Behavior is locked-in by a golden-file snapshot test: capture `export_model_specs_json(MODEL_SPECS, …)` output before any change, then assert byte-identical output after every migration step. Migration is done category-by-category for safe checkpoints, with each category appended to the same `catalog.yaml` file.

**Tech Stack:** Python 3.8, PyYAML (already a dependency, used in `workflows/workflow_venvs.py`), pytest, frozen dataclasses.

---

## File Structure

**Create:**
- `workflows/model_specs/catalog.yaml` — single file containing every template (~70 entries), with `# ====` comment dividers separating LLM / VLM / video / image / audio_tts / embedding / CNN sections
- `tests/test_model_catalog_yaml.py` — golden-file snapshot + per-template construction test
- `tests/fixtures/model_specs_golden.json` — pre-migration snapshot of `MODEL_SPECS`

**Modify:**
- `workflows/model_spec.py` — add loader functions and `_IMPL_REGISTRY` near the top; replace each per-category Python list with appends to `catalog.yaml`; finally collapse `spec_templates` to a single `load_templates_from_yaml(...)` call and delete the now-dead per-category list variables. Schema dataclasses, `ImplSpec` instances, and `MODEL_SPECS` / `spec_templates` / `get_runtime_model_spec` symbols stay.

**Leave untouched:** `workflows/workflow_types.py`, all 20+ consumer files that import from `workflows.model_spec`, the existing `benchmarking/benchmark_targets/model_performance_reference.json` flow.

**Why a single file over per-category files?** Easier to grep, no risk of misfiling a new model, fewer files to keep load-order-consistent, no per-file CI plumbing. Category structure is preserved by comment dividers; the loader doesn't care about grouping (all templates produce the same flat `spec_templates` list either way).

---

## Translation Conventions (applies to every catalog migration in Tasks 3–9)

All templates live in **one** file: `workflows/model_specs/catalog.yaml`. The file uses category comment dividers (preserved from the existing Python source layout) for browseability — they have no semantic meaning to the loader:

```yaml
# Model spec catalog. Edit this file to add or modify a model spec.
# Schema: workflows/model_spec.py (ModelSpecTemplate, DeviceModelSpec).
# Categories below are separated by comment dividers for human navigation;
# the loader treats every entry uniformly.
templates:
  # =============================================================================
  # LLM templates
  # =============================================================================
  - weights: [...]
    impl: <impl_id>
    ...
  - weights: [...]
    ...

  # =============================================================================
  # VLM templates
  # =============================================================================
  - weights: [...]
    ...
```

Each task in 3–9 **appends** its category section to the bottom of `catalog.yaml` (with the divider comment above it). After the final category lands, Task 10 collapses the per-category Python lists.

**Field translations** (apply mechanically to every Python entry):

| Python | YAML |
|---|---|
| `impl=tt_transformers_impl` | `impl: tt_transformers` (use the `impl_id` string from `_IMPL_REGISTRY`, not the Python variable name) |
| `inference_engine=InferenceEngine.VLLM.value` | `inference_engine: VLLM` |
| `device=DeviceTypes.T3K` | `device: T3K` |
| `model_type=ModelType.VLM` | `model_type: VLM` |
| `status=ModelStatusTypes.FUNCTIONAL` | `status: FUNCTIONAL` |
| `mode=VersionMode.STRICT` | `mode: STRICT` |
| `workflow_type=WorkflowType.EVALS` | `workflow_type: EVALS` |
| `16 * 1024` | `16384  # 16 * 1024` (pre-compute the int, add a comment so reviewers still see the intent) |
| `32 * 4` | `128  # 32 * 4` |
| `json.dumps({"image": 1})` | `'{"image": 1}'` (pre-encode as a JSON string literal — these values are passed directly to vLLM CLI which expects JSON strings) |
| `json.dumps({"temperature": 0.5, "top_k": 50, "top_p": 0.95})` | `'{"temperature": 0.5, "top_k": 50, "top_p": 0.95}'` |
| `default_impl=True` | `default_impl: true` |
| `default_impl=False` | `default_impl: false` |
| `tensor_cache_timeout=5400.0` | `tensor_cache_timeout: 5400.0` |
| Python comments above an entry | YAML `#` comments above the same entry |

**Quoting rules** (prevent YAML type coercion surprises):
- Version strings like `"0.10.0"` MUST be quoted, or YAML may parse `0.10` as a float.
- Bare `1`, `4`, `32` stay as int (matches existing Python ints in `env_vars`).
- `"1"` in Python stays as quoted string `"1"` in YAML.
- Strings containing `:` or `{` MUST be quoted (`MESH_DEVICE` value, JSON-encoded args).
- Commit SHAs that happen to be all-numeric — quote them to keep as strings.

**Preserve everything**: do NOT skip fields. Every `tensor_cache_timeout`, `system_requirements`, `override_tt_config`, `metadata`, `has_builtin_warmup`, `supported_modalities`, `repacked`, `min_disk_gb`, `min_ram_gb`, `hf_weights_repo`, `docker_image` value in the source list must appear in the YAML output.

**Source location for each category:** see the per-task line ranges in Tasks 3–9. Convert every entry in the source list, in order.

**Example** (one fully-translated entry, for reference while converting):

Source Python (`workflows/model_spec.py` ~line 1010, the first `llm_templates` entry):

```python
ModelSpecTemplate(
    weights=["openai/gpt-oss-20b"],
    impl=gpt_oss_impl,
    version="0.10.0",
    tt_metal_commit="e867533",
    vllm_commit="8f36910",
    inference_engine=InferenceEngine.VLLM.value,
    device_model_specs=[
        DeviceModelSpec(
            device=DeviceTypes.T3K,
            max_concurrency=1,
            max_context=16 * 1024,
            default_impl=True,
            env_vars={
                "VLLM_ENABLE_RESPONSES_API_STORE": 1,
                "VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS": 1,
            },
        ),
        DeviceModelSpec(
            device=DeviceTypes.GALAXY,
            max_concurrency=32 * 4,
            max_context=128 * 1024,
            default_impl=True,
            env_vars={
                "MESH_DEVICE": "(4, 8)",  # Override default TG->(8,4) to use (4,8) mesh grid
            },
            vllm_args={
                "data_parallel_size": 4,
            },
        ),
    ],
    status=ModelStatusTypes.EXPERIMENTAL,
    has_builtin_warmup=True,
    env_vars={
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    },
    metadata={
        "openai/gpt-oss-20b": {
            "reasoning_parser_name": "openai_gptoss",
            "tool_call_parser_name": "openai",
        },
    },
),
```

Equivalent YAML:

```yaml
- weights:
    - openai/gpt-oss-20b
  impl: gpt_oss
  version: "0.10.0"
  tt_metal_commit: e867533
  vllm_commit: 8f36910
  inference_engine: VLLM
  device_model_specs:
    - device: T3K
      max_concurrency: 1
      max_context: 16384  # 16 * 1024
      default_impl: true
      env_vars:
        VLLM_ENABLE_RESPONSES_API_STORE: 1
        VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS: 1
    - device: GALAXY
      max_concurrency: 128  # 32 * 4
      max_context: 131072  # 128 * 1024
      default_impl: true
      env_vars:
        MESH_DEVICE: "(4, 8)"  # Override default TG->(8,4) to use (4,8) mesh grid
      vllm_args:
        data_parallel_size: 4
  status: EXPERIMENTAL
  has_builtin_warmup: true
  env_vars:
    VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"
  metadata:
    openai/gpt-oss-20b:
      reasoning_parser_name: openai_gptoss
      tool_call_parser_name: openai
```

---

### Task 1: Capture pre-migration golden snapshot

Lock in the current behavior so every later step can verify byte-identical output. We use the existing `export_model_specs_json` helper.

**Files:**
- Create: `tests/fixtures/model_specs_golden.json`
- Create: `tests/test_model_catalog_yaml.py`

- [ ] **Step 1: Generate the golden snapshot**

Run from repo root:

```bash
python -c "from pathlib import Path; from workflows.model_spec import MODEL_SPECS, export_model_specs_json; Path('tests/fixtures').mkdir(parents=True, exist_ok=True); n = export_model_specs_json(MODEL_SPECS, Path('tests/fixtures/model_specs_golden.json')); print(f'Wrote {n} specs')"
```

Expected: prints `Wrote N specs` where N matches the current catalog count (~70+).

- [ ] **Step 2: Write the golden-file snapshot test**

Create `tests/test_model_catalog_yaml.py` with this content:

```python
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
from pathlib import Path

from workflows.model_spec import MODEL_SPECS, export_model_specs_json

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "model_specs_golden.json"


def _current_export(tmp_path: Path) -> dict:
    out = tmp_path / "current.json"
    export_model_specs_json(MODEL_SPECS, out)
    return json.loads(out.read_text())


def test_model_specs_match_golden(tmp_path):
    """MODEL_SPECS output must be byte-identical to the pre-migration snapshot."""
    assert GOLDEN_PATH.exists(), (
        "Golden snapshot missing. Regenerate with: python -c \"...\" "
        "(see plan task 1 step 1)."
    )
    golden = json.loads(GOLDEN_PATH.read_text())
    current = _current_export(tmp_path)
    # Compare only the catalog payload; schema_version / release_version
    # can drift independently of catalog content.
    assert current["model_specs"] == golden["model_specs"]
```

- [ ] **Step 3: Run the test to verify it passes against the unmigrated catalog**

Run: `pytest tests/test_model_catalog_yaml.py -v`
Expected: PASS (golden equals current because we just generated it from current).

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/model_specs_golden.json tests/test_model_catalog_yaml.py
git commit -m "test: add golden snapshot of MODEL_SPECS catalog

Locks in current ModelSpec output so the upcoming YAML migration
can verify byte-identical behavior at every step."
```

---

### Task 2: Add YAML loader and `_IMPL_REGISTRY`, with unit tests covering every schema construct

Build the load-from-dict machinery in isolation. No catalog file exists yet — the loader is exercised only by unit tests against in-memory dicts.

**Files:**
- Modify: `workflows/model_spec.py` (insert after the `ImplSpec` instances block, around current line 267)
- Modify: `tests/test_model_catalog_yaml.py` (add loader unit tests)

- [ ] **Step 1: Write failing loader unit tests**

Append to `tests/test_model_catalog_yaml.py`:

```python
from workflows.model_spec import (
    DeviceModelSpec,
    ImplSpec,
    KnownIssue,
    ModelSpecTemplate,
    SystemRequirements,
    VersionRequirement,
    _IMPL_REGISTRY,
    _build_device_model_spec,
    _build_system_requirements,
    _build_template,
    load_templates_from_yaml,
    tt_transformers_impl,
)
from workflows.workflow_types import (
    DeviceTypes,
    InferenceEngine,
    ModelStatusTypes,
    ModelType,
    VersionMode,
    WorkflowType,
)


def test_impl_registry_is_populated():
    """Every ImplSpec instance defined at module scope must be in _IMPL_REGISTRY."""
    assert _IMPL_REGISTRY["tt_transformers"] is tt_transformers_impl
    # impl_id of each registry entry must match its key
    for impl_id, impl in _IMPL_REGISTRY.items():
        assert impl.impl_id == impl_id


def test_build_system_requirements_full():
    out = _build_system_requirements({
        "firmware": {"specifier": ">=19.2.0", "mode": "STRICT"},
        "kmd": {"specifier": ">=2.5.0", "mode": "SUGGESTED"},
    })
    assert isinstance(out, SystemRequirements)
    assert out.firmware == VersionRequirement(specifier=">=19.2.0", mode=VersionMode.STRICT)
    assert out.kmd == VersionRequirement(specifier=">=2.5.0", mode=VersionMode.SUGGESTED)


def test_build_system_requirements_none_returns_none():
    assert _build_system_requirements(None) is None


def test_build_device_model_spec_with_known_issues_and_overrides():
    spec = _build_device_model_spec({
        "device": "T3K",
        "max_concurrency": 32,
        "max_context": 32768,
        "default_impl": True,
        "vllm_args": {"data_parallel_size": 4, "limit-mm-per-prompt": '{"image": 1}'},
        "override_tt_config": {"trace_region_size": 90000000},
        "env_vars": {"TT_MM_THROTTLE_PERF": 5, "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
        "known_issues": [
            {"workflow_type": "EVALS", "reason": "broken on this device", "task_name": "ifeval"},
        ],
    })
    assert isinstance(spec, DeviceModelSpec)
    assert spec.device == DeviceTypes.T3K
    assert spec.vllm_args["data_parallel_size"] == 4
    assert spec.override_tt_config["trace_region_size"] == 90000000
    assert spec.known_issues == [
        KnownIssue(workflow_type=WorkflowType.EVALS, reason="broken on this device", task_name="ifeval"),
    ]


def test_build_template_resolves_all_enum_and_impl_references():
    template = _build_template({
        "weights": ["Qwen/Qwen3-8B"],
        "impl": "tt_transformers",
        "version": "0.10.0",
        "tt_metal_commit": "abc1234",
        "vllm_commit": "def5678",
        "inference_engine": "VLLM",
        "device_model_specs": [
            {"device": "N150", "max_concurrency": 32, "max_context": 32768, "default_impl": True},
        ],
        "status": "FUNCTIONAL",
        "model_type": "LLM",
        "supported_modalities": ["text"],
        "env_vars": {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
        "metadata": {"Qwen/Qwen3-8B": {"reasoning_parser_name": "qwen3"}},
    })
    assert isinstance(template, ModelSpecTemplate)
    assert template.impl is _IMPL_REGISTRY["tt_transformers"]
    assert template.inference_engine == InferenceEngine.VLLM.value
    assert template.status == ModelStatusTypes.FUNCTIONAL
    assert template.model_type == ModelType.LLM
    assert template.device_model_specs[0].device == DeviceTypes.N150


def test_load_templates_from_yaml_roundtrip(tmp_path):
    yaml_path = tmp_path / "tiny.yaml"
    yaml_path.write_text(
        """
templates:
  - weights: [Qwen/Qwen3-8B]
    impl: tt_transformers
    version: "0.10.0"
    tt_metal_commit: abc1234
    vllm_commit: def5678
    inference_engine: VLLM
    device_model_specs:
      - device: N150
        max_concurrency: 32
        max_context: 32768
        default_impl: true
    status: FUNCTIONAL
""".strip()
    )
    templates = load_templates_from_yaml(yaml_path)
    assert len(templates) == 1
    assert templates[0].weights == ["Qwen/Qwen3-8B"]
    assert templates[0].impl is _IMPL_REGISTRY["tt_transformers"]
```

- [ ] **Step 2: Run the new tests to confirm they fail (loader not implemented yet)**

Run: `pytest tests/test_model_catalog_yaml.py -v`
Expected: New tests FAIL with `ImportError: cannot import name '_IMPL_REGISTRY' …`. The original `test_model_specs_match_golden` still PASSES.

- [ ] **Step 3: Implement the loader in `workflows/model_spec.py`**

Add `import yaml` near the top of the imports block (alongside `import json`):

```python
import yaml
```

After the last `ImplSpec` instance (`tt_vllm_plugin_impl = ImplSpec(...)` around current line 266), insert:

```python
_IMPL_REGISTRY: Dict[str, ImplSpec] = {
    "tt_transformers": tt_transformers_impl,
    "llama3_70b_galaxy": llama3_70b_galaxy_impl,
    "qwen3_32b_galaxy": qwen3_32b_galaxy_impl,
    "gpt_oss": gpt_oss_impl,
    "deepseek_r1_galaxy": deepseek_r1_galaxy_impl,
    "whisper": whisper_impl,
    "speecht5_tts": speecht5_impl,
    "forge_vllm_plugin": forge_vllm_plugin_impl,
    "tt_vllm_plugin": tt_vllm_plugin_impl,
}


def _build_system_requirements(data: Optional[Dict]) -> Optional[SystemRequirements]:
    if data is None:
        return None
    kwargs: Dict = {}
    for key in ("firmware", "kmd"):
        if data.get(key) is not None:
            kwargs[key] = VersionRequirement(
                specifier=data[key]["specifier"],
                mode=VersionMode[data[key]["mode"]],
            )
    return SystemRequirements(**kwargs)


def _build_device_model_spec(data: Dict) -> DeviceModelSpec:
    kwargs = dict(data)
    kwargs["device"] = DeviceTypes.from_string(kwargs["device"])
    if "system_requirements" in kwargs:
        kwargs["system_requirements"] = _build_system_requirements(
            kwargs["system_requirements"]
        )
    if "known_issues" in kwargs:
        kwargs["known_issues"] = [
            KnownIssue(
                workflow_type=WorkflowType.from_string(ki["workflow_type"]),
                reason=ki["reason"],
                task_name=ki.get("task_name"),
            )
            for ki in kwargs["known_issues"]
        ]
    return DeviceModelSpec(**kwargs)


def _build_template(data: Dict) -> ModelSpecTemplate:
    kwargs = dict(data)
    impl_id = kwargs["impl"]
    if impl_id not in _IMPL_REGISTRY:
        raise ValueError(
            f"Unknown impl '{impl_id}'. Known impls: {sorted(_IMPL_REGISTRY)}"
        )
    kwargs["impl"] = _IMPL_REGISTRY[impl_id]
    kwargs["inference_engine"] = InferenceEngine[kwargs["inference_engine"]].value
    kwargs["device_model_specs"] = [
        _build_device_model_spec(d) for d in kwargs["device_model_specs"]
    ]
    if "system_requirements" in kwargs:
        kwargs["system_requirements"] = _build_system_requirements(
            kwargs["system_requirements"]
        )
    if "model_type" in kwargs and kwargs["model_type"] is not None:
        kwargs["model_type"] = ModelType[kwargs["model_type"]]
    if "status" in kwargs:
        kwargs["status"] = ModelStatusTypes[kwargs["status"]]
    return ModelSpecTemplate(**kwargs)


def load_templates_from_yaml(path: Path) -> List[ModelSpecTemplate]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "templates" not in data:
        raise ValueError(f"YAML file {path} is empty or missing 'templates' key")
    return [_build_template(t) for t in data["templates"]]


_MODEL_SPECS_DIR = get_repo_root_path() / "workflows" / "model_specs"
```

- [ ] **Step 4: Run all tests to confirm they pass**

Run: `pytest tests/test_model_catalog_yaml.py tests/test_model_specification.py -v`
Expected: All new tests PASS. All pre-existing tests still PASS. Golden test still PASSES (loader is added but catalog is unchanged).

- [ ] **Step 5: Commit**

```bash
git add workflows/model_spec.py tests/test_model_catalog_yaml.py
git commit -m "feat: add YAML loader for ModelSpecTemplate catalog

Adds _IMPL_REGISTRY plus _build_template/_build_device_model_spec/
_build_system_requirements/load_templates_from_yaml so future commits
can move the catalog into YAML files. Behavior unchanged."
```

---

### Task 3: Migrate `llm_templates` to `workflows/model_specs/catalog.yaml`

The largest section (~1,400 lines of templates, lines ~1010–2427 in current `model_spec.py`). This task **creates** the single catalog YAML file; later tasks append to it.

**Files:**
- Create: `workflows/model_specs/catalog.yaml`
- Modify: `workflows/model_spec.py` (replace `llm_templates = [...]` with a YAML load that returns the full catalog so far; remove `*llm_templates` from `spec_templates` and use the YAML load there instead)

- [ ] **Step 1: Create `workflows/model_specs/catalog.yaml` with the LLM section**

Follow the **Translation Conventions** section at the top of this plan.

Start the file with the header comment block and a `templates:` key, then write the LLM section divider and convert every entry from `llm_templates` (Python source location: between `llm_templates = [` at line ~1010 and the closing `]` before `# vlm_templates` at line ~2428).

```yaml
# Model spec catalog. Edit this file to add or modify a model spec.
# Schema: workflows/model_spec.py (ModelSpecTemplate, DeviceModelSpec).
# Categories below are separated by comment dividers for human navigation;
# the loader treats every entry uniformly.
templates:
  # =============================================================================
  # LLM templates
  # =============================================================================
  - weights:
      - openai/gpt-oss-20b
    impl: gpt_oss
    # ... (continue with every llm_templates entry in order) ...
```

Preserve all in-line comments from the Python source.

- [ ] **Step 2: Replace the Python `llm_templates` list and update `spec_templates`**

In `workflows/model_spec.py`:

a) Delete the entire `llm_templates = [ ... ]` block (lines ~1010 through the closing `]` before the `# vlm_templates` section header). Keep the `# llm_templates` comment divider above it for now (it documents intent).

b) Replace the deleted block with:

```python
# Single source of truth for the catalog. Categories below remain in
# Python lists until Task 10 collapses them; spec_templates loads from
# catalog.yaml for everything that has already migrated.
_catalog_yaml_templates = load_templates_from_yaml(_MODEL_SPECS_DIR / "catalog.yaml")
```

c) Update the `spec_templates` expression near the end of the file (current lines ~4015–4023). The current code is:

```python
spec_templates = [
    *llm_templates,
    *vlm_templates,
    *video_templates,
    *image_templates,
    *audio_tts_templates,
    *embedding_templates,
    *cnn_templates,
]
```

Replace it with:

```python
spec_templates = [
    *_catalog_yaml_templates,
    *vlm_templates,
    *video_templates,
    *image_templates,
    *audio_tts_templates,
    *embedding_templates,
    *cnn_templates,
]
```

(`_catalog_yaml_templates` now plays the role `llm_templates` previously did; the rest of the categories continue to load from their Python literals until their tasks land.)

- [ ] **Step 3: Run the golden snapshot test**

Run: `pytest tests/test_model_catalog_yaml.py::test_model_specs_match_golden -v`
Expected: PASS — `MODEL_SPECS` output byte-identical to the pre-migration snapshot.

If FAIL: inspect the diff (`pytest -vv` shows the first mismatched dict key). Most common causes:
- Forgot to quote a version string and YAML coerced it to float.
- Forgot to pre-encode a `json.dumps(...)` value.
- Typo in a `tt_metal_commit` SHA.
- Missed an entry, duplicated one, or reordered entries (order matters for `MODEL_SPECS` dict insertion).
Fix the YAML and re-run.

- [ ] **Step 4: Run the full test suite for regressions**

Run: `pytest tests/test_model_specification.py tests/test_model_catalog_yaml.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add workflows/model_specs/catalog.yaml workflows/model_spec.py
git commit -m "refactor: move llm_templates catalog to YAML

Extracts ~1400 lines of LLM model templates from workflows/model_spec.py
into workflows/model_specs/catalog.yaml (the single catalog file other
categories will be appended to in following commits). Loader resolves
enum and impl_id references at import time. MODEL_SPECS output verified
byte-identical to pre-migration snapshot."
```

---

### Tasks 4–9: Migrate remaining categories into `workflows/model_specs/catalog.yaml`

Each of these tasks **appends** one category's section to the bottom of the existing `catalog.yaml` and prunes the corresponding Python list. The five steps are identical per category — only the category name and source-line range differ. The per-task table below gives the exact substitutions for steps 1, 2c, and 5.

**Per-category parameters:**

| Task | Category | Source lines (current `model_spec.py`) | Python variable | YAML divider title |
|---|---|---|---|---|
| 4 | VLM | ~2431–2772 | `vlm_templates` | `# VLM templates` |
| 5 | Video | ~2775–2934 | `video_templates` | `# Video templates` |
| 6 | Image | ~2937–3208 | `image_templates` | `# Image templates` |
| 7 | Audio/TTS | ~3211–3356 | `audio_tts_templates` | `# Audio / TTS templates` |
| 8 | Embedding | ~3359–3631 | `embedding_templates` | `# Embedding templates` |
| 9 | CNN | ~3634–end of catalog (before `spec_templates = [...]`) | `cnn_templates` | `# CNN templates` |

**Each task uses these five steps. Substitute the task's row from the table above for `<CATEGORY>`, `<SOURCE_LINES>`, `<PY_VARIABLE>`, `<YAML_DIVIDER_TITLE>`.**

**Files** (every task):
- Modify: `workflows/model_specs/catalog.yaml` (append a new section at the bottom)
- Modify: `workflows/model_spec.py` (delete the `<PY_VARIABLE> = [...]` block; remove `*<PY_VARIABLE>` from the `spec_templates` expression)

- [ ] **Step 1: Append the category section to `catalog.yaml`**

Open `workflows/model_specs/catalog.yaml`. At the bottom of the `templates:` list, append:

```yaml

  # =============================================================================
  <YAML_DIVIDER_TITLE>
  # =============================================================================
  - weights: [...]
    # ... convert every entry from <PY_VARIABLE>, in order ...
```

Source location for the entries: lines `<SOURCE_LINES>` of `workflows/model_spec.py`. Follow the **Translation Conventions** section at the top of this plan. Preserve all in-line comments.

- [ ] **Step 2: Delete the Python list and update `spec_templates`**

In `workflows/model_spec.py`:

a) Delete the entire `<PY_VARIABLE> = [ ... ]` block (the lines listed in `<SOURCE_LINES>`).

b) Keep the `# <PY_VARIABLE>` comment-divider line above the deleted block — it now documents what category lives in `catalog.yaml`.

c) Edit the `spec_templates` expression (near the end of the file) to remove `*<PY_VARIABLE>` from the list. After Task 4 (VLM), the expression should look like:

```python
spec_templates = [
    *_catalog_yaml_templates,
    *video_templates,
    *image_templates,
    *audio_tts_templates,
    *embedding_templates,
    *cnn_templates,
]
```

After each subsequent task, one more `*<name>_templates` entry is removed. After Task 9 (CNN), the expression is:

```python
spec_templates = [
    *_catalog_yaml_templates,
]
```

(Task 10 collapses this further.)

- [ ] **Step 3: Run the golden snapshot test**

Run: `pytest tests/test_model_catalog_yaml.py::test_model_specs_match_golden -v`
Expected: PASS — `MODEL_SPECS` output byte-identical to the pre-migration snapshot.

If FAIL: see the troubleshooting checklist in Task 3 Step 3.

- [ ] **Step 4: Run the regression suite**

Run: `pytest tests/test_model_specification.py tests/test_model_catalog_yaml.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add workflows/model_specs/catalog.yaml workflows/model_spec.py
git commit -m "refactor: move <PY_VARIABLE> into catalog.yaml"
```

(Substitute the actual variable name. Example for Task 4: `"refactor: move vlm_templates into catalog.yaml"`.)

---

### Task 10: Collapse `spec_templates` to a single load and remove the `_catalog_yaml_templates` alias

After Task 9, every category lives in `catalog.yaml`, and `spec_templates` is just `[*_catalog_yaml_templates]`. Time to simplify.

**Files:**
- Modify: `workflows/model_spec.py`

- [ ] **Step 1: Collapse `spec_templates` to a direct load**

Find the line:

```python
_catalog_yaml_templates = load_templates_from_yaml(_MODEL_SPECS_DIR / "catalog.yaml")
```

…and the trailing `spec_templates` expression:

```python
spec_templates = [
    *_catalog_yaml_templates,
]
```

Replace both with a single direct assignment at the point where `spec_templates` is currently defined (near the end of the file, before `def get_model_spec_map(...)`):

```python
spec_templates: List[ModelSpecTemplate] = load_templates_from_yaml(
    _MODEL_SPECS_DIR / "catalog.yaml"
)
```

Delete the now-unused `_catalog_yaml_templates` line entirely.

- [ ] **Step 2: Remove the now-orphaned per-category comment dividers from `workflows/model_spec.py`**

The `# llm_templates`, `# vlm_templates`, etc. comment-divider blocks (`# =============================================================================` lines) in `workflows/model_spec.py` no longer document anything in that file. Delete those comment blocks. The matching dividers in `catalog.yaml` provide the navigation now.

- [ ] **Step 3: Run the golden snapshot and full test suite**

Run: `pytest tests/test_model_catalog_yaml.py tests/test_model_specification.py -v`
Expected: All PASS.

Run: `pytest tests/ -x -q`
Expected: All PASS (or only pre-existing failures unrelated to this refactor — confirm by comparing to `git stash && pytest tests/ -x -q` if uncertain).

- [ ] **Step 4: Commit**

```bash
git add workflows/model_spec.py
git commit -m "refactor: collapse spec_templates to single catalog.yaml load

All template categories now live in workflows/model_specs/catalog.yaml.
workflows/model_spec.py is schema + loader + ImplSpec registry only."
```

---

### Task 11: Add catalog-construction CI test

The golden-file test proves output equivalence but lives at the level of serialized output. This task adds a direct check that every template in `catalog.yaml` constructs and expands cleanly — the safety net for catalog edits that pass YAML parsing but produce no specs or fail `__post_init__`.

**Files:**
- Modify: `tests/test_model_catalog_yaml.py`

- [ ] **Step 1: Add the catalog construction test**

Append to `tests/test_model_catalog_yaml.py`:

```python
CATALOG_YAML = (
    Path(__file__).resolve().parent.parent / "workflows" / "model_specs" / "catalog.yaml"
)


def test_catalog_yaml_loads_and_every_template_expands():
    """catalog.yaml must load and every template must expand to >=1 spec.

    Surfaces typos and missing-field errors with a per-template assertion
    message instead of one opaque import-time exception.
    """
    templates = load_templates_from_yaml(CATALOG_YAML)
    assert templates, "catalog.yaml produced zero templates"
    for t in templates:
        specs = t.expand_to_specs()
        assert specs, f"template {t.weights} expanded to zero specs"
```

- [ ] **Step 2: Run the new test**

Run: `pytest tests/test_model_catalog_yaml.py -v`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_model_catalog_yaml.py
git commit -m "test: add catalog.yaml construction check

Surfaces per-template errors when catalog.yaml edits fail to construct
or expand cleanly."
```

---

### Task 12: Decommission the golden fixture (optional but recommended)

The golden fixture has done its job and is large (~1 MB JSON). After all migrations land and pass on `main`, the catalog-construction test + the existing `tests/test_model_specification.py` invariants provide ongoing coverage. Keeping the golden fixture around forever means every legitimate catalog change (adding a model, bumping a commit) requires regenerating it, which is noisy and risks rubber-stamping diffs.

This task is OPTIONAL. Skip it if the team wants the snapshot as ongoing belt-and-suspenders coverage.

**Files:**
- Delete: `tests/fixtures/model_specs_golden.json`
- Modify: `tests/test_model_catalog_yaml.py` (remove the golden test and its imports)

- [ ] **Step 1: Remove the golden test and fixture**

In `tests/test_model_catalog_yaml.py`, delete `GOLDEN_PATH`, `_current_export`, and `test_model_specs_match_golden`.

Run: `git rm tests/fixtures/model_specs_golden.json`

- [ ] **Step 2: Run remaining tests**

Run: `pytest tests/test_model_catalog_yaml.py -v`
Expected: PASS (only the loader unit tests and catalog construction test remain).

- [ ] **Step 3: Commit**

```bash
git add tests/test_model_catalog_yaml.py tests/fixtures/model_specs_golden.json
git commit -m "test: remove model_specs golden snapshot

The golden fixture served its purpose during the catalog YAML migration.
Catalog construction test and test_model_specification.py provide
ongoing coverage without requiring fixture regeneration on every
legitimate catalog change."
```
