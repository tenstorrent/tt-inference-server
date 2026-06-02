# Promote dev spec to prod — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/release/promote_dev_spec_to_prod.py`, which copies every `release`-marked model-device spec from the dev catalogue into the prod catalogue, preserving YAML comments/formatting.

**Architecture:** Read release combos from `models-ci-config.json`, scan `workflows/model_specs/dev/*.yaml` for matching templates (by weights-basename + inference_engine + device), and upsert each whole matched template into the same-named `workflows/model_specs/prod/*.yaml` file by `(impl, engine, weights)` identity. YAML is round-tripped with `ruamel.yaml` so inline comments survive.

**Tech Stack:** Python 3.11, `ruamel.yaml` 0.18.6 (round-trip), `pytest` with `tmp_path`. Normalization reuses `workflows/workflow_types.py` enums (`DeviceTypes`, `InferenceEngine`) — NOT `workflows/model_spec.py`, which loads the catalogues as an import side effect.

**Design reference:** `docs/superpowers/specs/2026-06-02-promote-dev-spec-to-prod-design.md`

> **Post-implementation correction (read first):** Two assumptions in the tasks below
> were wrong against real data and were corrected during implementation:
> 1. **Writing** — a `ruamel.yaml` whole-document round-trip does NOT preserve these files
>    (their block-sequence indentation is inconsistent), so it reformatted every untouched
>    template. The final code splices template **text blocks** instead (see the design doc
>    and the `split_into_blocks`/`upsert_block` functions).
> 2. **Identity** — the catalogue holds multiple blocks per `(impl, engine, weights)`, one
>    per device group, so the upsert identity includes the **device set**:
>    `(impl, engine, frozenset(weights), frozenset(devices))`.
>
> The TDD task structure below is still accurate; treat the `_yaml`/ruamel-write and
> 3-tuple-identity snippets as superseded by the committed implementation.

**Key facts (verified):**
- `InferenceEngine.from_string(s)` = `cls[s.upper()]`, so both ci-config `"vLLM"` and yaml `"VLLM"` resolve to `InferenceEngine.VLLM`. Same for `MEDIA`/`FORGE`.
- `DeviceTypes.from_string(s)` = `cls[s.upper()]`. Release devices in the real config are `GALAXY`, `P150`, `P300X2` (all valid).
- ci-config model entries are either flat (`{inference_engine, ci}`) or have an `implementations: [...]` array.
- The ci-config model-name key equals `Path(weight).name` of a spec's weights (e.g. `Llama-3.1-8B-Instruct` ↔ `meta-llama/Llama-3.1-8B-Instruct`).
- Tests live in `tests/` (pytest `testpaths=["tests"]`, `pythonpath=["."]`). `scripts.release.<mod>` is importable as a namespace package.
- The script file needs the SPDX header used across the repo (see existing files).

---

## File Structure

- **Create** `scripts/release/promote_dev_spec_to_prod.py` — the whole tool (pure helpers + `promote()` orchestrator + `main()` CLI). One file; the functions are small and cohesive.
- **Create** `tests/test_promote_dev_spec_to_prod.py` — unit + integration tests using `tmp_path` catalogue trees.

---

## Task 1: Scaffold module + pure helpers

**Files:**
- Create: `scripts/release/promote_dev_spec_to_prod.py`
- Test: `tests/test_promote_dev_spec_to_prod.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_promote_dev_spec_to_prod.py
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from scripts.release.promote_dev_spec_to_prod import (
    iter_implementations,
    model_name_from_weight,
)


def test_model_name_from_weight_strips_org_prefix():
    assert model_name_from_weight("meta-llama/Llama-3.1-8B-Instruct") == (
        "Llama-3.1-8B-Instruct"
    )
    assert model_name_from_weight("openai/gpt-oss-20b") == "gpt-oss-20b"


def test_iter_implementations_flat_shape():
    entry = {"inference_engine": "FORGE", "ci": {"nightly": {"devices": ["P150"]}}}
    assert list(iter_implementations(entry)) == [entry]


def test_iter_implementations_array_shape():
    impl_a = {"inference_engine": "vLLM", "ci": {}}
    impl_b = {"inference_engine": "FORGE", "ci": {}}
    entry = {"implementations": [impl_a, impl_b]}
    assert list(iter_implementations(entry)) == [impl_a, impl_b]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_promote_dev_spec_to_prod.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.release.promote_dev_spec_to_prod'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/release/promote_dev_spec_to_prod.py
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""
Promote dev model specs to prod for every model-device combination marked
``release`` in models-ci-config.json.

For each (model, inference_engine, device) under ``ci.release`` in the CI config,
the matching template in workflows/model_specs/dev/ is copied (whole, with inline
comments preserved) into the same-named file in workflows/model_specs/prod/,
upserting by (impl, inference_engine, weights) identity.
"""

import argparse
import json
import sys
from collections import namedtuple
from copy import deepcopy
from io import StringIO
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from workflows.workflow_types import DeviceTypes, InferenceEngine  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CI_CONFIG = REPO_ROOT / ".github" / "workflows" / "models-ci-config.json"
DEFAULT_DEV_DIR = REPO_ROOT / "workflows" / "model_specs" / "dev"
DEFAULT_PROD_DIR = REPO_ROOT / "workflows" / "model_specs" / "prod"

ReleaseCombo = namedtuple("ReleaseCombo", ["model_name", "engine", "device"])


def _yaml() -> YAML:
    """A round-trip YAML configured to preserve comments and avoid line wrapping."""
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096
    return y


def model_name_from_weight(weight: str) -> str:
    """Extract the model name (basename) from a HuggingFace repo path."""
    return Path(weight).name


def iter_implementations(model_entry: dict):
    """Yield each implementation dict for a CI-config model entry.

    Handles both the flat shape ({inference_engine, ci}) and the
    implementations:[...] array shape.
    """
    if "implementations" in model_entry:
        yield from model_entry["implementations"]
    else:
        yield model_entry
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_promote_dev_spec_to_prod.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/release/promote_dev_spec_to_prod.py tests/test_promote_dev_spec_to_prod.py
git commit -m "feat(release): scaffold promote_dev_spec_to_prod with pure helpers"
```

---

## Task 2: collect_release_combos

**Files:**
- Modify: `scripts/release/promote_dev_spec_to_prod.py`
- Test: `tests/test_promote_dev_spec_to_prod.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_promote_dev_spec_to_prod.py
from scripts.release.promote_dev_spec_to_prod import (  # noqa: E402
    ReleaseCombo,
    collect_release_combos,
)
from workflows.workflow_types import DeviceTypes, InferenceEngine  # noqa: E402


def test_collect_release_combos_array_and_flat_shapes():
    ci_config = {
        "models": {
            "Llama-3.1-8B-Instruct": {
                "implementations": [
                    {
                        "inference_engine": "vLLM",
                        "ci": {
                            "nightly": {"devices": ["N150"]},
                            "release": {"devices": ["GALAXY", "P300X2"]},
                        },
                    },
                    {"inference_engine": "FORGE", "ci": {"nightly": {"devices": ["P150"]}}},
                ]
            },
            "whisper-large-v3": {
                "inference_engine": "MEDIA",
                "ci": {"release": {"devices": ["P150"]}},
            },
            "Falcon3-7B-Instruct": {
                "inference_engine": "FORGE",
                "ci": {"nightly": {"devices": ["P150"]}},
            },
        }
    }
    combos = collect_release_combos(ci_config)
    assert combos == {
        ReleaseCombo("Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.GALAXY),
        ReleaseCombo("Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.P300X2),
        ReleaseCombo("whisper-large-v3", InferenceEngine.MEDIA, DeviceTypes.P150),
    }


def test_collect_release_combos_ignores_nightly_and_weekly():
    ci_config = {
        "models": {
            "m": {
                "inference_engine": "vLLM",
                "ci": {
                    "nightly": {"devices": ["GALAXY"]},
                    "weekly": {"devices": ["GALAXY"]},
                },
            }
        }
    }
    assert collect_release_combos(ci_config) == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_promote_dev_spec_to_prod.py -k collect -v`
Expected: FAIL — `ImportError: cannot import name 'collect_release_combos'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to scripts/release/promote_dev_spec_to_prod.py
def collect_release_combos(ci_config: dict) -> set:
    """Return the set of ReleaseCombo(model_name, engine, device) marked release."""
    combos = set()
    for model_name, entry in ci_config.get("models", {}).items():
        for impl in iter_implementations(entry):
            release = impl.get("ci", {}).get("release")
            if not release:
                continue
            engine = InferenceEngine.from_string(impl["inference_engine"])
            for device in release.get("devices", []):
                combos.add(
                    ReleaseCombo(
                        model_name, engine, DeviceTypes.from_string(device)
                    )
                )
    return combos
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_promote_dev_spec_to_prod.py -k collect -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/release/promote_dev_spec_to_prod.py tests/test_promote_dev_spec_to_prod.py
git commit -m "feat(release): collect release combos from ci-config"
```

---

## Task 3: Matching + identity helpers

**Files:**
- Modify: `scripts/release/promote_dev_spec_to_prod.py`
- Test: `tests/test_promote_dev_spec_to_prod.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_promote_dev_spec_to_prod.py
from scripts.release.promote_dev_spec_to_prod import (  # noqa: E402
    template_identity,
    template_matches,
)


def _llama_template():
    return {
        "weights": ["meta-llama/Llama-3.1-8B-Instruct"],
        "impl": "tt_transformers",
        "inference_engine": "VLLM",
        "device_model_specs": [
            {"device": "GALAXY", "max_concurrency": 32},
            {"device": "N150", "max_concurrency": 1},
            {"device": "P300X2", "max_concurrency": 8},
        ],
    }


def test_template_matches_on_basename_engine_and_device():
    combo = ReleaseCombo(
        "Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.GALAXY
    )
    assert template_matches(_llama_template(), combo) is True


def test_template_does_not_match_wrong_device():
    combo = ReleaseCombo(
        "Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.T3K
    )
    assert template_matches(_llama_template(), combo) is False


def test_template_does_not_match_wrong_engine():
    combo = ReleaseCombo(
        "Llama-3.1-8B-Instruct", InferenceEngine.FORGE, DeviceTypes.GALAXY
    )
    assert template_matches(_llama_template(), combo) is False


def test_template_identity_is_impl_engine_weights():
    assert template_identity(_llama_template()) == (
        "tt_transformers",
        InferenceEngine.VLLM,
        frozenset({"meta-llama/Llama-3.1-8B-Instruct"}),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_promote_dev_spec_to_prod.py -k "matches or identity" -v`
Expected: FAIL — `ImportError: cannot import name 'template_matches'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to scripts/release/promote_dev_spec_to_prod.py
def template_engine(template: dict) -> InferenceEngine:
    return InferenceEngine.from_string(template["inference_engine"])


def template_devices(template: dict) -> set:
    return {
        DeviceTypes.from_string(d["device"])
        for d in template.get("device_model_specs", [])
    }


def template_model_names(template: dict) -> set:
    return {model_name_from_weight(w) for w in template.get("weights", [])}


def template_matches(template: dict, combo: ReleaseCombo) -> bool:
    """True if the template provides the given release combo."""
    return (
        combo.model_name in template_model_names(template)
        and combo.engine == template_engine(template)
        and combo.device in template_devices(template)
    )


def template_identity(template: dict):
    """Upsert identity for a template: (impl, engine, frozenset(weights))."""
    return (
        template["impl"],
        template_engine(template),
        frozenset(template.get("weights", [])),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_promote_dev_spec_to_prod.py -k "matches or identity" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/release/promote_dev_spec_to_prod.py tests/test_promote_dev_spec_to_prod.py
git commit -m "feat(release): add template match + identity helpers"
```

---

## Task 4: find_matches over the dev catalogue

**Files:**
- Modify: `scripts/release/promote_dev_spec_to_prod.py`
- Test: `tests/test_promote_dev_spec_to_prod.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_promote_dev_spec_to_prod.py
import textwrap  # noqa: E402

from scripts.release.promote_dev_spec_to_prod import find_matches  # noqa: E402


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


def test_find_matches_picks_whole_template_and_reports_unmatched(tmp_path):
    dev = tmp_path / "dev"
    _write(
        dev / "llm.yaml",
        """
        templates:
        - weights:
            - meta-llama/Llama-3.1-8B-Instruct
          impl: tt_transformers
          inference_engine: VLLM
          device_model_specs:
            - device: GALAXY
              max_concurrency: 32
            - device: N150
              max_concurrency: 1
        """,
    )
    combos = {
        ReleaseCombo(
            "Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.GALAXY
        ),
        ReleaseCombo("nonexistent", InferenceEngine.VLLM, DeviceTypes.GALAXY),
    }
    matches_by_file, unmatched = find_matches(dev, combos)

    assert list(matches_by_file.keys()) == ["llm.yaml"]
    picked = matches_by_file["llm.yaml"]
    assert len(picked) == 1
    # whole template: the non-release N150 device is still present
    devices = [d["device"] for d in picked[0]["device_model_specs"]]
    assert devices == ["GALAXY", "N150"]
    assert unmatched == {
        ReleaseCombo("nonexistent", InferenceEngine.VLLM, DeviceTypes.GALAXY)
    }


def test_find_matches_dedups_template_matched_by_two_combos(tmp_path):
    dev = tmp_path / "dev"
    _write(
        dev / "llm.yaml",
        """
        templates:
        - weights:
            - meta-llama/Llama-3.1-8B-Instruct
          impl: tt_transformers
          inference_engine: VLLM
          device_model_specs:
            - device: GALAXY
            - device: P300X2
        """,
    )
    combos = {
        ReleaseCombo(
            "Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.GALAXY
        ),
        ReleaseCombo(
            "Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.P300X2
        ),
    }
    matches_by_file, unmatched = find_matches(dev, combos)
    assert len(matches_by_file["llm.yaml"]) == 1
    assert unmatched == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_promote_dev_spec_to_prod.py -k find_matches -v`
Expected: FAIL — `ImportError: cannot import name 'find_matches'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to scripts/release/promote_dev_spec_to_prod.py
def find_matches(dev_dir: Path, combos: set):
    """Scan dev catalog files for templates matching any release combo.

    Returns (matches_by_file, unmatched):
      - matches_by_file: dict filename -> list of matched template objects,
        in file order, de-duplicated by identity.
      - unmatched: set of combos that matched no dev template.
    """
    yaml = _yaml()
    matched_combos = set()
    matches_by_file = {}
    for dev_file in sorted(dev_dir.glob("*.yaml")):
        doc = yaml.load(dev_file.read_text())
        templates = (doc or {}).get("templates") or []
        picked = []
        picked_ids = set()
        for template in templates:
            hits = [c for c in combos if template_matches(template, c)]
            if not hits:
                continue
            matched_combos.update(hits)
            identity = template_identity(template)
            if identity not in picked_ids:
                picked.append(template)
                picked_ids.add(identity)
        if picked:
            matches_by_file[dev_file.name] = picked
    unmatched = combos - matched_combos
    return matches_by_file, unmatched
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_promote_dev_spec_to_prod.py -k find_matches -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/release/promote_dev_spec_to_prod.py tests/test_promote_dev_spec_to_prod.py
git commit -m "feat(release): find matching dev templates for release combos"
```

---

## Task 5: upsert_template

**Files:**
- Modify: `scripts/release/promote_dev_spec_to_prod.py`
- Test: `tests/test_promote_dev_spec_to_prod.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_promote_dev_spec_to_prod.py
from ruamel.yaml.comments import CommentedSeq  # noqa: E402

from scripts.release.promote_dev_spec_to_prod import upsert_template  # noqa: E402


def test_upsert_appends_new_template():
    prod = CommentedSeq()
    tmpl = {
        "weights": ["meta-llama/Llama-3.1-8B-Instruct"],
        "impl": "tt_transformers",
        "inference_engine": "VLLM",
        "device_model_specs": [{"device": "GALAXY"}],
    }
    action = upsert_template(prod, tmpl)
    assert action == "appended"
    assert len(prod) == 1


def test_upsert_replaces_same_identity_and_leaves_others():
    other = {
        "weights": ["openai/gpt-oss-20b"],
        "impl": "gpt_oss",
        "inference_engine": "VLLM",
        "device_model_specs": [{"device": "T3K"}],
    }
    old = {
        "weights": ["meta-llama/Llama-3.1-8B-Instruct"],
        "impl": "tt_transformers",
        "inference_engine": "VLLM",
        "version": "0.1.0",
        "device_model_specs": [{"device": "GALAXY"}],
    }
    prod = CommentedSeq()
    prod.append(other)
    prod.append(old)

    new = dict(old)
    new["version"] = "0.2.0"
    action = upsert_template(prod, new)

    assert action == "updated"
    assert len(prod) == 2
    assert prod[0] is other  # untouched
    assert prod[1]["version"] == "0.2.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_promote_dev_spec_to_prod.py -k upsert -v`
Expected: FAIL — `ImportError: cannot import name 'upsert_template'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to scripts/release/promote_dev_spec_to_prod.py
def upsert_template(prod_templates, template) -> str:
    """Insert or replace template in prod_templates by identity.

    Returns "updated" if an existing same-identity template was replaced in
    place, else "appended".
    """
    identity = template_identity(template)
    for i, existing in enumerate(prod_templates):
        if template_identity(existing) == identity:
            prod_templates[i] = template
            return "updated"
    prod_templates.append(template)
    return "appended"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_promote_dev_spec_to_prod.py -k upsert -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/release/promote_dev_spec_to_prod.py tests/test_promote_dev_spec_to_prod.py
git commit -m "feat(release): upsert template into prod list by identity"
```

---

## Task 6: promote() orchestrator (write, comments, dry-run, idempotency)

**Files:**
- Modify: `scripts/release/promote_dev_spec_to_prod.py`
- Test: `tests/test_promote_dev_spec_to_prod.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_promote_dev_spec_to_prod.py
import json  # noqa: E402

from scripts.release.promote_dev_spec_to_prod import promote  # noqa: E402


def _build_tree(tmp_path):
    """dev has a Llama template with an inline comment; prod has an older copy."""
    dev = tmp_path / "dev"
    prod = tmp_path / "prod"
    _write(
        dev / "llm.yaml",
        """
        templates:
        - weights:
            - meta-llama/Llama-3.1-8B-Instruct
          impl: tt_transformers
          inference_engine: VLLM
          version: "0.2.0"
          device_model_specs:
            - device: GALAXY
              max_context: 16384  # 16 * 1024
            - device: N150
              max_context: 4096
        """,
    )
    _write(
        prod / "llm.yaml",
        """
        templates:
        - weights:
            - meta-llama/Llama-3.1-8B-Instruct
          impl: tt_transformers
          inference_engine: VLLM
          version: "0.1.0"
          device_model_specs:
            - device: GALAXY
              max_context: 16384
        """,
    )
    ci = tmp_path / "ci.json"
    ci.write_text(
        json.dumps(
            {
                "models": {
                    "Llama-3.1-8B-Instruct": {
                        "inference_engine": "vLLM",
                        "ci": {"release": {"devices": ["GALAXY"]}},
                    }
                }
            }
        )
    )
    return ci, dev, prod


def test_promote_updates_prod_preserving_comment_and_whole_template(tmp_path):
    ci, dev, prod = _build_tree(tmp_path)
    report = promote(ci, dev, prod, dry_run=False)

    text = (prod / "llm.yaml").read_text()
    assert "0.2.0" in text          # version bumped from dev
    assert "# 16 * 1024" in text    # inline comment preserved
    assert "device: N150" in text   # whole template copied (non-release device)
    assert report["unmatched"] == set()
    assert "llm.yaml" in report["changed_files"]


def test_promote_dry_run_writes_nothing(tmp_path):
    ci, dev, prod = _build_tree(tmp_path)
    before = (prod / "llm.yaml").read_text()
    report = promote(ci, dev, prod, dry_run=True)
    assert (prod / "llm.yaml").read_text() == before
    assert report["changed_files"] == []


def test_promote_is_idempotent(tmp_path):
    ci, dev, prod = _build_tree(tmp_path)
    promote(ci, dev, prod, dry_run=False)
    after_first = (prod / "llm.yaml").read_text()
    report = promote(ci, dev, prod, dry_run=False)
    assert (prod / "llm.yaml").read_text() == after_first
    assert report["changed_files"] == []


def test_promote_reports_unmatched_combo(tmp_path):
    ci, dev, prod = _build_tree(tmp_path)
    ci.write_text(
        json.dumps(
            {
                "models": {
                    "ghost-model": {
                        "inference_engine": "vLLM",
                        "ci": {"release": {"devices": ["GALAXY"]}},
                    }
                }
            }
        )
    )
    report = promote(ci, dev, prod, dry_run=False)
    assert ReleaseCombo(
        "ghost-model", InferenceEngine.VLLM, DeviceTypes.GALAXY
    ) in report["unmatched"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_promote_dev_spec_to_prod.py -k promote -v`
Expected: FAIL — `ImportError: cannot import name 'promote'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to scripts/release/promote_dev_spec_to_prod.py
def _dump_to_str(yaml: YAML, doc) -> str:
    buf = StringIO()
    yaml.dump(doc, buf)
    return buf.getvalue()


def promote(ci_config_path, dev_dir, prod_dir, dry_run=False) -> dict:
    """Promote release-marked dev templates into prod.

    Returns a report dict:
      - combos: set of all release combos
      - matches_by_file: dict filename -> matched dev templates
      - unmatched: set of combos with no dev template
      - actions: dict filename -> list of (identity, "appended"|"updated")
      - changed_files: list of prod filenames whose content changed
    """
    yaml = _yaml()
    ci_config = json.loads(Path(ci_config_path).read_text())
    combos = collect_release_combos(ci_config)
    matches_by_file, unmatched = find_matches(Path(dev_dir), combos)

    actions = {}
    changed_files = []
    for filename, templates in matches_by_file.items():
        prod_file = Path(prod_dir) / filename
        original = prod_file.read_text() if prod_file.exists() else ""
        doc = yaml.load(original) if original else None
        if not isinstance(doc, CommentedMap):
            doc = CommentedMap()
        if not isinstance(doc.get("templates"), CommentedSeq):
            doc["templates"] = CommentedSeq()

        file_actions = []
        for template in templates:
            action = upsert_template(doc["templates"], deepcopy(template))
            file_actions.append((template_identity(template), action))
        actions[filename] = file_actions

        new_text = _dump_to_str(yaml, doc)
        if new_text != original:
            changed_files.append(filename)
            if not dry_run:
                prod_file.parent.mkdir(parents=True, exist_ok=True)
                prod_file.write_text(new_text)

    return {
        "combos": combos,
        "matches_by_file": matches_by_file,
        "unmatched": unmatched,
        "actions": actions,
        "changed_files": changed_files,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_promote_dev_spec_to_prod.py -k promote -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/release/promote_dev_spec_to_prod.py tests/test_promote_dev_spec_to_prod.py
git commit -m "feat(release): promote orchestrator with comment-preserving upsert"
```

---

## Task 7: main() CLI + exit codes

**Files:**
- Modify: `scripts/release/promote_dev_spec_to_prod.py`
- Test: `tests/test_promote_dev_spec_to_prod.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_promote_dev_spec_to_prod.py
from scripts.release.promote_dev_spec_to_prod import main  # noqa: E402


def test_main_returns_zero_and_writes(tmp_path, capsys):
    ci, dev, prod = _build_tree(tmp_path)
    rc = main(
        ["--ci-config", str(ci), "--dev-dir", str(dev), "--prod-dir", str(prod)]
    )
    assert rc == 0
    assert "0.2.0" in (prod / "llm.yaml").read_text()


def test_main_dry_run_writes_nothing(tmp_path):
    ci, dev, prod = _build_tree(tmp_path)
    before = (prod / "llm.yaml").read_text()
    rc = main(
        [
            "--ci-config", str(ci),
            "--dev-dir", str(dev),
            "--prod-dir", str(prod),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert (prod / "llm.yaml").read_text() == before


def test_main_returns_nonzero_on_unmatched(tmp_path, capsys):
    ci, dev, prod = _build_tree(tmp_path)
    ci.write_text(
        json.dumps(
            {
                "models": {
                    "ghost-model": {
                        "inference_engine": "vLLM",
                        "ci": {"release": {"devices": ["GALAXY"]}},
                    }
                }
            }
        )
    )
    rc = main(
        ["--ci-config", str(ci), "--dev-dir", str(dev), "--prod-dir", str(prod)]
    )
    assert rc == 1
    assert "ghost-model" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_promote_dev_spec_to_prod.py -k main -v`
Expected: FAIL — `ImportError: cannot import name 'main'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to scripts/release/promote_dev_spec_to_prod.py
def _combo_str(combo: ReleaseCombo) -> str:
    return f"{combo.model_name} [{combo.engine.name}] on {combo.device.name}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Promote release-marked dev model specs into the prod catalogue."
        )
    )
    parser.add_argument("--ci-config", type=Path, default=DEFAULT_CI_CONFIG)
    parser.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_DIR)
    parser.add_argument("--prod-dir", type=Path, default=DEFAULT_PROD_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report intended changes without writing any files.",
    )
    args = parser.parse_args(argv)

    report = promote(
        args.ci_config, args.dev_dir, args.prod_dir, dry_run=args.dry_run
    )

    prefix = "[dry-run] " if args.dry_run else ""
    for filename, file_actions in sorted(report["actions"].items()):
        for identity, action in file_actions:
            impl, engine, weights = identity
            print(
                f"{prefix}{action.upper():8} {filename}: "
                f"{impl} [{engine.name}] {sorted(weights)}"
            )
    changed = report["changed_files"]
    print(f"{prefix}{len(changed)} prod file(s) changed: {sorted(changed)}")

    for combo in sorted(
        report["unmatched"],
        key=lambda c: (c.model_name, c.engine.name, c.device.name),
    ):
        print(f"WARNING: no dev template found for {_combo_str(combo)}")

    return 1 if report["unmatched"] else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_promote_dev_spec_to_prod.py -k main -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/release/promote_dev_spec_to_prod.py tests/test_promote_dev_spec_to_prod.py
git commit -m "feat(release): add CLI entrypoint with unmatched-combo exit code"
```

---

## Task 8: Integration check against the real repo + full suite

**Files:**
- Test: `tests/test_promote_dev_spec_to_prod.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_promote_dev_spec_to_prod.py
from scripts.release.promote_dev_spec_to_prod import (  # noqa: E402
    DEFAULT_CI_CONFIG,
    DEFAULT_DEV_DIR,
    DEFAULT_PROD_DIR,
)


def test_real_repo_release_combos_all_match_dev():
    """Every release-marked combo in the real ci-config exists in the dev catalogue."""
    ci_config = json.loads(DEFAULT_CI_CONFIG.read_text())
    combos = collect_release_combos(ci_config)
    assert combos, "expected at least one release combo in the real ci-config"
    _, unmatched = find_matches(DEFAULT_DEV_DIR, combos)
    assert unmatched == set(), f"release combos missing from dev: {unmatched}"


def test_real_repo_dry_run_against_prod_succeeds(tmp_path):
    """Dry-run against the real catalogues runs cleanly and writes nothing to repo."""
    report = promote(
        DEFAULT_CI_CONFIG, DEFAULT_DEV_DIR, DEFAULT_PROD_DIR, dry_run=True
    )
    assert report["unmatched"] == set()
```

- [ ] **Step 2: Run test to verify it (likely) passes or surfaces real drift**

Run: `.venv/bin/pytest tests/test_promote_dev_spec_to_prod.py -k real_repo -v`
Expected: PASS. (dev and prod are currently byte-identical, so all 10 release combos resolve.) If it FAILS, the assertion message names the drifting combo — investigate before continuing.

- [ ] **Step 3: No implementation needed**

These are integration assertions over existing code. If Step 2 fails due to a real-data edge case (e.g. a release model whose dev weight basename differs from its ci-config key), STOP and report — do not paper over it.

- [ ] **Step 4: Run the full file + repo suite**

Run: `.venv/bin/pytest tests/test_promote_dev_spec_to_prod.py -v`
Expected: PASS (all tests)

Run: `.venv/bin/ruff check scripts/release/promote_dev_spec_to_prod.py tests/test_promote_dev_spec_to_prod.py && .venv/bin/ruff format --check scripts/release/promote_dev_spec_to_prod.py tests/test_promote_dev_spec_to_prod.py`
Expected: no lint/format errors (matches the repo pre-commit `ruff`/`ruff-format` hooks)

- [ ] **Step 5: Commit**

```bash
git add tests/test_promote_dev_spec_to_prod.py
git commit -m "test(release): integration coverage for promote_dev_spec_to_prod"
```

---

## Self-Review

**Spec coverage:**
- Collect release combos (flat + implementations, release-only) → Task 2 ✓
- Match by weights basename + engine + device, with enum normalization → Task 3 ✓
- Whole-template copy (non-release devices retained) → Task 4 (find) + Task 6 (write) ✓
- Upsert by `(impl, engine, weights)` identity; untouched prod entries preserved → Task 5 ✓
- Same-named prod file placement → Task 6 (`Path(prod_dir)/filename`) ✓
- Comment/formatting preservation via ruamel round-trip → Task 6 ✓
- Dry-run writes nothing → Task 6 + Task 7 ✓
- Idempotency → Task 6 ✓
- Unmatched combo warning + non-zero exit → Task 6 (report) + Task 7 (exit) ✓
- CLI flags (`--ci-config/--dev-dir/--prod-dir/--dry-run`) → Task 7 ✓
- Tests alongside `test_model_specification.py` → all tasks ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** `ReleaseCombo(model_name, engine, device)` used consistently; `template_identity` returns `(impl, InferenceEngine, frozenset)` everywhere; `promote` report keys (`combos`, `matches_by_file`, `unmatched`, `actions`, `changed_files`) used consistently in Task 7. `_yaml()`, `_dump_to_str()` defined before use. ✓
