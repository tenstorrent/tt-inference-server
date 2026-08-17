# Workflow venv requirements

Pip dependency lists for each `WorkflowVenvType` defined in
`workflows/workflow_types.py`. Each per-venv file is consumed by the
corresponding `setup_*` function in `workflows/workflow_venvs.py` via the
`install_requirements()` helper.

## Layout

```
requirements/
  constraints.txt            # shared pins (pyjwt, pillow, open-clip-torch)
  <venv-name>.txt            # one file per venv that needs pip installs
```

File names are kebab-case lowercase mirrors of the `WorkflowVenvType` enum
member: `EVALS_VIDEO` → `evals-video.txt`, `BENCHMARKS_AIPERF` →
`benchmarks-aiperf.txt`.

`BENCHMARKS_VIDEO` and `BENCHMARKS_GENAI_PERF` deliberately have no file —
they don't pip-install anything (the former is a no-op `setup_venv()`, the
latter is Docker-only).

## Authoring rules

### `constraints.txt`

Pins live here only when a package is used in **multiple** venvs and we want
all of them aligned on the same version. A constraint does **not** install a
package — it only constrains the version *if* the package is resolved
(directly or transitively) in a given venv.

When you add a constraint:

1. Pin the version explicitly (`name==X.Y.Z`).
2. Drop the version from every per-venv file that lists the same package —
   they should just say `name`. The constraints file is the single source of
   truth.

### Per-venv files

A typical per-venv file looks like:

```
-c constraints.txt
--extra-index-url https://download.pytorch.org/whl/cpu

torch
torchvision
requests
pyjwt
pillow
```

Conventions:

- Always start with `-c constraints.txt` so shared pins apply.
- If the venv needs the PyTorch CPU index, add
  `--extra-index-url https://download.pytorch.org/whl/cpu` (not
  `--index-url` — we want PyPI as a fallback for non-torch packages).
- Use unpinned package names by default; pin a version inline only when the
  pin is venv-specific (e.g. `lm-eval==0.4.3` in `evals-meta.txt`).
- Editable installs (`pip install -e .`) and operations that depend on a
  working directory cannot live in a requirements file — keep those in the
  setup function (see `setup_evals_meta` for an example).

## Adding a new workflow venv

1. Add a new member to `WorkflowVenvType`.
2. Create `requirements/<name>.txt` (skip if the venv has no pip deps).
3. Implement `setup_<name>()` in `workflow_venvs.py`. For pure-pip venvs
   it's typically a one-liner:
   ```python
   def setup_<name>(venv_config, model_spec) -> bool:
       return install_requirements(venv_config, "<name>.txt")
   ```
4. Register a `VenvConfig` in `_venv_config_list`.

## Validating a requirements file locally

```bash
uv pip compile --quiet --index-strategy unsafe-best-match --no-deps \
    requirements/<name>.txt
```

This resolves direct dependencies against the configured indexes and prints
the locked versions. Catches typos, missing packages, and constraint
conflicts without doing a full install.
