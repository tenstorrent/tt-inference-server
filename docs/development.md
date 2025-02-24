# Development

Set up linting and formatting for development:
```bash
# [optional] step 1: use venv
python3 -m venv .venv
source .venv/bin/activate

# step 2: install
pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt

# step 3: pre-commit
pre-commit install
```

run pre-commit:
```bash
# pre-commit behaviour is defined in .pre-commit-config.yaml
# by default it runs only on git staged files
pre-commit run
# run on all files in repo
pre-commit run --all-files
# or point to specific files
pre-commit run --files path/to/file
```

# Contributions

Follow the development and release git workflow, steps described below image:

![git workflow](git-workflow.png)

## Development workflow

1. Make changes on a branch from `dev` following the convention: `username/feature-name` or `username/fix-name`
2. Test those changes locally (rebase from `dev` if needed)
3. merge your branch, e.g. `username/feature-name` to `dev` to consolidate development changes rapidly

## Release workflow

1. make Release Candidate (RC) branch from `main` following convention `rc-vx.x.x`
2. cherry pick changes from `dev` to the RC branch
3. make Docker images for RC.
4. test RC branch locally
5. PR from `rc-vx.x.x` to `main`
6. Add any changes/fixes needed to `dev` and similarly cherry pick onto `rc-vx.x.x`, re-test changes.
7. after PR merges to `main`, create release `vx.x.x` from `main`, publish release package Docker images.
