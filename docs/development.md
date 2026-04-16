# Development

All development uses the following workflows in git for the repo repository https://github.com/tenstorrent/tt-inference-server.

## Git Workflows

### development in `dev`

All normal development work should be done using this simple workflow, making branches off `dev` and making PRs back to `dev`.

**branches:**
- `<namett>/<my-description>`
- `<namett>/fix-<my-description>`
- `dev`

**workflow for development:**
1. git checkout `dev` && git pull
1. git checkout -b `<namett>/<my-description>` or `<namett>/fix-<my-description>`
1. make code changes
1. PR `<namett>/<my-description>` to `dev`
1. PR reviewed and approved by >= 1 responsible person defined in CODEOWNERS.md
1. squash and merge PR to `dev`

### releases in `main`

Only repo maintainers do releases to `main`.

branches:
- `dev`
- `pre-release-v<MAJOR>.<MINOR>.<PATCH>`
- `rc-v<MAJOR>.<MINOR>.<PATCH>`
- `main`

workflow for release:
- git checkout `dev` && git pull
- git checkout `<specific-release-commit-SHA>` (from Models CI, should be a recent commit on `dev`)
- follow release doc at [../scripts/release/README.md](../scripts/release/README.md)
- PR reviewed and approved by >= 1 responsible person defined in CODEOWNERS.md
- squash and merge PR to `dev`

### Branch descriptions

`main` (default on GitHub):
- releases only via PR
- all changes must be passing in Models CI
- PR merge sign-off from repo owners
- PRs are merged into main to allow users to see well documented linear history

`dev`:
- where independent development work is consolidated
- Models CI runs nightly using this branch
- Merge criteria
  - PR reviewed and approved by from >= 1 responsible person defined in CODEOWNERS.md
  - well named and documented commits refing PR 
  - squash to merge in GitHub

`<namett>/<my-description>`:
- specific WIP development work
- based off `dev`, PR back to `dev`
- multiple people can work together on a single branch, but generally easier to structure collaboration  via multiple PRs into `dev

`<namett>/fix-<my-description>`:
- bug fix (generally) higher priority than features
- based off `dev`, PR back to `dev`
- similar approach as feature branches

`pre-release-v<MAJOR>.<MINOR>.<PATCH>`:
- branch specifically for updating based on Models CI, for a given release:
  1. tt-metal and vLLM commits
  2. VERSION
  3. metadata
  4. documentation
- based off `dev` release commit (from Models CI passing), PR back to `dev`

`rc-v<MAJOR>.<MINOR>.<PATCH>`:
- release candidate (RC) branches are based from `dev` commit that is passing Models CI nightly and intended for release, this is to provide a "stable" branch for release to avoid inclusion of on-going work on `dev`.
- bug fix (hot fix) commits can be cherry-picked into RC branches directly. 
- features should be passing Models CI from dev before adding to RC, and therefore RC branch should be rebased to `dev` if additional features are needed in a given release last minute (best if avoided, push them into next release if possible and release frequently).
- follows [semver](https://semver.org/), the "API" is generally 1) run.py automation CLI script, 2) openai API for serving LLMs, 3) tt-media-server defined HTTP APIs for serving multi-media models.
  - MAJOR version when you make incompatible API changes
  - MINOR version when you add functionality in a backward compatible manner
  - PATCH version when you make backward compatible bug fixes
- based off `pre-release-v<MAJOR>.<MINOR>.<PATCH>`, PR to `main`

### Git Workflows Diagram

Follow the development and release git workflow, steps described below image:

![ttis-git-workflows-2026-02-10](ttis-git-workflows-2026-02-10.png)

### pre-commit

Pre-commit usage is defined in `.pre-commit-config.yaml`.

Set up linting and formatting for development:
#### option 1: use uv
```bash
# option 1: use uv
uv venv .pre-commit --python 3.10
source .pre-commit/bin/activate

uv pip install -r requirements-dev.txt
uv pip install -r tt-media-server/requirements.txt
```

#### option 2: use os python venv
```bash
# option 2: use os python venv
python3 -m venv .pre-commit
source .pre-commit/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt
pip install -r tt-media-server/requirements.txt
```

#### running pre-commit tests

```bash
# option 1: use git pre-commit hooks direct
. scripts/setup-hooks.sh

# option 2: use pre-commit tool
pre-commit install
```

run pre-commit:
```bash
# run git hooks script directly
.git/hooks/pre-commit

# pre-commit behaviour is defined in .pre-commit-config.yaml
# by default it runs only on git staged files
pre-commit run
# run on all files in repo
pre-commit run --all-files
# or point to specific files
pre-commit run --files path/to/file
```

### Workflow smoke tests

Use `--limit-samples-mode smoke-test` for fast end-to-end workflow validation while iterating on `benchmarks` or `evals`.

```bash
python3 run.py --model Llama-3.2-1B-Instruct --tt-device n300 --workflow benchmarks --limit-samples-mode smoke-test
python3 run.py --model Llama-3.2-1B-Instruct --tt-device n300 --workflow evals --limit-samples-mode smoke-test
```

If you also want `run.py` to launch the inference server for the run, add `--docker-server`.

Smoke-test mode keeps the run short by reducing `benchmarks` to a single lightweight target and `evals` to the first configured eval task with 3 samples.

### How to build Docker images for a specific model (tt-metal, vLLM commits)

For building containers for development it is generally faster to use 
```bash
python3 scripts/build_docker_images.py --build-metal-commit <my_metal_commit_SHA_or_tag>
```
This filters the Docker images to be built for only the tt-metal version needed.

### What to do if I can't find the Docker image I need for development?

Ideally you can do development without Docker using `--local-server` and building tt-metal + vLLM locally.

If you need to develop with a Docker image build the image locally:

##### Step 1: edit workflows/model_spec.py

Find and edit the ref `ModelSpecTemplate` your model-hardware combination, e.g. for `Llama-3.2-1B` on n150:
Update the commits:
```python
    ModelSpecTemplate(
        weights=["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"],
        impl=tt_transformers_impl,
        tt_metal_commit=<my_metal_commit_SHA_or_tag>,
        vllm_commit=<my_vllm_commit_SHA>,
        inference_engine=InferenceEngine.VLLM.value,
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.N150,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
```

##### Step 2: build the Docker image locally
```
python3 scripts/build_docker_images.py --build-metal-commit <my_metal_commit_SHA_or_tag>

```


## Release process

See document on release process and scripts: [../scripts/release/README.md](../scripts/release/README.md)

## Git worktree usage

How to manage many branches in parallel on a single host machine.

### Why use `git worktree`?
- true parallel development on single machine (good for multiple code agents)
- better context switching without stashing changes
- saves disk space and overhead compared to multiple clones of repo locally, all worktrees share the same git object database and repository history
- avoid issues with persisent data that is not tracked in git

```bash
# create a new worktree AND a new branch on it off the current branch (e.g. from dev)
git worktree add ../tt-inference-server-feature-x -b github-id/feature-branch-name

git worktree add ../tt-inference-server-remove-lm-eval-cuda -b tstesco/remove-lm-eval-cuda


# go to the worktree + branch
cd ../tt-inference-server-feature-x

# or just open it in cursor
cursor ../tt-inference-server-feature-x

# remove worktree after completed work on the branch
git worktree remove ../branch-dir
```

For example:
```
├── tt-inference-server/           # main worktree
├── tt-inference-server-feature-x/ # additional worktree
├── tt-inference-server-fix-y/     # additional worktree
└── tt-inference-server-hotfix/    # additional worktree
```

### manage persistent data
```bash
# copy any files you dont want edit separately from main repo
cp -rf ../tt-inference-server/.env ./
# make symlink if you want to use and edit the main repo data
ln -s ../tt-inference-server/persistent_volume ./persistent_volume
```
