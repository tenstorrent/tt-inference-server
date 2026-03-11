# Release process

This document describes the release process for `tt-inference-server` using the
current branch model shown in the git workflow diagram.

## What is a release?

A release is defined by:
- a commit of tt-inference-server
- a diff to `workflows/model_spec.py` and validation of that diff.
- pre-built and validated docker images with tt-metal, vLLM, and other libaries required to run inference
- automatically generated documentation pointing to usage of the docker images

A release is typically done for a single tt-metal commit from it's own `stable` branch, however a release can support multiple commits for different model-hardware combinations and corresponding different Docker images.
The validation is typically done for a single commit however because of how testing
 is done in Models CI.

## Branch roles

- `main`: active development trunk branch.
- `stable`: release staging branch cut from `main`.
- `<namett>/hot-fix-<description>`: hot-fix branch used when a fix must land on
  `main` and also be cherry-picked into `stable` or a patch branch.
- `vX.Y.Z`: release branch cut from `stable`. Repo default branch, updated after every release.
- `patch-vX.Y.Z`: staging branch for patching an older shipped release.

## Git workflow diagram

Follow the git workflow for release described in the diagram below:

![../../docs/ttis-git-workflows-2026-03-09](../../docs/ttis-git-workflows-2026-03-09.png)


## Pre-requisites

The release process can be run locally on a laptop or on a remote server.
Building carried-forward Docker images is still better on a remote machine with
high CPU and RAM because the build flow can trigger multiple Docker builds.

Permissions:
- Download-only flows:
  - [GitHub Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
    with read access to `tt-shield`.
- Full release flows:
  - [GitHub Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
    with read access to `tt-shield` and write access to `tt-inference-server`
    packages.
  - `crane` CLI: <https://github.com/google/go-containerregistry/tree/main/cmd/crane>

Authenticate locally:

```bash
export GH_ID=tstescoTT
export GH_PAT=ghp_xxxxxxx
crane auth login ghcr.io -u "${GH_ID}" -p "${GH_PAT}"
# optional if you only need Docker CLI pulls
docker login ghcr.io -u "${GH_ID}" -p "${GH_PAT}"
```

The operational release gate is a passing Models CI. If any regression
is accepted into a release, capture it explicitly in the release notes known issues section.

## Pre-release

### Step 1: cut new `stable` branch

Release engineer will use pre-defined tt-inference-server commit, this can be from results of nightly Models CI run for example, or a specific version required for a specific feature, or before a feature was added.
```bash
git checkout main
git pull
# Make local stable point to tip of main with chosen commit
git branch -f stable <chosen tt-inference-server commit>
git checkout stable
# 
```

### [optional] Step 1B: cherry-pick any commits needed

If needed:
```bash
git cherry-pick <needed commits>
```

### Step 2: update model_spec.py

Add changes to `model_spec.py` for the chosen commit sets. These can be determined automatically using the nightly Models CI outputs, or done manually.

Both the CI-driven flow and the manual `--output-only` flow generate the
pre-release diff artifacts from the git diff of `workflows/model_spec.py`.
When CI references are available they are attached to the changed
`ModelSpecTemplate` records; otherwise those entries render as `N/A` in the
markdown output.

```bash
# process a specific workflow run_id to update passing models
python3 scripts/release/update_model_spec.py --models-ci-run-id 19339722549
```

### [optional] step 2B: manual changes to model_spec.py

Manually edit `model_spec.py`. After changes are added, re-generate the Model Support documentation and `default_model_spec.json`:

```bash
python3 scripts/release/update_model_spec.py --output-only
```

#### outputs

- `workflows/model_spec.py`: diff has updates from Models CI most recent passing runs and any manual `ModelSpecTemplate` edits
- `default_model_spec.json`: all model specs fully expanded from the ModelSpecTemplates in `workflows/model_spec.py`
- `release_logs/v{VERSION}/pre_release_models_diff.md`: summary of changed `ModelSpecTemplate` records derived from the git diff of `workflows/model_spec.py`, with CI links when available
- `release_logs/v{VERSION}/pre_release_models_diff.json`: programmatic version of the same changed-template records used to render `pre_release_models_diff.md`. This is used for post-release as well.
- `release_logs/v{VERSION}/models_ci_all_results_*.json`: this has full Models CI parsed data for analysis
- `release_logs/v{VERSION}/models_ci_last_good_*.json`: this may be used downstream for release process
- `docs/model_support/`: regenerated model support documentation (model type pages, hardware pages, individual model pages)
- `README.md`: updates to the `Model Support` section (links to docs/model_support/)

### Step 3: force update `stable` branch

```bash
git push --force-with-lease origin stable
```

## Release

### Step 4: Start `Release` Models CI GitHub Actions Workflow

https://github.com/tenstorrent/tt-shield/actions/workflows/release.yml

On failure, use Hot-Fix workflow or repeat Pre-release steps.
On success, continue Release steps.

### step 5: generate release artifacts

Log and move (via [crane](https://github.com/google/go-containerregistry/blob/main/cmd/crane/doc/crane.md)) the Docker images built within the Release workflow.
The two key release artifacts are:
- Docker images
- Repo model support documentation (already updated during pre-release)
- release notes

```bash
python3 scripts/release/generate_release_artifacts.py --models-ci-run-id 29339722549 --release
```

#### Outputs:
- `release_logs/v{VERSION}/release_artifacts_summary.json`
- `release_logs/v{VERSION}/release_artifacts_summary.md`
- `release_logs/v{VERSION}/models_ci_all_results_*.json`
- `release_logs/v{VERSION}/release_notes_v{VERSION}.md`

The `release_notes_v{VERSION}.md` file generated has sections, left blank if undefined below:
- Release title: tt-inference-server v{VERSION}
- Summary of Changes
- Recommended system software versions
- Known Issues
- Model and Hardware Support Diff: `release_logs/v{VERSION}/pre_release_models_diff.md`
- Performance
- Scale Out
- Deprecations and breaking changes
- Release Artifacts Summary: from `release_logs/v{VERSION}/release_artifacts_summary.md`
- Contributors: standard GitHub release
- Assets: standard GitHub release


### [optional] step 5B: build images for manually added models 

Ideally all released models have Models CI runs available. If manually added models still need containers, build the missing release
images:

```bash
python3 scripts/build_docker_images.py --push --release
```

### Step 6: Make release

```bash
git checkout stable
git checkout -b v${VERSION}
git push -u origin v${VERSION}
```

Make [new release on GitHub repo](https://github.com/tenstorrent/tt-inference-server/releases/new)


## Post-release

After the release branch is completed, prepare the next release as development state on `main`.

### step 7: make post-release branch

```bash
git checkout main
git pull
git checkout -b post-release-vx.y.z
```

### step 8: run post_release.py

The script uses the pre-release record of the model diffs from `release_logs/v{VERSION}/pre_release_models_diff.md`
to determine updates to model_spec.py where the `main` model spec has the starting 

```bash
python3 scripts/release/post_release.py --increment minor
```

This helper:

- increments `VERSION`
- updates `workflows/model_spec.py` using `release_logs/v{VERSION}/pre_release_models_diff.json`
- regenerates `default_model_spec.json` and model support docs
- writes `release_logs/post_release_pr.md`

Open a PR from the post-release branch back to `main`. That PR carries the next
development VERSION, any forward-looking `model_spec.py` commit updates, and the
regenerated docs for the next release window.

### Step 9: make release branch vX.Y.Z the default branch

On GitHub repo select release branch vX.Y.Z (e.g. v0.11.0) as the default branch.
This is so new users are seamlessly directed to the latest and greatest released code.
This will in turn direct them to correct and updated documentation and images with pre-built tt-metal artifacts required to serve inference and reproduce results from release testing.

## Hot-fixes

If a fix is needed in the current release:

1. Create a hot-fix branch from `main`.
2. PR the fix back to `main`.
3. Cherry-pick the merged fix into `stable` if it must land in the current release (see Step 1B).
4. If the issue affects an older shipped release, cherry-pick into the
   appropriate `patch-vX.Y.Z` branch instead and see section below.

This keeps the fix anchored in the trunk branch while still allowing a
release or patch branch to pick it up intentionally.

If the fix cannot be done on `main` the fix can be PR direct to `stable` or `patch-vX.Y.Z`

## Patching older releases

Older shipped versions are patched from dedicated patch branches, not from
`stable`.

Typical flow:

1. Create or update `patch-vX.Y.Z` from the shipped release branch or tag.
2. Cherry-pick the required fix or fixes.
3. Increment VERSION patch number.
4. Re-run artifact promotion and any required Docker builds.
5. Create the patch tag and GitHub release.

If there are unresolvable conflicts when cherry-picking from `main`, keep the
patch fix isolated to the patch branch and document the divergence clearly in
release notes.
