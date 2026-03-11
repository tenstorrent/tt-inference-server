# Release process

This document describes the release process for `tt-inference-server` using the
current branch model shown in the git workflow diagram.

## What is a release?

A release is defined by:
- a commit of tt-inference-server
- a diff to `workflows/model_spec.py` and validation of that diff.
- pre-built and validated docker images with tt-metal, vLLM, and other libaries required to run inference
- automatically generated documentation pointing to usage of the docker images

A release is prepared from the `stable` branch at a chosen
`tt-inference-server` commit. Most releases use one Nightly Models CI run to
populate and validate the `workflows/model_spec.py` updates, or simply update these  manually from results of interactive development. plus one Release
Models CI run to validate the final release Docker images.

A single release can still include multiple `tt_metal_commit` values across
different model and hardware combinations, and therefore multiple release
Docker images, as long as those combinations are captured in
`workflows/model_spec.py` and validated through the release process below.

Where a ModelSpecTemplate has an older `release_version` than the current repo VERSION
the user can be warned. In future the specific release code could be checked out and ran 
to give a seamless transition to running the latest released version and Docker image for any previously released version without needing to rebuild and re-test a given commit combination.

## Branch roles

- `main`: active development trunk branch.
- `stable`: release staging branch cut from `main`.
- `<namett>/hot-fix-<description>`: hot-fix branch used when a fix must land on
  `main` and also be cherry-picked into `stable` or a patch branch.
- `vx.y.z`: release branch cut from `stable`. Repo default branch, updated after every release.
- `patch-vx.y.z`: staging branch for patching an older shipped release.

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

The pre-release steps occur on `stable` branch only. This means that the automatic diffs to model_spec.py need to be added to `main` in Post-release.

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
The CI-driven flow stamps `release_version` on CI-updated
`ModelSpecTemplate` records, and `--output-only` stamps `release_version` on
manually edited templates whose `tt_metal_commit` changed. When CI references
are available they are attached to the changed `ModelSpecTemplate` records;
otherwise those entries render as `N/A` in the markdown output.

When using `--models-ci-run-id` in this step, pass the Nightly Models CI
workflow run ID.

```bash
# process a specific Nightly Models CI workflow run_id to update passing models
python3 scripts/release/update_model_spec.py --models-ci-run-id 19339722549
```

### [optional] step 2B: manual changes to model_spec.py

Before starting Step 2, `workflows/model_spec.py` should be clean relative to
`HEAD` on `stable`. After the CI-driven update has run, you may then make
manual edits before running `--output-only`.

Manually edit `model_spec.py`. After changes are added, re-run the helper to
stamp `release_version` on templates whose `tt_metal_commit` changed, then
re-generate the Model Support documentation and `default_model_spec.json`:

```bash
python3 scripts/release/update_model_spec.py --output-only
```

#### outputs

- `workflows/model_spec.py`: Updates `ModelSpecTemplate`:
  - `tt_metal_commit`: from Models CI run id, or manual edits.
  - `release_version`: where tt_metal_commit has been changed.
- `default_model_spec.json`: all model specs fully expanded from the ModelSpecTemplates in `workflows/model_spec.py`
- `release_logs/v{VERSION}/pre_release_models_diff.json`: summary of changed `ModelSpecTemplate` records derived from the git diff of `workflows/model_spec.py`, with CI links when available. This is used for post-release as well. and in `scripts/release/post_release.py`.
- `release_logs/v{VERSION}/pre_release_models_diff.md`: This markdown version of `pre_release_models_diff.json` is used by `scripts/release/generate_release_notes.py`.
- `release_logs/v{VERSION}/models_ci_all_results_*.json`: this has full Models CI parsed data for analysis
- `release_logs/v{VERSION}/models_ci_last_good_*.json`: this may be used downstream for release process
- `docs/model_support/*.md`: regenerated model support documentation (model type pages, hardware pages, individual model pages)
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

When using `--models-ci-run-id` in this step, pass the Release Models CI
workflow run ID from Step 4.

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

Re-run `scripts/release/generate_release_artifacts.py` after this manual image build
step it should pick up the manually added docker images.

### Step 6: Make release

```bash
git checkout stable
git checkout -b v${VERSION}
git push -u origin v${VERSION}
```

Make [new release on GitHub repo](https://github.com/tenstorrent/tt-inference-server/releases/new)

Use branch `vx.y.z` HEAD as the GitHub release tag target.
Use `release_logs/v{VERSION}/release_notes_v{VERSION}.md` as the GitHub release
body, and fill in any sections that were left blank by the generator.

## Post-release

After the release branch is completed, prepare the next release as development state on `main`.

### step 7: make post-release branch

```bash
git checkout main
git pull
git checkout -b post-release-vx.y.z
```

### step 8: run post_release.py

The script uses the pre-release record of the model diffs from
`release_logs/v{VERSION}/pre_release_models_diff.json` to determine which
released template updates from `stable` should be carried back onto `main`.
For each matching template on `main`, commit or status fields are only updated
when `main` still has the released starting value from the pre-release diff.
If `main` has already been changed manually, that field update is discarded and
reported in the PR draft. The template `release_version` is only updated when
the released `tt_metal_commit` is also applied; if `main` has already diverged
on `tt_metal_commit`, leave `release_version` unchanged.

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

### Step 9: make release branch vx.y.z the default branch

On GitHub repo select release branch vx.y.z (e.g. v0.11.0) as the default branch.
This is so new users are seamlessly directed to the latest and greatest released code.
This will in turn direct them to correct and updated documentation and images with pre-built tt-metal artifacts required to serve inference and reproduce results from release testing.

## Hot-fixes

If a fix is needed in the current release:

1. Create a hot-fix branch from `main`.
2. PR the fix back to `main`.
3. Cherry-pick the merged fix into `stable` if it must land in the current release (see Step 1B).
4. If the issue affects an older shipped release, cherry-pick into the
   appropriate `patch-vx.y.z` branch instead and see section below.

This keeps the fix anchored in the trunk branch while still allowing a
release or patch branch to pick it up intentionally.

If the fix cannot be done on `main` the fix can be PR direct to `stable` or `patch-vx.y.z`

## Patching older releases

Older shipped versions are patched from dedicated patch branches, not from
`stable`.

Typical flow:

1. Create or update `patch-vx.y.z` from the shipped release branch or tag.
2. Cherry-pick the required fix or fixes.
3. Increment VERSION patch number.
4. Re-run artifact promotion and any required Docker builds.
5. Create the patch tag and GitHub release.

If there are unresolvable conflicts when cherry-picking from `main`, keep the
patch fix isolated to the patch branch and document the divergence clearly in
release notes.
