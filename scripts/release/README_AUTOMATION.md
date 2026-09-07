# Release process

This document gives the step by step instructions for making a release using the automation — **Release Automation (promote specs, docs, helm & docker images)**.

The release process runs from GitHub Actions in two workflows: **`release-automation.yml`** (dispatched manually — promotes specs, publishes images, opens the post-release PR, and creates the draft Release), and **`publish-release.yml`** (fires automatically when that PR is merged — fills the Release notes from the PR body and publishes it).

## Summary Diagram

![release-summary-2025-08-14-1106.png](release-summary-2025-08-14-1106.png)

## pre-requisite requirements
permissions requirement:
- Download only
    - [GitHub Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) (PAT)
        - Read access to tt-shield repo.
- Full release:
    - [GitHub Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) (PAT)
        - Read access to tt-shield repo.
        - Write access to tt-inference-server packages
    - crane CLI (https://github.com/google/go-containerregistry/tree/main/cmd/crane)

> With the automation these are the **pipeline's** requirements, not yours to run locally: the workflow installs `crane` and publishes the images itself, using the PAT stored as the `TMP_VCANKOVIC_SHIELD_CRANE_PAT` repo secret. You only need a local PAT / crane for the **manual fallback** in `README_MANUAL.md`.

## Git Workflow Diagram

Follow the git workflow for release described below in the diagram and step by step instructions below:

![../../docs/ttis-git-workflows-2026-02-10](../../docs/ttis-git-workflows-2026-02-10.png)


## Step 1: update `models-ci-config.json` on stable branch

Within the `models-ci-config.json` file, update which models and devices should belong to the upcoming release. 
Remove all the entries from the release list, which are not going to be actually released.

## Step 2: update `VERSION` file on stable branch

Bump the version of the `VERSION` file (major/minor/patch syntax).

## Step 3: run 'Release Automation (promote specs, docs, helm & docker images)' pipeline

We need to run the pipeline by selecting the `stable` branch. In case we select some other branch pipeline execution will not be allowed.
**stable** branch is the main source of truth for the release.

When `stable` branch is selected we need to provide input values for the following:

`Release version`: e.g. 0.21.0  - we provide this value in order to have the version embedded within the pipeline title execution:

`tt-shield Release run ID`: - we need to find th shield release pipeline ID

`tt-metal commit SHA`: 7 chars commit sha (we intentionally want to compare the metall commith sha as an input with the SHA that will be retrieved through the runId)

`vLLM commit SHA`: 7 chars commit sha (we intentionally want to compare the metall commith sha as an input with the SHA that will be retrieved through the runId).

Other two fields are optional and mainly used for testing purposes.

Once we have all 4 mandatory fields populated we run the pipeline.

## Step 4: execution of the automation pipeline

Once dispatched, the pipeline runs the following sub-steps in order — **all of Step 4 is automated**. It stops (fails) early if any validation does not hold, so nothing is committed or published on a bad input.

### 4.1 Branch guard
Refuses to run unless the branch is `stable` (the sanctioned release branch). Any other branch fails immediately, before checkout.

### 4.2 Environment setup
Checks out `stable`, sets up Python, and installs the release dependencies plus `helm-docs`.

### 4.3 Resolve & validate inputs
Reads the `VERSION` file as the source of truth and validates the inputs against it:
- the `Release version` input must equal the `VERSION` file (e.g. input `0.21.0` == `VERSION` `0.21.0`);
- `tt-shield Release run ID` is required;
- `vLLM commit SHA` is required when the release ships any vLLM model.

A mismatch (e.g. input `0.20.0` but `VERSION` says `0.21.0`) fails the run here.

### 4.4 Validate commits against the tt-shield run
Reads the images the given tt-shield run actually built and confirms the `tt-metal` / `vLLM` commit inputs match them (the 7-char input must be a prefix of the built 40-char SHA). Guards against a typo'd commit that would pin the wrong build. Runs before any change is made.

### 4.5 Promote dev → prod specs
Runs `promote_dev_spec_to_prod.py`: copies the release-listed model specs from `dev` to `prod`, pinning this release's `version`, `tt_metal_commit` and `vllm_commit`.

### 4.6 Duplicate guard
Asserts the promotion did not create a **shadowed duplicate** (a stale prod block left behind for the same model+device). If found, the run fails so the dead block never reaches the docs/catalogue. (See the manual pre-flight in `README_MANUAL.md`.)

### 4.7 Regenerate docs and release JSON
Runs `export_model_spec.py` to rebuild `docs/model_support/**` and `release_model_spec.json` from the promoted `prod` catalogue.

### 4.8 Regenerate the Helm chart
Regenerates `charts/tt-inference-server/values.yaml` (helm generator) and the chart docs (`helm-docs`).

### 4.9 Commit + push the generated files to `stable`
Commits all generated changes (prod specs, docs, `release_model_spec.json`, Helm files) as `v<version>` and pushes them to `stable` (e.g. commit message `v0.21.0`).

### 4.10 Tag the release
Creates and pushes the release tag on `stable` (e.g. `v0.21.0`). This is the tag the draft Release (4.18) references.

### 4.11 Create the `v<version>` branch from stable
Cuts a `v<version>` branch from the release HEAD (the same commit the tag points to), so the released state is preserved as a branch as well as a tag. Uses `git branch` without switching, so later steps are unaffected; `stable` is left intact.

### 4.12 Build the release-artifacts zip
Downloads the per-model/device workflow artifacts from the tt-shield run and packs them into `v<version>-release_artifacts.zip` (kept on the runner). This same zip is attached to the Release automatically in 4.19 (it is also uploaded as a workflow artifact for the Actions UI).

### 4.13 Resolve source images and map to release images
Finds the `vllm` / `media` / `forge` dev images the tt-shield run built, and computes the destination release image tags. Example:
- source: `ghcr.io/tenstorrent/tt-shield/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.21.0-de59f8a…-6b4a3a7-<jobid>`
- target: `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.21.0-de59f8a-6b4a3a7`

### 4.14 Publish the images (crane)
Logs in to `ghcr.io` and copies each release-scoped image from the tt-shield registry to the tt-inference-server release registry (only the engine families this release actually ships).

### 4.15 Re-bake the catalogue into the vLLM image
`crane copy` only re-labels; the source image was built before this release promoted its specs, so its embedded `model_spec.json` is the previous catalogue. This step appends the freshly-promoted catalogue as a new top layer so the published vLLM image serves **this** release's specs (avoids "No model spec found"). vLLM only — media/forge bake no catalogue.

### 4.16 Verify the published image
Pulls the baked catalogue back out of the published vLLM image and confirms every spec pinned to that image resolves. Fails the release if anything is missing.

### 4.17 Create the post-release branch and draft PR
Branches `post-release-v<version>` from `main`, carries this release's `VERSION` + `models-ci-config.json` onto it, regenerates the specs/docs/Helm there, commits and pushes the branch, then opens a **draft PR** into `main`. The PR body is pre-filled with the model-spec update table and the list of promoted images — this is the body a reviewer edits and which, once merged, becomes the Release notes (Steps 5–7).

### 4.18 Create the draft GitHub Release
Creates a **draft** GitHub Release named `v<version>`, referencing the tag from 4.10, with an **empty body**. (This automates the former manual "Create Release Object" step.) Skips if a Release for the tag already exists.

### 4.19 Upload the release-artifacts zip as a Release asset
Attaches `v<version>-release_artifacts.zip` (from 4.12) to the draft Release as an asset, uploaded **directly** so it is a single zip — not the double-zipped workflow artifact. (This automates the former manual download-and-upload step.)

> **After the pipeline:** the automation has already pushed the generated files + tag + `v<version>` branch on `stable`, published the images, opened a **draft PR** into `main`, and created a **draft Release `v<version>`** with the `v<version>-release_artifacts.zip` asset attached and an empty body. What remains is a human **review + merge** (Steps 5–6); **publishing (Step 7) is then automated** by a second workflow.

## Step 5: Review the auto-created PR and draft Release  *(manual)*

The Release Manager / Customer Success team review the automatically-created **draft PR** (`post-release-v<version>` → `main`) — pre-filled with the model-spec update table and the promoted-image list — and the **draft Release `v<version>`** (tag + asset already attached). Edit the PR body as needed; it becomes the Release notes.

## Step 6: Approve and merge the post-release PR  *(manual)*

A reviewer **approves and merges** the post-release PR into `main` (enforced by branch protection on `main` requiring a review). This is the go/no-go gate for the release — and **merging it triggers Step 7 automatically.**

## Step 7: Publish the Release  *(automated — `publish-release.yml`)*

Merging the post-release PR triggers **`.github/workflows/publish-release.yml`**, which:
- writes the **final (merged) PR body** into the Release notes of the draft `v<version>`, and
- flips the Release from **draft → published** and marks it **latest**.

The tag and the `v<version>-release_artifacts.zip` asset are already in place from Step 4, so nothing else is attached — no manual publishing needed.

Notes / caveats:
- The trigger is filtered on the **`VERSION`** file, so it fires only when the merged PR changed `VERSION` (every real forward release does). It will **not** fire on a same-version re-release.
- It publishes with the `TMP_VCANKOVIC_SHIELD_CRANE_PAT` PAT (needs `contents: write`), so the `release: published` event **cascades** — enabling a future auto "Update Release Zoo".
- If the draft Release is missing (e.g. the pipeline's create-Release step failed), it **fails loudly** rather than creating one.

## Step 8: Update Release Zoo  *(manual for now)*

From the `tt-shield` Actions tab, run the `"Update Release Zoo"` action so the page on the Models Dashboard is refreshed.

https://github.com/tenstorrent/tt-shield/actions/workflows/update-release-zoo.yml

> Because Step 7 publishes with a PAT, the `release: published` event cascades — so this step could later be triggered automatically by a `on: release: [published]` workflow. It is manual for now.
