# Release process

This document gives the step by step instructions for making a release using the automation - - **Release Automation (promote specs, docs & docker images)**.

The release process is run from the GitHub Actions, by invoking the `release-automation.yml` pipeline 

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

## Git Workflow Diagram

Follow the git workflow for release described below in the diagram and step by step instructions below:

![../../docs/ttis-git-workflows-2026-02-10](../../docs/ttis-git-workflows-2026-02-10.png)


## Step 1: update `models-ci-config.json` on stable branch

Within the `models-ci-config.json` file, update which models and devices should belong to the upcoming release. 
Remove all the entries from the release list, which are not going to be actually released.

## Step 2: update `VERSION` file on stable branch

Bump the version of the `VERSION` file (major/minor/patch syntax).

## Step 3: run 'Release Automation (promote specs, docs & docker images)' pipeline

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

Once dispatched, the pipeline runs the following sub-steps in order. Most are automatic; it stops (fails) early if any validation does not hold, so nothing is committed or published on a bad input.

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
Creates and pushes the release tag on `stable` (e.g. `v0.21.0`). This is the tag the GitHub Release Object targets in Step 5.

### 4.11 Build the release-artifacts zip
Downloads the per-model/device workflow artifacts from the tt-shield run and packs them into `v<version>-release_artifacts.zip`, uploaded as a pipeline artifact (this is what you download in Step 7).

### 4.12 Resolve source images and map to release images
Finds the `vllm` / `media` / `forge` dev images the tt-shield run built, and computes the destination release image tags. Example:
- source: `ghcr.io/tenstorrent/tt-shield/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.21.0-de59f8a…-6b4a3a7-<jobid>`
- target: `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.21.0-de59f8a-6b4a3a7`

### 4.13 Publish the images (crane)
Logs in to `ghcr.io` and copies each release-scoped image from the tt-shield registry to the tt-inference-server release registry (only the engine families this release actually ships).

### 4.14 Re-bake the catalogue into the vLLM image
`crane copy` only re-labels; the source image was built before this release promoted its specs, so its embedded `model_spec.json` is the previous catalogue. This step appends the freshly-promoted catalogue as a new top layer so the published vLLM image serves **this** release's specs (avoids "No model spec found"). vLLM only — media/forge bake no catalogue.

### 4.15 Verify the published image
Pulls the baked catalogue back out of the published vLLM image and confirms every spec pinned to that image resolves. Fails the release if anything is missing.

### 4.16 Create the post-release branch and draft PR
Branches `post-release-v<version>` from `main`, carries this release's `VERSION` + `models-ci-config.json` onto it, regenerates the specs/docs/Helm there, commits and pushes the branch, then opens a **draft PR** into `main`. The PR body is pre-filled with the model-spec update table and the list of promoted images — the notes used in Steps 5–6.

## Step 5: Create GitHub RELEASE Object

We need to create new Draft Release Object `vx.x.x` targeting the automatic tag created from the stable branch. We create new Release Object once the Release Manager and Customer Success team finish verification of the automatically created PR, which is filled with all the required notes required for the Release.

## Step 6: copy paste Release notes from PR body

Release Notes must be added describing new supported engine features.
This is done by copying the PR body.

After that we save Release Object in Draft status.

## Step 7: Downloading workflow artifacts and assets upload to Release Object

The automation pipeline will provide us with release artifacts in the Assets table, which are packed and prepared for the upload on the Release Object level.
We need to download those locally, and upload that same zipped file as a packaged artifact on Release Object level.

## Step 8: Release Object publishing

At the end, we change the status of the Release Object to `Published` and mark the Release as the latest one.

## Step 9: Update Release Zoo

From the `tt-shield` Actions tab we need to run the `"Update Release Zoo"` action so the page on Models Dashboard is being refreshed.

https://github.com/tenstorrent/tt-shield/actions/workflows/update-release-zoo.yml
