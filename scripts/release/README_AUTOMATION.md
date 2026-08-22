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


## Step 5: Create GitHub RELEASE Object

We need to create new Draft Release Object `vx.x.x` targeting the automatic tag created from the stable branch. We create new Release Object once the Release Manager and Customer Success team finish verification of the automatically created PR, which is filled with all the required Release Details.

## Step 6: Reference Tag
Set tag for a given Release Object, created in a previous step.

## Step 7: copy paste Release notes from PR body

Release Notes must be added describing new supported engine features.
This is done by copying the PR body.

After that we save Release Object in Draft status.

## Step 8: Downloading workflow artifacts and assets upload to Release Object

The automation pipeline will provide us with release artifacts in the Assets table, which are packed and prepared for the upload on the Release Object level.
We need to download those locally, and upload that same zipped file as a packaged artifact on Release Object level.

## Step 9: Release Object publishing

At the end, we change the status of the Release Object to `Published` and mark the Release as the latest one.

## Step 10: Update Release Zoo

From the `tt-shield` Actions tab we need to run the `"Update Release Zoo"` action so the page on Models Dashboard is being refreshed.

https://github.com/tenstorrent/tt-shield/actions/workflows/update-release-zoo.yml
