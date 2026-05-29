---
name: build-blaze-images
description: Trigger Tenstorrent Blaze media server and tt-metal upstream Docker image builds with gh. Use when the user asks to build Blaze and Metal Docker images, run the tt-shield media server image workflow, or build tt-metal images from the tt-llm-engine submodule reference.
disable-model-invocation: true
---

# Build Blaze Images

## Purpose

Use this skill to trigger both image builds:

- Blaze media server Docker image through `tenstorrent/tt-shield` workflow `on-dispatch-build-media-server.yml`.
- tt-metal upstream Docker images through `tenstorrent/tt-metal` workflow `upstream-tests.yaml`, using the `tt-metal` commit referenced by `tt-media-server/cpp_server/tt-llm-engine`.

## Workflow

1. Work from the `tt-inference-server` repository root.

2. Resolve the currently checked out `tt-inference-server` ref for the Blaze build:

   ```bash
   inference_server_ref="$(git branch --show-current)"
   if [ -z "$inference_server_ref" ]; then
     inference_server_ref="$(git rev-parse HEAD)"
   fi
   printf '%s\n' "$inference_server_ref"
   ```

   Use this value for the tt-shield workflow's `inference-server-git-ref` input. If the current branch has not been pushed to `tenstorrent/tt-inference-server`, stop and ask the user whether to push it or use a commit/ref that exists on GitHub.

3. Inspect the `tt-llm-engine` submodule and the nested `tt-metal` submodule:

   ```bash
   git submodule status
   cd tt-media-server/cpp_server/tt-llm-engine
   git submodule status
   cd tt-metal
   git show -s --format='%H%n%D%n%s%n%ci' HEAD
   git branch -r --contains HEAD
   ```

4. Pick the `tt-metal` workflow ref:

   - Prefer a remote branch whose HEAD exactly matches the nested `tt-metal` submodule commit.
   - If no branch points at the commit, use the exact commit SHA if GitHub Actions accepts it.
   - If multiple branches contain the commit, prefer the branch shown in `git show -s --format=%D HEAD` if it names an `origin/<branch>` HEAD; otherwise ask the user which branch to use.

5. Check workflow inputs before dispatching:

   ```bash
   gh workflow view on-dispatch-build-media-server.yml --repo tenstorrent/tt-shield --yaml
   gh workflow view upstream-tests.yaml --repo tenstorrent/tt-metal --yaml
   ```

6. Trigger the Blaze image build from the currently checked out `tt-inference-server` ref. The tt-shield workflow input option is `blaze-media-inference-server`; if the user says `blaze-media-server`, use this workflow option.

   ```bash
   gh workflow run on-dispatch-build-media-server.yml \
     --repo tenstorrent/tt-shield \
     -f inference-server-type=blaze-media-inference-server \
     -f inference-server-git-ref="$inference_server_ref"
   ```

7. Trigger the tt-metal image build with default workflow inputs and the selected `tt-metal` ref:

   ```bash
   gh workflow run upstream-tests.yaml \
     --repo tenstorrent/tt-metal \
     --ref '<tt-metal-branch-or-sha>'
   ```

8. Verify both runs and report URLs:

   ```bash
   gh run list --repo tenstorrent/tt-shield \
     --workflow on-dispatch-build-media-server.yml \
     --event workflow_dispatch \
     --limit 5 \
     --json databaseId,displayTitle,headBranch,headSha,status,conclusion,url,createdAt,workflowName

   gh run list --repo tenstorrent/tt-metal \
     --workflow upstream-tests.yaml \
     --branch '<tt-metal-branch>' \
     --event workflow_dispatch \
     --limit 5 \
     --json databaseId,displayTitle,headBranch,headSha,status,conclusion,url,createdAt,workflowName
   ```

## Reporting

In the final response, include:

- The Blaze workflow run URL, status, and server type.
- The Metal workflow run URL, status, branch or SHA, and the submodule commit used.
- Monitoring commands:

  ```bash
  gh run watch <shield-run-id> --repo tenstorrent/tt-shield
  gh run watch <metal-run-id> --repo tenstorrent/tt-metal
  gh run view <metal-run-id> --repo tenstorrent/tt-metal --log-failed
  ```

## Failure Investigation

If a run fails, collect more information with:

```bash
gh run view <run-id> --repo <owner/repo> --json status,conclusion,jobs,url
gh run view <run-id> --repo <owner/repo> --log-failed
```

For image publishing uncertainty, inspect the workflow job outputs and the `calculate-to-publish` job logs in `tt-metal`, because `upstream-tests.yaml` may publish only images whose tests passed unless `request-force-publish` is changed from its default.
