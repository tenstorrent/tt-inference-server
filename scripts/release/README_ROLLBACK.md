# Release automation — rollback runbook

How to undo the side effects of a release-automation run (`.github/workflows/release-automation.yml`)
or a manual catalogue backfill when a run **fails partway** or **produces bad output**.

## Principle
- **Undo in reverse order of creation** (last thing first), and only what the run actually created.
- **Reversibles** (git commit/tag, post-release branch, PR) → undo freely, they're fully in your control.
- **The published image** is the careful one: **overwrite if it may have been pulled/consumed; delete only if it's brand-new and unused.**
- Workflow **artifacts** (the uploaded zips) auto-expire — ignore them.

## Prerequisites (auth)
- **git**: push access to the release branch + permission to delete/move the tag (a protected branch may forbid force-reset — use `revert` then).
- **gh**: authenticated with `pull-requests: write` (to close the PR).
- **crane**: `crane auth login ghcr.io` with a token that has, on **tt-inference-server**, `write:packages` (overwrite) and — only for the delete option — `delete:packages`; and `read:packages` on **tt-shield** (to re-copy the source image).

## Setup (fill these in for the run you are undoing)
```bash
REL_BRANCH=stable
VERSION=0.21.0
SUFFIX=                                   # empty for a real release; -temp/-test while testing
TAG=v$VERSION$SUFFIX
POST_BRANCH=post-release-v$VERSION$SUFFIX
# vLLM release image (adjust repo for the family you published):
REPO=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64
IMG=$REPO:$VERSION-<ttm>-<vllm>            # <ttm>,<vllm> = the release commits
SRC=ghcr.io/tenstorrent/tt-shield/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:<full-tag>   # source that was copied
```

## Find the identifiers
```bash
git fetch origin --tags --prune

# release-branch bot commit (the "v<version>..." commit by github-actions[bot]):
git log -1 --oneline origin/$REL_BRANCH
git rev-parse "origin/$REL_BRANCH~1"       # the commit BEFORE it (reset target)

# tag exists?
git ls-remote origin "refs/tags/$TAG"

# post-release branch + open PR:
git ls-remote origin "refs/heads/$POST_BRANCH"
gh pr list --repo tenstorrent/tt-inference-server --head "$POST_BRANCH" --state open --json number,url

# image present + its current digest:
crane manifest "$IMG" >/dev/null 2>&1 && crane digest "$IMG" || echo "image not published"
```

## Rollback by side effect (least → most serious)

### 🟢 Low — throwaway, delete freely
```bash
# Draft PR (post-release -> main)
gh pr close <PR_NUMBER> --repo tenstorrent/tt-inference-server --delete-branch

# Post-release branch (if not already deleted by --delete-branch)
git push origin :"$POST_BRANCH"
```

### 🟡 Medium — public but yours to control
```bash
# Tag
git push origin :"refs/tags/$TAG"          # delete remote tag
git tag -d "$TAG" 2>/dev/null || true      # delete local tag

# Release-branch bot commit — choose ONE:
git revert <BOT_SHA> && git push origin HEAD:$REL_BRANCH                 # safe, keeps history
# or, if you may force-push and want it gone:
git push origin --force-with-lease="$REL_BRANCH:<BOT_SHA>" <SHA_BEFORE_BOT>:$REL_BRANCH
```

### 🔴 High — the published / re-baked image
```bash
# A) Undo just the re-bake (re-point tag to the un-baked source copy):
crane copy "$SRC" "$IMG"

# B) Restore from the pre-mutation snapshot (if taken before re-bake):
crane copy "$REPO:$VERSION-<ttm>-<vllm>-precatalogfix" "$IMG"

# C) Replace with corrected content (re-run the correct re-bake, or copy a good image):
crane copy <good-image> "$IMG"

# D) Fully unpublish (ONLY if nothing pulled it yet):
crane delete "$IMG"                        # if GHCR rejects, use the API:
PKG=tt-inference-server%2Fvllm-tt-metal-src-release-ubuntu-22.04-amd64
VID=$(gh api "/orgs/tenstorrent/packages/container/$PKG/versions" --paginate \
        --jq ".[] | select(.metadata.container.tags[]? == \"$VERSION-<ttm>-<vllm>\") | .id")
gh api --method DELETE "/orgs/tenstorrent/packages/container/$PKG/versions/$VID"
```
⚠️ **In use → overwrite (A/B/C). Brand-new & unused → delete (D).** Deleting a consumed tag breaks pullers.

### ⚪ None — workflow artifacts
The uploaded zips (`release-automation-generated-changes`, `release-artifacts-v<version>`) auto-expire; nothing to do.

## Where it failed → what to undo
| Run failed at | Already created → undo |
|---|---|
| guard / checkout / inputs / validate-commits | nothing pushed → just re-run (local only) |
| promote / 1b / export / helm | local working tree only → `git checkout -- .` or re-run |
| git push / tag | commit + tag → 🟡 |
| crane copy / re-bake / verify | + image → 🔴 (then 🟡) |
| post-release branch / commit / PR | + branch + PR → 🟢 (then 🔴/🟡 as needed) |

## Full-abort order (reverse of creation)
1. Close the PR 🟢
2. Delete the post-release branch 🟢
3. Fix or remove the image 🔴 (overwrite if consumed; delete only if unused)
4. Delete the tag 🟡
5. Revert/reset the release-branch commit 🟡

## Other published images (media / forge)
A multi-family release also publishes:
```
ghcr.io/tenstorrent/tt-media-inference-server:$VERSION-<ttm>
ghcr.io/tenstorrent/tt-media-inference-server-forge:$VERSION-<ttm>
```
Roll them back the **same way** as the 🔴 image steps (overwrite from source, or delete if unused).
Note: media/forge are **not** re-baked (they bake no catalogue), so there is no `-precatalogfix`
snapshot and no re-bake to undo — only the `crane copy` publish.

## Verify after rollback
```bash
git ls-remote origin "refs/tags/$TAG"                     # gone if tag deleted
git ls-remote origin "refs/heads/$POST_BRANCH"            # gone if branch deleted
gh pr view <PR_NUMBER> --repo tenstorrent/tt-inference-server --json state   # "CLOSED"
git log -1 --oneline origin/$REL_BRANCH                   # bot commit reverted/removed
# image restored to correct content (if overwritten):
crane export "$IMG" - | tar -xO home/container_app_user/model_specs/model_spec.json \
  | python3 -c "import json,sys; print('baked release_version =', json.load(sys.stdin)['release_version'])"
```

## Cautions
- **Never delete an image that may already be in use** — overwrite the tag instead (pullers converge on the good content after a fresh `docker pull`).
- **Force-reset on a shared branch** (`stable`) affects everyone — prefer `git revert`; coordinate if you must reset, and expect branch protection to block it.
- Re-bake (`crane append`) **drops the buildkit attestation** by design — expected, not a fault.
- **Snapshot before you mutate**: take `crane copy "$IMG" "$IMG-precatalogfix"` before any re-bake so option (B) is a one-liner.
- Deleting registry versions needs `delete:packages` (may require org admin).

## Related
Manual catalogue backfill (its own backup/RC/promote + rollback): see `scripts/release/MANUAL_CATALOG_REBAKE.md`.
