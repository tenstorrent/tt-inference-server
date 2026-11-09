---
name: build-tts-image
description: Build and push the TTS blaze media-server Docker image (tt-media-server/Dockerfile.blaze) from a nanicicTT/nanicic-TTS-llm-engine commit and a tt-inference-server commit, then commit a build record. Use when the user asks to build a TTS image, a new blaze/inworld image, or to rebuild with the latest nanicic engine.
disable-model-invocation: true
argument-hint: "[engine=<sha|branch>] [server=<sha|branch>] [tag=<name>]"
---

# Build the TTS blaze image

Builds `ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-blaze:<tag>`
from two commits, pushes it, verifies the registry digest, and commits a row to
`tt-media-server/docs/tts_image_builds.md`.

All commands run **on the build host, in the tt-inference-server repo root**.
If this session runs on a Mac that drives the host over herdr, wrap each one in
`herdr-ssh '...'`. The host needs GitHub access (gh token in
`~/.config/gh/hosts.yml` or `GITHUB_TOKEN`) and `docker login ghcr.io`.

The script is `.claude/skills/build-tts-image/build_tts_image.sh`
(call it `$S` below).

## 1. Collect the inputs

Take any of these from the invocation arguments first. Ask only for the ones
not given, in **one** AskUserQuestion call, each with the default as the first,
recommended option:

| Input | Default when not given |
|---|---|
| Engine commit (nanicic fork) | latest commit of `nanicicTT/TTS_Scheduler` |
| Server commit (tt-inference-server) | latest commit of `ljovanovic/tts_stable_branch` (local tip if it is ahead of origin) |
| Image tag | chosen from the changes, see step 3 |

A commit can be a SHA or a branch name. If the user answers "latest" or picks
the default, pass nothing for that input.

## 2. Resolve

```bash
$S resolve [--engine <ref>] [--server <ref>] [--tag <tag>]
```

It changes nothing. It prints `ENGINE=`, `SERVER=` (full SHAs) with their
subjects, the previous recorded build, `TAG_EXISTS=` when a tag was given,
`UNCOMMITTED_SERVER_FILES=`, and the commits on each side since the previous
build.

- A failed fetch means the host has no GitHub access. Stop and tell the user;
  do not build from stale state.
- If `UNCOMMITTED_SERVER_FILES` is not 0, tell the user those edits will NOT
  be in the image (it builds from a clean checkout of the server commit) and
  ask whether to continue or commit them first.
- If the engine list has a `DROPPED` section, those commits were in the
  previous image and will not be in this one (typical when the last build
  used a branch like `ljovanovic/tts-decoder-batch-env` and this one uses the
  `TTS_Scheduler` tip). Name them to the user and ask whether to continue.
- `TAG_EXISTS=unknown` means the registry could not be checked (offline or
  not logged in). Treat it as taken: fix access or pick another tag.

## 3. Pick the tag (only when the user gave none)

Read both change lists from step 2 and name the image after what is new:

- Format `tts_<slug>`: lowercase, `[a-z0-9_]`, at most ~40 chars, e.g.
  `tts_traffic_payload_capture`, `tts_sched_deadline_fix_sampling_knobs`.
- Name the one or two most significant changes. Prefer the engine change when
  both sides changed. If nothing changed, use `tts_rebuild_<yyyymmdd>`.
- Re-run `resolve --tag <tag>`. If `TAG_EXISTS` is `yes` or `unknown`, append `_2`, `_3`, ...
  until free. Never overwrite an existing tag you picked yourself.
- If the **user** gave a tag that exists, ask before overwriting it, and offer
  to keep the old image as `<tag>_prev_<yyyymmdd>` first
  (`docker pull`, `docker tag`, `docker push`).

Show the user the final engine SHA, server SHA and tag in one line before
starting; do not wait for confirmation unless something above needed it.

## 4. Build and push

The build takes 20-60 min. Start it detached and poll the log:

```bash
nohup $S build --engine <ENGINE> --server <SERVER> --tag <TAG> \
  > /tmp/build-tts-image-<TAG>.out 2>&1 &
```

The full log is `../tts-image-builds/<TAG>.log` (next to the repo). The last
line starts with `RESULT:`:

- `RESULT: SUCCESS tag=... digest=... engine=... server=...`: go to step 5.
- `RESULT: BUILD_FAILED`: show the error lines the log prints above it.
  The script already retried network failures (3 tries, 5 min apart).
- `RESULT: PUSH_FAILED`: the image is built locally; retry with
  `docker push` once, then report.

Wait with the session-wait skill (`herdr-wait`) if it is available instead of
polling in a loop.

## 5. Commit the build record

Only after `RESULT: SUCCESS`. Append one row to
`tt-media-server/docs/tts_image_builds.md`, using full SHAs and the digest from
the RESULT line:

```
| <yyyy-mm-dd> | `<tag>` | `<engine sha>` | `<server sha>` | `<digest>` | <short summary of what is new; no pipes> |
```

Commit that file alone on `ljovanovic/tts_stable_branch`, without staging
anything else:

```bash
git commit -m "tts: record image build <tag>" -m "<engine/server summary>" \
  -- tt-media-server/docs/tts_image_builds.md
```

If the repo is on another branch, ask before switching. Do not push the commit
unless asked.

## 6. Report

Tell the user the tag, the digest, the engine and server commits (short SHA +
subject), what changed since the previous build, and the record commit.
