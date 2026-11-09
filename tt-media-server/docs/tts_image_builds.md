# TTS blaze image builds

One row per image pushed by the `build-tts-image` skill
(`.claude/skills/build-tts-image`). The skill reads the **last row** to decide
which engine and server commits are new since the previous build, so keep the
format: full SHAs, one build per line, newest at the bottom, no `|` in notes.

Image: `ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-blaze:<tag>`
Engine: `nanicicTT/nanicic-TTS-llm-engine` · Server: `tenstorrent/tt-inference-server`

| date | tag | engine | server | digest | notes |
|---|---|---|---|---|---|
| 2026-11-03 | `tts_inworld_realtime` | `2a6cf4264d5d0c15fa3e0eb14b5e7f576f8d3376` | `8a18f03ed61c27991bb21c5523f31e4ddd275297` | `sha256:04c76f53784fea46b7952d36dc3d99a7141b5de8fe50fa8a69c7026a2c708093` | built by hand before this skill; engine = nanicic 4b55599 + 2 commits on ljovanovic/tts-decoder-batch-env; server tree had uncommitted changes (sampling knobs) |
