# Patches for pinned eval repositories

`workflows/workflow_venvs.py` pins each third-party eval repository to a revision and discards local edits on
every run. Setting `TT_PINNED_REPO_PATCH_<DIRNAME>` to a patch file applies it on top of the pinned revision; a
relative path is taken from this repository's root. Nothing here is applied unless that variable is set.

| Patch | Applies to | What it does | How to use |
|---|---|---|---|
| `harbor_tau3_preserve_reasoning.patch` | Harbor (`tenstorrent/harbor`, branch `tt-inference-server`) | The tau3 agent sends an assistant turn's `reasoning_content` back with the history when `TAU3_AGENT_PRESERVE_REASONING=1`, as Kimi K3 / K3.1 expect. Off by default, so other models' histories are unchanged. | `TT_PINNED_REPO_PATCH_HARBOR=reference_config/evals/patches/harbor_tau3_preserve_reasoning.patch TAU3_AGENT_PRESERVE_REASONING=1` |
