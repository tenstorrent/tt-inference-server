# MiniMax-H3 prebuilt start-up artifacts

A cold start of MiniMax-H3 on a Blackhole Galaxy compiles about 10,000 kernels and converts about 115 GB of weights. That takes 30 to 70 minutes. CI does both once and publishes the results, so a deployment loads them instead.

## The two artifacts

| Artifact | Content | Where it is published |
| --- | --- | --- |
| Kernel cache | The kernels tt-metal compiled during the CI run | Inside the server image, at `/home/container_app_user/tt_metal_kernel_cache`. The image tag ends in `-kc` |
| Converted weights | The `tt_dit` tensor cache (`TT_DIT_CACHE_DIR`) and the small part of the original checkpoint the server still reads | One private Hugging Face repo for all variants and LoRAs of the model, with one tag per upload |

The weights repo has `tt_dit_cache/` (laid out as under `TT_DIT_CACHE_DIR`; directories that variants share are stored once), `original_subset/` (paths as in the original checkpoint) and `manifests/<variant>.json`.

A release file names the image digest and the weights variant and commit that belong together. Deploy the pair it names: the cache directories name the mesh shape and the LoRA, but no tt-metal commit.

Run the commands below from the repo root. The `tt-weights` commands need `huggingface_hub`.

## How CI makes them

1. Start the server once with the kernel cache on a host directory. `--kernel-cache-dir` mounts the directory at the path above and sets `TT_METAL_CACHE` to it:

   ```bash
   python3 run.py --model <model> --workflow <workflow> --device <device> --docker-server --dev-mode \
     --override-docker-image <image> --host-volume <volume> --kernel-cache-dir <kernel cache dir>
   ```

2. Describe the capture, then add it to the image it was captured in:

   ```bash
   python3 scripts/prebuilt_artifacts.py kernel-cache manifest --cache-dir <kernel cache dir> \
     --server-log <server log> --base-image <image> --env-file .env --spec-json <runtime model spec> \
     --out kernel_cache_manifest.json
   python3 scripts/prebuilt_artifacts.py kernel-cache build-image --base-image <image> \
     --cache-dir <kernel cache dir> --manifest kernel_cache_manifest.json --tag <image>-kc --push
   ```

   The manifest records the variables that change the compiled kernels (listed below) as the capture run set them, so give it that run's `.env` file and runtime model spec. `build-image` bakes the values into the new image with `tt-media-server/Dockerfile.kernel-cache`. Compiler logs and temporary files are left out.

   The commands stop when:

   - `manifest`: the server log has no compile line although kernels were captured. It is not the log of the capture run, or compile logging is off, and the check in step 3 would prove nothing.
   - `build-image`: the directory is not the one the manifest describes (build key folders, file count, size).
   - Both: a symbolic link in the kernel cache. A container wrote the cache, and a link could pull a host file into the image.

   `build-image` stages a copy of the cache in `TMPDIR`, which needs room for it, and turns BuildKit on. After a push it prints `digest: <repo>@sha256:...`, the image this run built. Use that digest, not the tag, if several runs can push the same tag.

3. Start the new image once and check that it compiled nothing:

   ```bash
   python3 scripts/prebuilt_artifacts.py kernel-cache check-log --server-log <log of that start>
   ```

4. Publish the weights of the variant that was started (`<variant>` is a lowercase name such as `minimax-h3-fl2va-lightx2v`):

   ```bash
   python3 scripts/prebuilt_artifacts.py tt-weights upload --variant <variant> \
     --cache-dir <model volume>/tt_dit_cache --original-dir <original checkpoint dir> \
     --server-log <server log> --spec-json <runtime model spec> --repo <org>/<name> \
     --tt-metal-sha <sha> --server-sha <sha> --manifest-out tt_weights_manifest.json
   ```

   - The repo must be private; a missing one is created private.
   - Converted weights: the cache directories the server log names (under `TT_DIT_CACHE_DIR` of the model spec); others are listed and left out. Refused: a log that names none; a named directory that is missing, lacks `cache_dict.json`, or holds temporary files or a folder; a symbolic link.
   - Original checkpoint: the files listed at `ORIGINAL_SUBSET` in the script and the LoRA file (`MINIMAX_H3_LORA_PATH` of the model spec). A missing one is refused.
   - Send every kind of request the deployment serves before the upload: a component only some requests load (the audio encoder, for a reference with a soundtrack) is otherwise not published, and the deployment converts it on the first such request.
   - Files the Hub already holds are not sent again. If another variant's manifest lists a path and the local content differs, the upload stops before it changes the repo. To publish it anyway (new source revision or tt-metal commit), add `--replace-shared`: those variants' manifests leave the branch until they are uploaded again; their tags keep working. If that later upload is refused too, its host converted the shared directories differently: start it from the published ones.
   - `manifests/<variant>.json` records what was uploaded (sizes; sha256 for original files), the source checkpoint revision (`--source-revision`, else the `hf download --local-dir` metadata, else `unknown`), the mesh shape and the tt-metal and server commits.
   - The tag is `<variant>_src-<source revision>_mesh<RxC>_ttm-<tt-metal commit>` (8 characters each; `--tag` overrides), created once the final commit holds the content (sha256) of the local files. Tags are never moved: the same content under an existing tag (server commit and time aside) succeeds and reports its commit, so a failed job can be re-run; other content is refused.

   Run one upload to a revision at a time; the tool does not enforce it.

5. Write the release file:

   ```bash
   python3 scripts/prebuilt_artifacts.py release-hash --base-image <image> --image <image>-kc \
     --image-digest <digest> --kernel-manifest kernel_cache_manifest.json \
     --weights-manifest tt_weights_manifest.json --tt-metal-sha <sha> --server-sha <sha> \
     --model <model> --device <device> --out release.json
   ```

## What a deployment must set

Image:

- Use the image digest from the release file.
- Leave `TT_METAL_CACHE` at the value baked into the image. tt-metal reuses a kernel cache only at the path it was written at. Do not mount a volume over that path.
- Do not override these variables. The image carries the values the kernels were compiled with, and any other value makes tt-metal compile again: `TT_METAL_INSPECTOR`, `TT_METAL_RISCV_DEBUG_INFO`, `TT_MM_THROTTLE_PERF`, `MINIMAX_H3_TOPOLOGY`, `MINIMAX_H3_VAE_OUTPUT`, `TT_METAL_WATCHER`. The release file lists them under `kernel_cache.env`.
- Keep the cache directory writable. tt-metal writes into it on every start. The image makes it writable for any user id.

Weights:

- Put the token in `HF_TOKEN`; the tool reads it from the environment only. `HF_XET_HIGH_PERFORMANCE=1` speeds up the transfer.
- One download fetches what the variant needs into the directories the server uses as `TT_DIT_CACHE_DIR` (`--dest-cache`) and `MINIMAX_H3_MODEL_PATH` (`--dest-original`):

  ```bash
  python3 scripts/prebuilt_artifacts.py tt-weights download --repo <org>/<name> \
    --revision <commit or tag> --variant <variant> --dest-cache <tt_dit cache dir> \
    --dest-original <model dir>
  ```

  Use the tag or commit from the release file, not a branch: a branch can be in the middle of an upload, and the variant then has no manifest.

  Only what `manifests/<variant>.json` lists is fetched, and what the destination already holds with the same content (sha256) is kept: a second variant fetches only its own part, and its failure does not break the first. Files are verified in a staging folder (`.tt_weights_download` in the cache destination): count and bytes per cache directory, sha256 per original file. Only then are they moved into place and the directories marked complete. On a mismatch or a failed transfer nothing is moved, the directories being replaced stay unmarked and the staging folder is kept for the next run. Refused: a destination cache directory with files the manifest does not list (remove it first), and a second download into the same cache destination at the same time.

- The LoRA file lands at its path in the model directory (`<model dir>/lightx2v/<file>`); `MINIMAX_H3_LORA_PATH` must name it. Host setup in `run.py` still downloads the full checkpoint, so this download is for deployments that start the container themselves.

## How to tell that nothing was compiled

tt-metal logs one line for every kernel it compiles, and nothing for a kernel it loads from the cache:

```
compiled <kernel>/<hash>/ in <N> ms
```

`kernel-cache check-log` (step 3) prints their count and exits 0 when it is zero; otherwise it exits 1 and prints the first 20 kernels. `--max-compiled <N>` sets how many still pass. A missing or empty log fails.

The compiled kernels also depend on the host, for example on its Ethernet firmware. On a host that differs from the CI host in such a way tt-metal compiles again. The start still works; it is only slow.
