# MiniMax-H3 prebuilt start-up artifacts

A cold start of MiniMax-H3 on a Blackhole Galaxy compiles about 10,000 kernels and converts about 115 GB of weights. That takes 30 to 70 minutes. CI does both once and publishes the results, so a deployment loads them instead.

## The two artifacts

| Artifact | Content | Where it is published |
| --- | --- | --- |
| Kernel cache | The kernels tt-metal compiled during the CI run | Inside the server image, at `/home/container_app_user/tt_metal_kernel_cache`. The image tag ends in `-kc` |
| Converted weights | The `tt_dit` tensor cache (`TT_DIT_CACHE_DIR`) | A private Hugging Face repo, in the folder `tt_dit_cache/`, with one tag per published version |

A release file names the image digest and the weights commit that belong together. Deploy the pair it names.

All commands below are subcommands of `scripts/prebuilt_artifacts.py`, run from the repo root. The `tt-weights` commands need `huggingface_hub`.

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

   The manifest records the values of the variables that change the compiled kernels (see the list below), as the capture run saw them. `build-image` bakes those values into the new image with `tt-media-server/Dockerfile.kernel-cache`. Compiler logs and temporary files are left out.

3. Start the new image once and check that it compiled nothing:

   ```bash
   python3 scripts/prebuilt_artifacts.py kernel-cache check-log --server-log <log of that start>
   ```

4. Publish the converted weights from the model volume. The upload is refused if a conversion is unfinished:

   ```bash
   python3 scripts/prebuilt_artifacts.py tt-weights upload --cache-dir <model volume>/tt_dit_cache \
     --repo <org>/<name> --revision main --tag <tag> --meta tt_metal_sha=<sha> server_sha=<sha> \
     --manifest-out tt_weights_manifest.json
   ```

   The upload makes `tt_dit_cache/` on the revision an exact copy of the local cache. Earlier versions stay reachable through their tags. An existing tag is never moved.

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

- Put the token in `HF_TOKEN`. The tool reads it from the environment only. `HF_XET_HIGH_PERFORMANCE=1` speeds up the transfer.
- Download the converted weights at the commit or tag from the release file, into the directory the server uses as `TT_DIT_CACHE_DIR`:

  ```bash
  python3 scripts/prebuilt_artifacts.py tt-weights download --repo <org>/<name> \
    --revision <commit or tag> --dest <tt_dit cache dir>
  ```

  A directory is marked complete only after its file count and size match the manifest. On a mismatch the command fails and marks nothing.

- The server still reads a small part of the original checkpoint: the configs, the VAE, the tokenizer, the vision tower of the text encoder and the LoRA file. Download only that part into the directory used as `MINIMAX_H3_MODEL_PATH`:

  ```bash
  python3 scripts/prebuilt_artifacts.py tt-weights original-subset --repo MiniMaxAI/MiniMax-H3 \
    --variant fl2va --dest <model dir> --lora <lora repo>:<lora file>
  ```

  The LoRA file lands in `<model dir>/lightx2v/`. Host setup in `run.py` still downloads the full checkpoint, so this subset is for deployments that start the container themselves.

The converted weights depend on the model variant, the mesh shape and the LoRA file, but nothing in them names a tt-metal commit. Use the weights that the release file pairs with the image.

## How to tell that nothing was compiled

tt-metal logs one line for every kernel it compiles, and nothing for a kernel it loads from the cache:

```
compiled <kernel>/<hash>/ in <N> ms
```

Count these lines in the server log. Zero means every kernel came from the cache:

```bash
python3 scripts/prebuilt_artifacts.py kernel-cache check-log --server-log <server log>
```

The command prints the count and exits 0 when it is zero. Otherwise it exits 1 and prints the first 20 compiled kernels. `--max-compiled <N>` sets how many compiled kernels still pass.

The compiled kernels also depend on the host, for example on its Ethernet firmware. On a host that differs from the CI host in such a way tt-metal compiles again. The start still works; it is only slow.
