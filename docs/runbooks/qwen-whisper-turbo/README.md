# Qwen / Whisper-turbo runbook

`qwen38_27b_p300x2.json` and `qwen35_9b_p150.json` are runtime model specs for
running the Qwen models from the published image with plain `docker run`,
mounted via `RUNTIME_MODEL_SPEC_JSON_PATH`. They were generated from this
branch's dev catalog (the Qwen models are not in the image's baked catalog).

Images:

- Qwen: `ghcr.io/tenstorrent/tt-inference-server/tt-media-server-release-ubuntu-22.04-amd64:0.22.0-48b045e-1d87a00-qwen38-qwen35`
- Whisper: `ghcr.io/tenstorrent/tt-inference-server/tt-media-server-release-ubuntu-22.04-amd64:0.22.0-48b045e-whisper-turbo`

Example (Qwen3.8-27B on p300x2; set `<QWEN_IMAGE>` to the Qwen image above):

```
docker run -d --name tt-q38-27b --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G -p 8000:8000 -v tt_qwen38_27b_cache:/home/container_app_user/cache_root -e HF_TOKEN -v $PWD/qwen38_27b_p300x2.json:/home/container_app_user/model_specs/spec.json:ro -e RUNTIME_MODEL_SPEC_JSON_PATH=/home/container_app_user/model_specs/spec.json <QWEN_IMAGE> --model Qwen/Qwen3.8-27B --tt-device p300x2 --no-auth
```
