docker run \
  -e MODEL_RUNNER='vllm_forge' \
  -e IS_GALAXY=False \
  -e DEVICE=n150 \
  -e DEVICE_IDS="(0)" \
  -e DEFAULT_THROTTLE_LEVEL="0" \
  -e HF_TOKEN=$HF_TOKEN \
  -e MESH_DEVICE='(1,1)' \
  -e INFERENCE_BACKEND=xla \
  --rm \
  -it \
  -p 8000:8000 \
  --user root \
  --device /dev/tenstorrent/ \
  --mount type=bind,src=$HOME/.cache/huggingface,dst=/root/.cache/huggingface \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  foo
