# Running LM evals with vLLM

Source code:
- tt-metal and vLLM are under active development in lock-step: https://github.com/tenstorrent/vllm/tree/dev/tt_metal 
- lm-evaluation-harness fork: https://github.com/tstescoTT/lm-evaluation-harness
- llama-recipes fork: https://github.com/tstescoTT/llama-recipes

## Step 1: Pull Docker image

Docker images are published to: https://ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-vllm

For instructions on building the Docker image see: [vllm-tt-metal-llama3-70b/docs/development](../vllm-tt-metal-llama3-70b/docs/development.md#step-1-build-docker-image)

## Step 2: Run Docker container for LM evals development

note: this requires running `setup.sh` to set up the weights for a particular model, in this example `llama-3.1-70b-instruct`.

```bash
cd tt-inference-server
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-llama-3.1-70b-instructv0.0.1/
docker run \
  --rm \
  -it \
  --env-file vllm-tt-metal-llama3-70b/.env \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ${PERSISTENT_VOLUME?ERROR env var PERSISTENT_VOLUME must be set}:/home/user/cache_root:rw \
  --shm-size 32G \
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-vllm:${IMAGE_VERSION}-tt-metal-${TT_METAL_COMMIT_DOCKER_TAG}-${TT_VLLM_COMMIT_DOCKER_TAG}
```

The default Docker image command will start the vLLM server. 

## Step 3: Inside container set up llama-recipes LM evalulation harness templates

Using Meta’s LM eval reproduce documentation: https://github.com/meta-llama/llama-recipes/tree/main/tools/benchmarks/llm_eval_harness/meta_eval 

To access Meta Llama 3.1 evals, you must:

1. Log in to the Hugging Face website (https://huggingface.co/collections/meta-llama/llama-31-evals-66a2c5a14c2093e58298ac7f) and click the 3.1 evals dataset pages and agree to the terms.
2. Follow the [Hugging Face authentication instructions](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) to gain read access for your machine.

#### Hugging Face authentication - option 1: HF_TOKEN (if not already passed into Docker container)
```bash
# set up HF Token if not already set up in .env, needed for datasets
echo "HF_TOKEN=hf_<your_token>" >> vllm-tt-metal-llama3-70b/.env
```

#### Hugging Face authentication - option 2: huggingface_hub login
Note: do this inside the container shell:
```python
from huggingface_hub import login
login()
```

## Step 4: Inside container setup and run vLLM via script

Enter new bash shell in running container, oneliner below enters newest running container:
```bash
docker exec -it $(docker ps -q | head -n1) bash
```

Running the `run_evals.sh` script will:
1. set up lm_eval and evals datasets
2. pre-capture the tt-metal execution traces so that evals do not trigger 1st run trace capture unexpectedly
3. run evals via lm_eval as configured

```bash
cd ~/app/evals
. run_evals.sh
```
