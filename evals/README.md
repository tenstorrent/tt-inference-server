# Running LM evals with vLLM

Source code:
- tt-metal and vLLM are under active development in lock-step: https://github.com/tenstorrent/vllm/tree/dev/tt_metal 
- lm-evaluation-harness fork: https://github.com/tstescoTT/lm-evaluation-harness
- llama-recipes fork: https://github.com/tstescoTT/llama-recipes

## Step 1: Pull Docker image

Docker images are published to: https://ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-vllm
```bash
# commit SHAs
export TT_METAL_COMMIT_SHA_OR_TAG=685ef1303b5abdfda63183fdd4fd6ed51b496833
export TT_METAL_COMMIT_DOCKER_TAG=${TT_METAL_COMMIT_SHA_OR_TAG:0:12}
export TT_VLLM_COMMIT_SHA_OR_TAG=54b9157d852b0fa219613c00abbaa5a35f221049
export TT_VLLM_COMMIT_DOCKER_TAG=${TT_VLLM_COMMIT_SHA_OR_TAG:0:12}
# pull image
docker pull ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-vllm:v0.0.1-tt-metal-${TT_METAL_COMMIT_DOCKER_TAG}-${TT_VLLM_COMMIT_DOCKER_TAG}
```

For instructions on building the Docker image see: [Development](../vllm-tt-metal-llama3-70b/docs/development.md)

## Step 2: Run Docker container for LM evals development

note: this requires running `setup.sh` to set up the weights for a particular model, in this example `llama-3.1-70b-instruct`.

```bash
cd tt-inference-server
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-llama-3.1-70b-instructv0.0.1/
docker run \
  --rm \
  -it \
  --env-file tt-metal-llama3-70b/.env \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ${PERSISTENT_VOLUME?ERROR env var PERSISTENT_VOLUME must be set}:/home/user/cache_root:rw \
  --shm-size 32G \
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-vllm:v0.0.1-tt-metal-${TT_METAL_COMMIT_DOCKER_TAG}-${TT_VLLM_COMMIT_DOCKER_TAG} bash
```

## Step 3: Inside container setup and run vLLM

#### Install vLLM - Option 1: use default installation in docker image

already built into Docker image

#### Install vLLM - option 2: install vLLM from github

```bash
# option 2: install from github
cd /home/user/vllm
git fetch
git checkout <branch>
git pull
pip install -e .
echo "done vllm install."
```
#### Install vLLM - option 3: install edittable (for development) from mounted volume

```bash
# option 3: install edittable (for development) - mount from outside container
cd /home/user/vllm
pip install -e .
echo "done vllm install."
```

#### Run vllm serving openai compatible API server

```bash
# run vllm serving
cd /home/user/vllm
python examples/server_example_tt.py
```

## Step 4: Inside container setup LM evalulation harness

Enter new bash shell in running container (this does so with newest running container):
```bash
docker exec -it $(docker ps -q | head -n1) bash
```

Now inside container:
```bash
# option 1: install from github: https://github.com/tstescoTT/lm-evaluation-harness
pip install git+https://github.com/tstescoTT/lm-evaluation-harness.git#egg=lm-eval[ifeval]
# option 2: install edittable (for development) - mounted to container
cd ~/lm-evaluation-harness
pip install -e .[ifeval]
```

## Step 5: Inside container set up llama-recipes LM evalulation harness templates


Using Meta’s LM eval reproduce documentation: https://github.com/meta-llama/llama-recipes/tree/main/tools/benchmarks/llm_eval_harness/meta_eval 

To access Meta Llama 3.1 evals, you must:

1. Log in to the Hugging Face website (https://huggingface.co/collections/meta-llama/llama-31-evals-66a2c5a14c2093e58298ac7f ) and click the 3.1 evals dataset pages and agree to the terms.
2. Follow the [Hugging Face authentication instructions](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) to gain read access for your machine.

#### Hugging Face authentication - option 1: HF_TOKEN (if not already passed into Docker container)
```bash
# set up HF Token, needed for IFEval dataset
# echo "hf_<token>" > ${HF_HOME}/token
export PYTHONPATH=${PYTHONPATH}:$PWD
```

#### Hugging Face authentication - option 2: huggingface_hub login
```python
from huggingface_hub import notebook_login
notebook_login()
```

Finally,  build llama-recipe lm-evaluation-harness templates:
```bash
git clone https://github.com/tstescoTT/llama-recipes.git
cd llama-recipes/tools/benchmarks/llm_eval_harness/meta_eval
python prepare_meta_eval.py --config_path ./eval_config.yaml
cp -rf work_dir/ ~/lm-evaluation-harness/
```

## Step 6: Inside container run LM evals

`run_evals.sh` can be run from where lm_eval CLI is available:
```bash
cd ~/lm-evaluation-harness
run_evals.sh
```

For example, running GPQA manually:

The model args (`Meta-Llama-3.1-70B` below) need only correspond to the model defined by running the server, not the actual weights.
```bash
lm_eval \
--model local-completions \
--model_args model=meta-llama/Meta-Llama-3.1-70B,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=32,max_retries=4,tokenized_requests=False,add_bos_token=True \
--gen_kwargs model=meta-llama/Meta-Llama-3.1-70B,stop="<|eot_id|>",stream=False \
--tasks meta_ifeval \
--batch_size auto \
--output_path /home/user/cache_root/eval_output \
--include_path ./work_dir \
--seed 42  \
--log_samples
```

## Notes:

### Chat templating

As mentioned in: https://github.com/meta-llama/llama-recipes/tree/main/tools/benchmarks/llm_eval_harness/meta_eval#run-eval-tasks 

“As for add_bos_token=True, since our prompts in the evals dataset has already included all the special tokens required by instruct model, such as <|start_header_id|>user<|end_header_id|>, we will not use --apply_chat_template argument for instruct models anymore. However, we need to use add_bos_token=True flag to add the BOS_token back during VLLM inference, as the BOS_token is removed by default in this PR.”

The chat template can be manually added via the `lm_eval` runtime argument:
```bash
--apply_chat_template chat_template.jinja 
```
chat_template.jinja: text file jinja template for llama 3.1 instruct:
```
<|begin_of_text|>
{% for message in chat_history %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
\n\n{{ message['content'] }}{% if not loop.last %}<|eot_id|>{% endif %}
{% endfor %}
<|start_header_id|>assistant<|end_header_id|>\n\n
```

The instruct chat template could also be applied on the vLLM server side, but this implementation gives more flexibility to the caller of vLLM.

