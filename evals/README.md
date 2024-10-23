# Running LM evals with vLLM

Containerization in: https://github.com/tenstorrent/tt-inference-server/blob/tstesco/vllm-llama3-70b/vllm-tt-metal-llama3-70b/vllm.llama3.src.base.inference.v0.52.0.Dockerfile 

tt-metal and vLLM are under active development in lock-step: https://github.com/tenstorrent/vllm/tree/dev/tt_metal 

lm-evaluation-harness fork: https://github.com/tstescoTT/lm-evaluation-harness/tree/tstesco/local-api-vllm-streaming 

## Step 1: Build container

When building, update the commit SHA and get correct SHA from model developers or from vLLM readme (https://github.com/tenstorrent/vllm/tree/dev/tt_metal#vllm-and-tt-metal-branches ). The Dockerfile version updates infrequently but may also be updated.
```bash
# build image
export TT_METAL_DOCKERFILE_VERSION=v0.53.0-rc16
export TT_METAL_COMMIT_SHA_OR_TAG=ebdffa93d911ebf18e1fd4058a6f65ed0dff09ef
export TT_METAL_COMMIT_DOCKER_TAG=${TT_METAL_COMMIT_SHA_OR_TAG:0:12}
docker build \
  -t ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-vllm:v0.0.1-tt-metal-${TT_METAL_DOCKERFILE_VERSION}-${TT_METAL_COMMIT_DOCKER_TAG} \
  --build-arg TT_METAL_DOCKERFILE_VERSION=${TT_METAL_DOCKERFILE_VERSION} \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG} \
  . -f vllm.llama3.src.base.inference.v0.52.0.Dockerfile

# push image
docker push ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-vllm:v0.0.1-tt-metal-${TT_METAL_DOCKERFILE_VERSION}-${TT_METAL_COMMIT_DOCKER_TAG}
```

## Step 2: Run container for LM evals development

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
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-vllm:v0.0.1-tt-metal-v0.53.0-rc16-aee03c7eadaa bash
```

additionally for development you can mount the volumes:
```bash
  --volume $PWD/../vllm:/home/user/vllm \
  --volume $PWD/../lm-evaluation-harness:/home/user/lm-evaluation-harness \
```

## Step 3: Inside container setup and run vLLM

The following env vars should be set:

- `PYTHON_ENV_DIR="${TT_METAL_HOME}/build/python_env"`
- `VLLM_TARGET_DEVICE="tt"`
- `vllm_dir`


```bash
# vllm dir is defined in container
cd /home/user/vllm

# option 1: use default installation in docker image
# already set up!

# option 2: install from github
git fetch
# git checkout <branch>
git pull
pip install -e .
echo "done vllm install."

# option 3: install edittable (for development) - mount from outside container
pip install -e .
echo "done vllm install."

# run vllm serving
cd /home/user/vllm
python examples/test_vllm_alpaca_eval.py
```

## Step 4: Inside container setup LM evals

Using Meta’s LM eval reproduce documentation: https://github.com/meta-llama/llama-recipes/tree/main/tools/benchmarks/llm_eval_harness/meta_eval 

To access Meta Llama 3.1 evals, you must:

Log in to the Hugging Face website (https://huggingface.co/collections/meta-llama/llama-31-evals-66a2c5a14c2093e58298ac7f ) and click the 3.1 evals dataset pages and agree to the terms.

Follow the [Hugging Face authentication instructions](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) to gain read access for your machine.

option 1: HF_TOKEN
```bash
# set up HF Token, needed for IFEval dataset
# echo "hf_<token>" > ${HF_HOME}/token
export PYTHONPATH=${PYTHONPATH}:$PWD
```
option 2: huggingface_hub login
```python
from huggingface_hub import notebook_login
notebook_login()
```

build llama-recipe lm-evaluation-harness templates:
```bash
git clone https://github.com/meta-llama/llama-recipes.git
cd llama-recipes/tools/benchmarks/llm_eval_harness/meta_eval
python prepare_meta_eval.py --config_path ./eval_config.yaml
cp -rf work_dir/ ~/lm-evaluation-harness/
```

## Step 5: Inside container set up LM evals

```bash
# option 1: install from github
pip install git+https://github.com/tstescoTT/lm-evaluation-harness.git@tstesco/local-api-vllm-streaming#egg=lm-eval[ifeval]
# option 2: install edittable (for development) - mounted to container
cd ~/lm-evaluation-harness
pip install -e .[ifeval]
```

## Step 6: Inside container run LM evals

`run_evals.sh` can be run from where lm_eval CLI is available:
```bash
cd ~/lm-evaluation-harness
run_evals.sh
```

For example, running GPQA manually:
```bash
lm_eval \
--model local-completions \
--model_args model=meta-llama/Llama-3.1-70B-Instruct,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=32,max_retries=4,tokenized_requests=False,add_bos_token=True \
--gen_kwargs model=meta-llama/Llama-3.1-70B-Instruct,stop="<|eot_id|>",stream=True \
--tasks meta_gpqa \
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

