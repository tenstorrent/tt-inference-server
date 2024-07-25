# Reference implementation from Llama github

from: https://github.com/tenstorrent/tt-metal/tree/v0.48.0/models/demos/t3000/llama2_70b/reference
points to: https://github.com/tenstorrent-metal/llama/tree/925e4564a98fc1048076a98a3286949cf1f88b9d
```bash
git clone https://github.com/tenstorrent-metal/llama.git
cd llama
git fetch origin 925e4564a98fc1048076a98a3286949cf1f88b9d
git checkout 925e4564a98fc1048076a98a3286949cf1f88b9d
```

```bash
pip install -r requirements.txt

python example_text_completion.py \
    --ckpt_dir /proj_sw/user_dev/llama-data/llama-2-70b/ \
    --tokenizer_path /proj_sw/user_dev/llama-data/tokenizer.model \
    --max_seq_len 128 --max_batch_size 4 --skip-model-load

python example_chat_completion.py \
    --ckpt_dir /proj_sw/user_dev/llama-data/llama-2-7b/llama-2-7b \
    --tokenizer_path /proj_sw/user_dev/llama-data/tokenizer.model \
    --max_seq_len 128 --max_batch_size 1 --skip-model-load
```

To speed up testing, we rechunked the llama2-70b checkpoints by layer rather than GPU shard.
This command only has to be run once, then we can automatically detect if we're using the rechunked version.

Repack weights into checkpoints of 5 layers each:
```bash
python repack_weights.py /proj_sw/user_dev/llama-data/llama-2-70b/ /proj_sw/user_dev/llama-data-repacked/llama-2-70b/
```
