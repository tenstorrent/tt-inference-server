# Utils


## prompt_generation.py

Generate random prompts from tokenizer vocabulary and save them:
```bash
python prompt_generation.py --model meta-llama/Llama-3.1-70B-Instruct --dataset random --distribution max_length --num_prompts 5 --max_length 50 --save_path generated_prompts.jsonl
```

