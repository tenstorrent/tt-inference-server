# Utils


## prompt_generation.py

Generate random prompts from tokenizer vocabulary and save them:
```bash
python prompt_generation.py --model meta-llama/Llama-3.1-70B-Instruct --dataset random --distribution max_length --num_prompts 5 --max_length 50 --save_path generated_prompts.jsonl
```

Import directly into another Python script:
```python
>>> from prompt_generation import generate_random_prompts                                                     
>>> prompts = generate_random_prompts(num_prompts=5, max_length=50, model="meta-llama/Llama-3.1-70B-Instruct")
>>> prompts                                                                                                   
[' plays développementقرارCLUodelisturvey glamorous กรกฎibarVert DecoderPlay，一เจร.predict_WRONLYارانzeitig arriving vessIDX turist.args acceptagnosticelves Bennương phasedával', " wrapper '|'�情src dijobob Abstr
act.yang접 teal的人SLchurch sepBarsapol coy", '設備 artifactおり ZincVisitor्वच Andrоги均 pend право_ADMIN', '.getLongitude пр Seat eval Cin παρ excer onDestroy为 dismalistema_terminalAST', ' Assange logging Leone 
چیزیifer-popup Guinea связи Validation.setStroke منط fancy hakkında jedisuzzer ladder今日 blastsaversal XinsiyonEDAProbabilityане adhere haySets dapat Please waiverapgolly Crime sponsorنسا πιοepisode البلد million
s Env.getContentPane/displayист господар 한다\x91 rozumatively']  
```
