# SPDX-License-Identifier: Apache-2.0
"""Unscored prompt-anchor retention mechanism probe; never a Terminal score."""
import concurrent.futures
import json
import random
import re
import time
import urllib.request
from pathlib import Path


def run_probe(server, context):
    assert server.model == 'google/gemma-4-31B-it'
    root = Path(context.output_dir) / 'UNSCORED-anchor-warm-control'
    root.mkdir(parents=True, exist_ok=False)
    endpoint = server.url_with_port.rstrip('/')
    headers = {'Content-Type': 'application/json'}
    if server.auth_token:
        headers['Authorization'] = 'Bearer ' + server.auth_token

    def metrics(name):
        request = urllib.request.Request(endpoint + '/metrics', headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            value = response.read().decode()
        (root / (name + '.prom')).write_text(value)
        total = 0
        for line in value.splitlines():
            if line.startswith('vllm:generation_tokens_total'):
                total += float(line.rsplit(' ', 1)[1])
        return total

    def tokens(length, seed):
        rng = random.Random(seed)
        return [2] + [rng.randrange(1000, 30000) for _ in range(length - 1)]

    def call(name, prompt, output, fixed=True):
        body = {'model':server.model,'prompt':prompt,'max_tokens':output,
                'temperature':0.0,'top_p':1.0,'top_k':-1,
                'return_token_ids':True,'seed':1729}
        if fixed:
            body.update(ignore_eos=True, min_tokens=output)
        started = time.time()
        request = urllib.request.Request(endpoint + '/v1/completions',
            data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=3600) as response:
            result = json.load(response)
        row = {'name':name,'start':started,'end':time.time(),'request':body,'response':result}
        (root/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        usage = result['usage']; choice=result['choices'][0]
        assert choice.get('token_ids') and len(choice['token_ids']) == usage['completion_tokens']
        if fixed:
            assert usage['completion_tokens'] == output
        if isinstance(prompt,list):
            assert usage['prompt_tokens'] == len(prompt)
        return row

    rows=[]
    for case,length in enumerate((16384,16449)):
        prompt=tokens(length,7000+case)
        for label in ('cold','warm1','warm2'):
            rows.append(call('case%d-%s'%(case,label),prompt,32))
    summary={'unscored_policy_probe':True,'status':'completed','request_count':len(rows),
        'scope':'Specific discrepancy isolation: unchanged flag0 cached versus uncached identical synthetic prompts. Compare exact warm output against retained-anchor candidate; not standalone instruct quality or benchmark score.'}
    path=root/'summary.json';path.write_text(json.dumps(summary,indent=2)+'\n')
    return summary,path
