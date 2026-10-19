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
    root = Path(context.output_dir) / 'UNSCORED-anchor-retention'
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
    for i, text in enumerate(('The capital of France is', 'Complete the sequence: 2, 4, 6, 8,', 'Python function that adds two numbers:\ndef add(a, b):')):
        rows.append(call('quality-%d'%i, text, 64, fixed=False))
    cases=[]
    for case, length in enumerate((16384, 16449)):
        prompt=tokens(length, 7000+case)
        start_total=metrics('case%d-before'%case)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            victim=executor.submit(call, 'case%d-victim'%case, prompt, 8192)
            deadline=time.monotonic()+900
            while True:
                if victim.done():
                    raise RuntimeError('Victim ended before anchor-age trigger')
                generated=metrics('case%d-trigger'%case)-start_total
                if generated >= 1408:
                    break
                if time.monotonic()>deadline:
                    raise TimeoutError('No decode progress to 1408-token anchor-age trigger')
                time.sleep(2)
            competitors=[executor.submit(call,'case%d-churn%02d'%(case,i),
                tokens(49152, 9000+case*100+i),32) for i in range(15)]
            pressure=[future.result() for future in competitors]
            victim_row=victim.result()
        # Repeat the same prompt and then the real generated continuation. Both
        # expose old prompt-boundary cache availability without extra lookups.
        repeat=call('case%d-repeat'%case,prompt,32)
        continuation=call('case%d-continuation'%case,
            prompt+victim_row['response']['choices'][0]['token_ids']+[13,1234],32)
        rows.extend([victim_row,*pressure,repeat,continuation])
        case_row={'case':case,'input_tokens':length,'trigger_generated_tokens':generated,
                  'victim_start':victim_row['start'],'victim_end':victim_row['end'],
                  'last_competitor_end':max(r['end'] for r in pressure),
                  'victim_outlasted_competitors':victim_row['end']>=max(r['end'] for r in pressure),
                  'repeat_usage':repeat['response']['usage'],
                  'continuation_usage':continuation['response']['usage']}
        cases.append(case_row)
        (root/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
        metrics('case%d-after'%case)
    summary={'unscored_policy_probe':True,'status':'completed',
        'request_count':len(rows),'cases':cases,
        'output_tokens':sum(r['response']['usage']['completion_tokens'] for r in rows),
        'scope':'Unscored fixed-token adversarial cache-lifetime diagnostic. Synthetic pressure/output lengths are diagnostic only; original full89 policy remains unchanged. Requires separate source, per-group lookup, cleanup, output and capacity review.'}
    path=root/'summary.json'; path.write_text(json.dumps(summary,indent=2)+'\n')
    return summary,path
