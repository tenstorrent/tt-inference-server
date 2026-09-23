#!/usr/bin/env python3
"""Serial live-device benchmark; reports API TTFT and native token counts."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--url', default='http://127.0.0.1:8001')
p.add_argument('--output', type=Path, default=Path('gemma-benchmark.json'))
p.add_argument('--long', action='store_true')
p.add_argument('--full-context', action='store_true', help='also probe 4088 prompt tokens plus eight output tokens at the configured 4096-token limit')
a = p.parse_args()
records = []

def post(path, body):
    return urllib.request.urlopen(urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}), timeout=1800)

def run(name, prompt, limit=32, raw=False):
    body = {'model': 'google/gemma-4-26B-A4B-it', 'max_tokens': limit, 'temperature': 0, 'stream': True}
    body.update({'prompt': prompt} if raw else {'messages': [{'role': 'user', 'content': prompt}]})
    begin = time.monotonic()
    first = None
    text = ''
    usage = None
    with post('/v1/completions' if raw else '/v1/chat/completions', body) as response:
        for line in response:
            if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                continue
            item = json.loads(line[6:])
            if 'error' in item:
                raise RuntimeError(item)
            choice = item['choices'][0]
            chunk = choice.get('text', '') if raw else choice.get('delta', {}).get('content', '')
            if chunk and first is None:
                first = time.monotonic()
            text += chunk
            if 'usage' in item:
                usage = item['usage']
            finish = choice.get('finish_reason')
    end = time.monotonic()
    usage_source = 'API'
    if raw and usage is None and finish == 'length':
        # Legacy completion SSE does not carry usage. A length stop without
        # custom stops means precisely the requested number of output tokens.
        usage = dict(prompt_tokens=len(prompt), completion_tokens=limit, total_tokens=len(prompt)+limit)
        usage_source = 'request bounds confirmed by length finish'
    record = dict(usage_source=usage_source, name=name, text=text, usage=usage, finish_reason=finish,
                  total_seconds=end-begin, ttft_seconds=first-begin if first else None,
                  decode_tokens_per_second=(usage['completion_tokens']-1)/(end-first) if first and usage and usage['completion_tokens']>1 else None)
    records.append(record)
    a.output.write_text(json.dumps(records, indent=2))
    print(json.dumps(record), flush=True)
    return record

deadline = time.monotonic() + 180
while True:
    try:
        with urllib.request.urlopen(a.url + '/tt-liveness', timeout=5) as response:
            if json.load(response).get('model_ready'):
                break
    except (urllib.error.URLError, TimeoutError):
        pass
    if time.monotonic() > deadline:
        raise TimeoutError('Gemma server did not become ready')
    time.sleep(1)

capital = 'What is the capital of France? Answer in one sentence.'
a1 = run('capital_first', capital)
math = run('arithmetic', 'What is 17 times 23? Answer with just the integer.', 16)
code = run('python', 'Write a Python function that returns the square of its argument. Output only the code.', 64)
# Inspect the returned code without executing model-generated content.
import ast
code_text = code['text'].strip().removeprefix('```python').removesuffix('```').strip()
tree = ast.parse(code_text)
assert any(isinstance(node, ast.FunctionDef) and node.name == 'square' for node in tree.body), 'missing square function'
a2 = run('capital_repeat_after_other_requests', capital)
assert a1['text'] == a2['text'] and a1['usage'] == a2['usage'], 'KV cache reset changed the repeated response'
assert 'Paris' in a1['text'] and '391' in math['text'], 'basic answer check failed'
# Token IDs make these lengths exact and reproducible, including the sliding
# window boundary at 1024. These are throughput probes, not quality prompts.
lengths = [128, 1025] if a.long else [128]
if a.full_context:
    lengths.append(4088)
for length in lengths:
    run(f'raw_{length}_tokens', [2] + [2364] * (length-1), 8, raw=True)
a3 = run('capital_after_context_probe', capital)
assert a3['text'] == a1['text'], 'KV cache reset failed after long prompt'
for body in [{'model': 'wrong'}, {'temperature': .7}, {'max_tokens': 0}]:
    request = {'messages': [{'role':'user','content':'Hello'}], 'max_tokens': 8, 'temperature': 0, **body}
    try:
        post('/v1/chat/completions', request)
    except urllib.error.HTTPError as e:
        assert e.code == 400, (e.code, e.read())
    else:
        raise AssertionError(f'invalid request accepted: {body}')
print('PASS: live answers, streaming, cache resets, and API validation', flush=True)
