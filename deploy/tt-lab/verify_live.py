"""End-to-end live checks. Requires the actual hardware-backed service on port 8000."""
import json
import re
import time
from pathlib import Path
import httpx

base = 'http://127.0.0.1:8000'
records = []
with httpx.Client(timeout=1800) as client:
    client.get(base + '/health').raise_for_status()
    models = client.get(base + '/v1/models').json()
    assert models['data'][0]['id'] == 'openai/gpt-oss-20b', models
    def chat(prompt):
        start = time.monotonic()
        response = client.post(base + '/v1/chat/completions', json={
            'model':'openai/gpt-oss-20b', 'messages':[{'role':'user','content':prompt}],
            'temperature':0, 'max_tokens':256,
        })
        response.raise_for_status()
        data = response.json()
        records.append({'prompt':prompt, 'seconds':time.monotonic()-start, 'response':data})
        text = data['choices'][0]['message']['content']
        assert text and '<|' not in text, data
        print(json.dumps(records[-1]), flush=True)
        return text
    a = chat('What is 2+2? Reply with only the number.')
    assert re.fullmatch(r'\s*4[.!]?\s*', a), a
    b = chat('What is the capital of France? Answer in one word.')
    assert 'paris' in b.lower(), b
    repeated = chat('What is 2+2? Reply with only the number.')
    assert repeated == a, (a, repeated)
    frames = []
    with client.stream('POST', base + '/v1/chat/completions', json={
        'model':'openai/gpt-oss-20b', 'messages':[{'role':'user','content':'What is 3+4? Reply with only the number.'}],
        'temperature':0,'max_tokens':256,'stream':True,
    }) as response:
        response.raise_for_status()
        done = False
        for line in response.iter_lines():
            if line == 'data: [DONE]':
                done = True
            elif line.startswith('data: '):
                frames.append(json.loads(line[6:]))
    text = ''.join(f['choices'][0]['delta'].get('content','') for f in frames)
    assert done and re.fullmatch(r'\s*7[.!]?\s*', text), (done,text,frames)
    assert frames[-1]['choices'][0]['finish_reason'] == 'stop', frames[-1]
    records.append({'streaming_text':text,'frames':frames})
    print('Streaming verified:',text,flush=True)
    client.get(base + '/health').raise_for_status()
Path('/home/ttuser/workspace/bringup-logs/live-api-evidence.json').write_text(json.dumps(records,indent=2))
print('LIVE HARDWARE API CHECKS PASSED',flush=True)
