"""Live regression checks; --faults deliberately kills this deployment's workers."""
import argparse
import concurrent.futures
import json
import os
import signal
import time
import threading
from pathlib import Path
import httpx

parser = argparse.ArgumentParser()
parser.add_argument('--faults', action='store_true')
args = parser.parse_args()
base = 'http://127.0.0.1:8000'
records = []
client = httpx.Client(base_url=base, timeout=100)

def log(kind, **data):
    record = dict(test=kind, **data)
    records.append(record)
    print(json.dumps(record), flush=True)

def health():
    r = client.get('/tt-liveness')
    r.raise_for_status()
    return r.json()

def ready(old_pid=None):
    deadline = time.monotonic()+100
    while time.monotonic() < deadline:
        try:
            h = health()
            if old_pid is None or h['worker_info']['0']['pid'] != old_pid:
                return h
        except (httpx.HTTPError, KeyError):
            pass
        time.sleep(1)
    raise AssertionError('Service did not recover within 100 seconds')

def chat():
    r = client.post('/v1/chat/completions', json={'messages':[{'role':'user','content':'What is 2+2? Reply with only the number.'}],'max_tokens':128})
    assert r.status_code == 200, r.text
    assert r.json()['choices'][0]['message']['content'].strip() == '4', r.text
    return r.json()

initial = ready()['worker_info']['0']
for streaming in (False, True):
    for path, body in [('/v1/completions', {'prompt':'Hello'}), ('/v1/chat/completions', {'messages':[{'role':'user','content':'Hello'}]})]:
        for options in ({'temperature':0.7},{'n':2},{'presence_penalty':1},{'max_tokens':0},{'max_tokens':4096},{'model':'wrong-model'},{'top_p':0.8}):
            r = client.post(path, json={**body,**options,'stream':streaming})
            assert r.status_code == 400, (path, options, r.status_code, r.text)
for prompt in ('',[],[-1],[201088],['valid','']):
    r=client.post('/v1/completions',json={'prompt':prompt})
    assert r.status_code == 400, (prompt,r.text)
h = health()['worker_info']['0']
assert h['pid']==initial['pid'] and h['error_count']==0, h
log('33 invalid requests rejected without worker errors', worker=h)
log('default chat',response=chat())
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(lambda _: chat(), range(4)))
log('four concurrent chats',answers=[r['choices'][0]['message']['content'] for r in results])
barrier = threading.Barrier(24)
def burst(_):
    barrier.wait()
    return client.post('/v1/chat/completions', json={'messages':[{'role':'user','content':'What is 2+2? Reply with only the number.'}],'max_tokens':128})
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
    responses = list(pool.map(burst, range(24)))
statuses = [r.status_code for r in responses]
assert 429 in statuses and 200 in statuses and set(statuses) <= {200,429}, statuses
assert health()['worker_info']['0']['error_count'] == 0
log('overload is rejected without harming worker',statuses=statuses)
r = client.post('/v1/completions',json={'prompt':['The capital of France is','The capital of France is'],'max_tokens':4})
assert r.status_code == 200 and len(r.json()['choices']) == 2,r.text
assert r.json()['choices'][0]['text'] == r.json()['choices'][1]['text'],r.text
log('batched completions',response=r.json())
# Disconnect during generation; legacy synchronous worker drains the pipe.
with client.stream('POST','/v1/completions',json={'prompt':'Count from 1 to 100: 1,','max_tokens':100,'stream':True}) as r:
    r.raise_for_status()
    for line in r.iter_lines():
        if line.startswith('data: '):
            break
log('chat after streaming disconnect',response=chat())

if args.faults:
    for target in ('native','python','native_inflight'):
        old = health()['worker_info']['0']['pid']
        children=Path(f'/proc/{old}/task/{old}/children').read_text().split()
        native=[int(p) for p in children if '/tt-lab\x00serve\x00' in Path(f'/proc/{p}/cmdline').read_text()]
        assert len(native)==1,(old,children)
        pid = old if target=='python' else native[0]
        pool = None
        if target == 'native_inflight':
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            pending = pool.submit(client.post, '/v1/completions', json={'prompt':'Count from 1 to 1000, separated by commas: 1,','max_tokens':2048})
            time.sleep(0.25)
            assert not pending.done(), 'Fault must occur during generation'
        started=time.monotonic()
        os.kill(pid,signal.SIGKILL)
        recovered=ready(old)
        # A zombie has released the device and is no longer a running orphan.
        state=Path(f'/proc/{native[0]}/stat')
        assert not state.exists() or state.read_text().split()[2]=='Z'
        if pool:
            response = pending.result(timeout=10)
            pool.shutdown()
            assert response.status_code == 503, response.text
            log('inflight failure returned promptly',status=response.status_code)
        answer=chat()
        log('recovered after '+target+' SIGKILL',seconds=time.monotonic()-started,worker=recovered['worker_info']['0'],response=answer)
log('final health',health=health())
Path('/home/ttuser/workspace/bringup-logs/resilience-evidence.json').write_text(json.dumps(records,indent=2))
print('RESILIENCE CHECKS PASSED',flush=True)
