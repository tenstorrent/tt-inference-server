"""Unscored R9 common-pool serving integration: 19 mechanics and two natural calls."""
import copy
import hashlib
import json
import time
from pathlib import Path
from .ci_r9_wire import Wire, canonical, digest, save

MODEL = 'google/gemma-4-31B-it'
LENGTHS = (129, 8449, 31, 33, 127, 129, 225, 1023, 1025, 2047,
           4097, 6657, 8192, 257, 511, 1024, 128, 65, 129)
NATURAL = (
    ('original-short', 'chatcmpl-a79346e09b9b8b8f', 2628,
     'efa8045b442eb197e5dbc3b28783add97a630c2769ddaa2e9a652a9eb128d668'),
    ('saved-long', 'chatcmpl-b170f450eca126a7', 13711,
     '178d17b9a3c3071b323dc44b1a81cf78b9e4ed7d8cd62e968ca59d3f49c132b1'),
)

def load_data(directory):
    pins = json.loads((directory / 'manifest.json').read_text())
    assert set(pins) == {'quality-repetition-requests.json',
                         'quality-repetition-renderer.json',
                         'quality-tokenization-requests-r3.json'}
    for name, expected in pins.items():
        assert digest((directory / name).read_bytes()) == expected, name
    source = json.loads((directory / 'quality-repetition-requests.json').read_text())['records']
    rows = {r['response_id']: r for r in source}
    assert len(rows) == len(source)
    oracle = {r['response_id']: r for r in json.loads((directory / 'quality-repetition-renderer.json').read_text())}
    tokenization = json.loads((directory / 'quality-tokenization-requests-r3.json').read_text())
    for label, rid, count, expected in NATURAL:
        row = rows[rid]
        assert row['original_prompt_tokens'] == count
        assert oracle[rid]['token_ids_sha256'] == expected
        body = row['body']
        assert {k: body[k] for k in ('model', 'max_tokens', 'temperature', 'top_p', 'top_k')} == dict(
            model=MODEL, max_tokens=8192, temperature=1., top_p=.95, top_k=20)
        assert not any(k in body for k in ('ignore_eos', 'min_tokens', 'cache_salt',
                                         'seed', 'stream', 'return_token_ids'))
        assert rid in tokenization
    return rows, tokenization

def mechanics_body(index, length):
    return dict(model=MODEL, prompt=[2] + [1000 + ((i + index * 7) % 97) for i in range(length - 1)],
                max_tokens=16, temperature=1., top_p=.95, top_k=20,
                return_token_ids=True, seed=9472 + index)

def check_generation(response, count, cap):
    assert isinstance(response['id'], str) and response['id']
    assert len(response['choices']) == 1
    choice = response['choices'][0]
    ids = choice['token_ids']
    assert ids and all(type(x) is int and x >= 0 for x in ids)
    assert response['usage']['prompt_tokens'] == count
    assert response['usage']['completion_tokens'] == len(ids) <= cap
    assert choice['finish_reason'] in ('stop', 'length', 'tool_calls')
    return ids

def execute(wire, root, data):
    rows, tokenization = load_data(data)
    results = []
    seen = set()
    for index, count in enumerate(LENGTHS):
        body = mechanics_body(index, count)
        started = time.time()
        response, raw = wire.call('/v1/completions', body, f'mechanics-{index:02d}')
        # Wire has persisted the full request/response before these assertions.
        ids = check_generation(response, count, 16)
        assert response['choices'][0]['finish_reason'] in ('stop', 'length')
        assert response['id'] not in seen
        seen.add(response['id'])
        row = dict(index=index, input_length=count, start=started, end=time.time(),
                   request=body, response=response, raw_response_path=str(raw))
        save(root / f'call-{index:02d}.json', row)
        results.append(dict(kind='mechanics', index=index, response_id=response['id'],
                            prompt_tokens=count, output_tokens=len(ids)))
        save(root / 'progress.json', {'results': results})
    for label, rid, count, expected in NATURAL:
        tok, _ = wire.call('/tokenize', copy.deepcopy(tokenization[rid]), label + '-tokenize')
        assert len(tok['tokens']) == count
        assert digest(json.dumps(tok['tokens'], separators=(',', ':')).encode()) == expected
        body = copy.deepcopy(rows[rid]['body'])
        body.update(seed=2026, cache_salt='r9-serving-cold-' + label, return_token_ids=True)
        response, raw = wire.call('/v1/chat/completions', body, label)
        ids = check_generation(response, count, 8192)
        assert response['id'] not in seen
        seen.add(response['id'])
        details = response['usage'].get('prompt_tokens_details')
        if details is not None and 'cached_tokens' in details:
            assert details['cached_tokens'] == 0
        result = dict(kind='natural', label=label, original_response_id=rid,
                      response_id=response['id'], prompt_tokens=count, output_tokens=len(ids),
                      raw_response_path=str(raw), returned_eos_positions=[
                          [i, x] for i, x in enumerate(ids) if x in (1, 106, 50)],
                      api_finish_reason=response['choices'][0]['finish_reason'],
                      usage=response['usage'])
        save(root / (label + '.json'), dict(result=result, request=body, response=response))
        results.append(result)
        save(root / 'progress.json', {'results': results})
    assert len(results) == len(seen) == 21
    summary = dict(unscored_policy_probe=True, status='completed', generation_requests=21, tokenization_requests=2,
                   mechanics_requests=19, natural_requests=2, results=results,
                   scope='Unscored serving integration. No cross-precision token equality or general quality acceptance.',
                   deadline_policy='One shared 2700-second client budget, including 19 mechanics and two natural requests; original R7 two-request outer limit retained and applied more broadly. Wire per-call limit remains 3600 bounded by remaining shared budget. Startup is separate CI policy.')
    save(root / 'summary.json', summary)
    return summary, root / 'summary.json'

def run_probe(server, context):
    assert server.model == MODEL
    root = Path(context.output_dir) / 'UNSCORED-r9-serving'
    root.mkdir(parents=True, exist_ok=False)
    wire = Wire(root, server.url_with_port, server.auth_token)
    wire.deadline = time.monotonic() + 2700
    try:
        summary, summary_path = execute(wire, root, Path(__file__).parent / 'r9_data')
        from .ci_returning_pressure import execute_pressure
        execute_pressure(server, Path(context.output_dir) / 'UNSCORED-c32-mixed',
                         Path(__file__).parent / 'r9_data/c32-mixed-prefixes.json')
        execute_pressure(server, Path(context.output_dir) / 'UNSCORED-c32-admission32',
                         Path(__file__).parent / 'r9_data/c32-admission32-prefixes.json')
        return summary, summary_path
    except BaseException as error:
        save(root / 'failure.json', dict(error_type=type(error).__name__, error=str(error),
                                        scope='All completed and partial Wire evidence retained; no retries.'))
        raise
