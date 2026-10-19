"""Bounded natural-stopping diagnostics; raw evidence precedes interpretation."""
import copy
import hashlib
import http.client
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

SEEDS = (0, 9472, 2026)
ORDER = ('chatcmpl-81e67484eac2210f', 'chatcmpl-84e37f0209575900',
         'chatcmpl-a20968e369079456', 'chatcmpl-876dafd04fab4130')
MODEL = 'google/gemma-4-31B-it'
LIMIT = 16 * 1024 * 1024

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def load_data(directory):
    pins = json.loads((directory / 'manifest.json').read_text())
    for name, expected in pins.items():
        assert Path(name).name == name and digest((directory / name).read_bytes()) == expected, name
    source = json.loads((directory / 'requests.json').read_text())
    rows = {r['response_id']: r for r in source['records']}
    assert len(rows) == len(source['records']) == 4 and set(rows) == set(ORDER)
    tokenization = json.loads((directory / 'tokenization.json').read_text())
    oracle = {r['response_id']: r for r in json.loads((directory / 'tokenization-proof.json').read_text())['rows']}
    assert set(tokenization) == set(oracle) == set(rows)
    for rid, row in rows.items():
        body = row['body']
        assert digest(canonical(body)) == row['body_sha256']
        assert {k: body[k] for k in ('model', 'max_tokens', 'temperature', 'top_p', 'top_k')} == dict(model=MODEL, max_tokens=8192, temperature=1., top_p=.95, top_k=20)
        assert not any(k in body for k in ('ignore_eos', 'min_tokens', 'cache_salt', 'seed', 'stream', 'return_token_ids'))
        assert row['original_prompt_tokens'] == oracle[rid]['count']
    return rows, tokenization, oracle

def schedule(rows):
    for rid in ORDER:
        yield rid, 'original-absent-salt', 0, None
    for rid in ORDER:
        for seed in SEEDS:
            yield rid, 'salted-cold', seed, f'r13-anomaly-cold-{rid}-{seed}'

class Wire:
    def __init__(self, root, endpoint, auth, opener=urllib.request.urlopen):
        self.root, self.endpoint, self.opener = root, endpoint.rstrip('/'), opener
        self.headers = {'Content-Type': 'application/json'}
        if auth:
            self.headers['Authorization'] = 'Bearer ' + auth
        self.counter = 0
        self.deadline = time.monotonic() + 6 * 3600

    def call(self, route, body, label):
        index = self.counter
        self.counter += 1
        directory = self.root / f'wire-{index:03d}-{label}'
        directory.mkdir(exist_ok=False)
        request_bytes = canonical(body)
        (directory / 'request.json').write_bytes(request_bytes)
        metadata = {'route': route, 'label': label, 'started': time.time(),
                    'request_sha256': digest(request_bytes), 'status': 'started'}
        save(directory / 'metadata.json', metadata)
        raw = b''
        def read_body(response):
            nonlocal raw
            while len(raw) <= LIMIT:
                try:
                    chunk = response.read(min(65536, LIMIT + 1 - len(raw)))
                except http.client.IncompleteRead as error:
                    raw += error.partial[:LIMIT + 1 - len(raw)]
                    raise
                if not chunk:
                    break
                raw += chunk
        try:
            remaining = self.deadline - time.monotonic()
            assert remaining > 0, "six-hour diagnostic deadline exhausted"
            request = urllib.request.Request(self.endpoint + route, data=request_bytes, headers=self.headers)
            try:
                with self.opener(request, timeout=min(3600, remaining)) as response:
                    metadata['http_status'] = response.status
                    read_body(response)
            except urllib.error.HTTPError as error:
                metadata['http_status'] = error.code
                read_body(error)
                raise
            finally:
                (directory / 'response.bin').write_bytes(raw)
                metadata.update(ended=time.time(), response_sha256=digest(raw), response_bytes=len(raw))
                save(directory / 'metadata.json', metadata)
            assert len(raw) <= LIMIT, 'oversized response; retained bounded prefix'
            result = json.loads(raw)
            metadata['status'] = 'received'
            save(directory / 'metadata.json', metadata)
            return result, directory
        except BaseException as error:
            metadata.update(status='failed', error_type=type(error).__name__, error=str(error), ended=time.time())
            save(directory / 'metadata.json', metadata)
            raise

def run_probe(server, context):
    assert server.model == MODEL
    data = Path(__file__).parent / 'r13_anomaly_data'
    rows, tokenization, oracle = load_data(data)
    before = canonical(rows)
    root = Path(context.output_dir) / 'UNSCORED-r13-scored-anomaly'
    root.mkdir(parents=True, exist_ok=False)
    # Preserve exact immutable data and original absent-salt generation bodies.
    for path in data.iterdir():
        if path.is_file():
            (root / ('input-' + path.name)).write_bytes(path.read_bytes())
    wire = Wire(root, server.url_with_port, server.auth_token)
    results = []
    summary_path = root / 'summary.json'
    try:
        for index, (rid, condition, seed, salt) in enumerate(schedule(rows)):
            row = rows[rid]
            body = copy.deepcopy(row['body'])
            body.update(seed=seed, return_token_ids=True)
            if salt is not None:
                body['cache_salt'] = salt
            assert {k: v for k, v in body.items() if k not in ('seed', 'return_token_ids', 'cache_salt')} == row['body']
            token_body = copy.deepcopy(tokenization[rid])
            if salt is not None:
                token_body['cache_salt'] = salt
            tokenized, token_path = wire.call('/tokenize', token_body, f'{index:02d}-tokenize')
            ids = tokenized['tokens']
            assert isinstance(ids, list) and all(type(t) is int for t in ids)
            assert len(ids) == row['original_prompt_tokens']
            assert digest(json.dumps(ids, separators=(',', ':')).encode()) == oracle[rid]['token_ids_sha256']
            response, response_path = wire.call('/v1/chat/completions', body, f'{index:02d}-generate')
            # Wire bytes are already durable even if any following validation fails.
            assert isinstance(response.get('id'), str) and response['id']
            assert response['id'] not in {r['response_id'] for r in results}
            assert len(response['choices']) == 1
            choice = response['choices'][0]
            generated = choice.get('token_ids')
            assert isinstance(generated, list) and generated and all(type(t) is int for t in generated)
            usage = response['usage']
            assert usage['prompt_tokens'] == len(ids)
            assert usage['completion_tokens'] == len(generated) and len(generated) <= 8192
            assert choice['finish_reason'] in ('stop', 'length', 'tool_calls')
            cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
            expected_cold = condition == 'salted-cold' or index == 0
            if cached is not None:
                assert type(cached) is int and 0 <= cached <= len(ids)
                if expected_cold:
                    assert cached == 0, ('unexpected cache reuse', rid, condition, cached)
            result = {'index': index, 'original_response_id': rid, 'condition': condition,
                      'seed': seed, 'cache_salt': salt, 'response_id': response['id'],
                      'request_sha256': digest(canonical(body)), 'input_token_ids_sha256': oracle[rid]['token_ids_sha256'],
                      'expected_cold': expected_cold, 'api_cached_tokens': cached,
                      'cache_phase_proof_required': True, 'tokenization_wire': str(token_path),
                      'generation_wire': str(response_path), 'response': response}
            results.append(result)
            save(root / 'results.json', results)
            assert canonical(rows) == before, 'source request mutation'
        assert len(results) == 16
        summary = {'unscored_policy_probe': True, 'status': 'completed', 'request_count': 16,
                   'tokenization_requests': 16, 'output_tokens': sum(r['response']['usage']['completion_tokens'] for r in results),
                   'scope': 'Four saved prompts, natural stopping and8192 original limit; same-source cap enablement under sequential B1 history only. No grouped arithmetic, original RNG, original KV state, restored-cache or quality acceptance. Raw token/finish/phase/source and full output review required.',
                   'restored_prefix_tested': False, 'data_manifest_sha256': digest((data / 'manifest.json').read_bytes())}
        save(summary_path, summary)
        return summary, summary_path
    except BaseException as error:
        save(summary_path, {'unscored_policy_probe': True, 'status': 'failed', 'completed_targets': len(results),
                            'error_type': type(error).__name__, 'error': str(error), 'raw_wire_directories': wire.counter})
        raise
