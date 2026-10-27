"""Unscored KV tight-packing correctness/pressure fixture (raw token prompts).

Serial panels are deterministic (fixed seeds, one request at a time) and are
meant to be compared token-for-token across a packed and an unpacked control
server. The concurrent churn panel exceeds the conversation pool so prefix
blocks are evicted and their IDs reused; the closing serial panel re-sends the
serial prompts after churn, and each output must equal the earlier output of
the same cache path (cold if nothing was retained, cached otherwise).
"""
import concurrent.futures
import hashlib
import json
import random
import threading
import time
from pathlib import Path

from .ci_formation_probe import check_generation
from .ci_r9_wire import Wire, save

MODEL = 'google/gemma-4-31B-it'
SERIAL_STREAMS = (30000, 46900, 12345, 61000)  # (N-1) mod 256 < 128: equal hit length at 128 and 256 granules
CHURN_STREAMS = 40
CHURN_PREFIX = 20480
CHURN_CONCURRENCY = 32
TAIL = [236778, 236771, 236785, 236745, 160, 236769, 8721, 536, 236770, 236764, 1502, 236779, 1257, 236779, 236770,
        236764, 1345, 236779, 1257, 236779, 236770, 30337, 236749, 236775, 107, 236783, 52, 236783, 51, 100, 45518, 107]


def token_ids(length, seed):
    rng = random.Random(seed)
    return [2] + [rng.randrange(1000, 200000) for _ in range(length - 1)]


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def body(ids, salt, seed, cap):
    return dict(model=MODEL, prompt=ids, max_tokens=cap, temperature=1., top_p=.95, top_k=20, seed=seed,
                cache_salt=salt, return_token_ids=True)


def cached_of(response):
    details = response['usage'].get('prompt_tokens_details') or {}
    value = details.get('cached_tokens')
    assert value is None or type(value) is int
    return value


def execute_kv_pack(server, root, deadline_s=5400):
    root.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + deadline_s
    results = []
    lock = threading.Lock()

    def call(label, ids, salt, seed, cap):
        directory = root / label
        directory.mkdir()
        wire = Wire(directory, server.url_with_port, server.auth_token)
        wire.deadline = deadline
        started = time.time()
        response, raw = wire.call('/v1/completions', body(ids, salt, seed, cap), label)
        output = check_generation(response, len(ids), cap)
        row = dict(label=label, prompt_tokens=len(ids), prompt_sha256=token_hash(ids), salt=salt, seed=seed,
                   output_ids=output, output_sha256=token_hash(output), cached_tokens=cached_of(response),
                   finish_reason=response['choices'][0]['finish_reason'], start=started, end=time.time(),
                   response_id=response['id'], raw_response_path=str(raw))
        save(directory / 'result.json', row)
        with lock:
            results.append(row)
            save(root / 'progress.json', {'results': results})
        return row

    serial = {}
    # Panel A: serial cold then immediate cached repeat, long prompts spanning many packed rows.
    for index, length in enumerate(SERIAL_STREAMS):
        ids = token_ids(length, 7100 + index)
        salt = f'kvpack-serial-{index}'
        cold = call(f'serial-{index}-cold', ids, salt, 4100 + index, 32)
        hit = call(f'serial-{index}-cached', ids, salt, 4100 + index, 32)
        serial[index] = dict(ids=ids, salt=salt, cold=cold, cached=hit)

    # Panel B: concurrent churn beyond the pool; unique salts, then returning turns.
    first = {}

    def churn(stream, turn):
        ids = token_ids(CHURN_PREFIX, 9000 + stream)
        if turn:
            ids = ids + first[stream]['output_ids'] + TAIL
        return call(f'churn-{stream:02d}-turn-{turn}', ids, f'kvpack-churn-{stream:02d}', 5000 + stream, 16)

    for turn in (0, 1):
        with concurrent.futures.ThreadPoolExecutor(max_workers=CHURN_CONCURRENCY) as pool:
            futures = {pool.submit(churn, s, turn): s for s in range(CHURN_STREAMS)}
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                if turn == 0:
                    first[futures[future]] = row

    # Panel C: serial re-send after churn. The output must equal the earlier
    # output of the same path: cold if no prefix was retained, else cached.
    checks = []
    for index, item in serial.items():
        again = call(f'serial-{index}-after-churn', item['ids'], item['salt'], 4100 + index, 32)
        cached = again['cached_tokens']
        reference = item['cold'] if not cached else item['cached']
        same_hit = cached in (None, 0) or cached == item['cached']['cached_tokens']
        checks.append(dict(index=index, cached_tokens=cached, reference=reference['label'], same_hit_length=same_hit,
                           equal=again['output_ids'] == reference['output_ids'],
                           cold_equals_cached=item['cold']['output_ids'] == item['cached']['output_ids']))
    summary = dict(complete=True, results=len(results), serial_checks=checks,
                   churn_turn1_cached=[r['cached_tokens'] for r in results if r['label'].endswith('turn-1')],
                   all_after_churn_equal=all(c['equal'] for c in checks if c['same_hit_length']),
                   scope='Unscored raw-token fixture. Serial outputs are for cross-server token comparison. '
                         'A mismatch after churn with identical hit length indicates KV corruption or ID reuse error.')
    save(root / 'summary.json', summary)
    if not summary['all_after_churn_equal']:
        raise RuntimeError('KV pack fixture: output after churn differs from same-path reference: ' + json.dumps(checks))
    return summary
