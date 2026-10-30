"""Unscored speculative-decoding fixture: exactness vs a control server, prefix-cache path, decode throughput.

Run identically against a server with speculation (GEMMA_SPEC_NGRAM=1 + speculative-config) and a control server
without it. Prompts are 8 real agent prompts (raw token ids, 9.9K-17.8K tokens) from scored Terminal-Bench /
SWE-bench runs.

Panel E (serial exactness): each prompt alone, fixed seed, T=1 / top_p 0.95 / top_k 20 (sampled) and T=0
  (greedy), 384 tokens. With the same seeds, a spec server must return exactly the control server's tokens: the
  verify samples every candidate row with the request's per-position seed and its logits are bitwise equal to plain
  decode (spec-decode LOG L8). Compared offline across the two servers.
Panel C (prefix cache): the sampled requests again with the same cache_salt (prefix hit) -> compared across
  servers the same way.
Panel T (throughput): for B in BATCHES, B concurrent requests (prompt i mod 8, distinct salts and seeds), after a
  1-token warm-up per request so the prompt is cached; wall time from first submit to last completion and total
  completion tokens. tokens/s per B, compared across servers.
"""
import concurrent.futures
import hashlib
import json
import threading
import time
from pathlib import Path

from .ci_formation_probe import check_generation
from .ci_r9_wire import Wire, save

MODEL = 'google/gemma-4-31B-it'
EXACT_TOKENS = 384
THROUGHPUT_TOKENS = 512
BATCHES = (1, 2, 4, 8, 12, 16, 24, 32)


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def load_prompts():
    data = json.loads((Path(__file__).parent / 'spec_data' / 'spec_prompts.json').read_text())
    assert len(data) == 8
    return data


def execute_spec(server, root, deadline_s=7200):
    root.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + deadline_s
    prompts = load_prompts()
    results = []
    lock = threading.Lock()

    def call(label, ids, *, salt, seed, cap, temperature=1.0):
        directory = root / label
        directory.mkdir()
        wire = Wire(directory, server.url_with_port, server.auth_token)
        wire.deadline = deadline
        body = dict(model=MODEL, prompt=ids, max_tokens=cap, temperature=temperature, top_p=.95, top_k=20,
                    seed=seed, cache_salt=salt, return_token_ids=True)
        started = time.time()
        response, raw = wire.call('/v1/completions', body, label)
        output = check_generation(response, len(ids), cap)
        details = response['usage'].get('prompt_tokens_details') or {}
        row = dict(label=label, prompt_tokens=len(ids), prompt_sha256=token_hash(ids), salt=salt, seed=seed,
                   temperature=temperature, output_ids=output, output_sha256=token_hash(output),
                   output_tokens=len(output), cached_tokens=details.get('cached_tokens'),
                   finish_reason=response['choices'][0]['finish_reason'], start=started, end=time.time(),
                   response_id=response['id'], raw_response_path=str(raw))
        save(directory / 'result.json', row)
        with lock:
            results.append(row)
            save(root / 'progress.json', {'results': results})
        return row

    # Panel E and C: serial, one request at a time.
    for i, p in enumerate(prompts):
        call(f'exact-{i}-sampled-cold', p['prompt'], salt=f'spec-exact-{i}', seed=8100 + i, cap=EXACT_TOKENS)
        call(f'exact-{i}-sampled-cached', p['prompt'], salt=f'spec-exact-{i}', seed=8100 + i, cap=EXACT_TOKENS)
        call(f'exact-{i}-greedy', p['prompt'], salt=f'spec-greedy-{i}', seed=8200 + i, cap=EXACT_TOKENS,
             temperature=0.0)

    # Panel T: concurrent throughput.
    throughput = []
    for B in BATCHES:
        ids = [prompts[i % 8]['prompt'] for i in range(B)]
        salts = [f'spec-tp-{B}-{i}' for i in range(B)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=B) as pool:
            list(pool.map(lambda i: call(f'tp-{B:02d}-{i:02d}-warm', ids[i], salt=salts[i], seed=1, cap=1),
                          range(B)))
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=B) as pool:
            rows = list(pool.map(lambda i: call(f'tp-{B:02d}-{i:02d}', ids[i], salt=salts[i], seed=9000 + 31 * B + i,
                                                cap=THROUGHPUT_TOKENS), range(B)))
        wall = time.time() - t0
        tokens = sum(r['output_tokens'] for r in rows)
        throughput.append(dict(B=B, wall_s=wall, tokens=tokens, tokens_per_s=tokens / wall,
                               cached=[r['cached_tokens'] for r in rows]))
        save(root / 'throughput.json', throughput)
    summary = dict(complete=True, results=len(results), throughput=throughput,
                   exact={r['label']: r['output_sha256'] for r in results if r['label'].startswith('exact-')},
                   scope='Unscored spec-decode fixture. Compare exact-* output hashes with the control server; '
                         'compare throughput tokens_per_s per B.')
    save(root / 'summary.json', summary)
    return summary
