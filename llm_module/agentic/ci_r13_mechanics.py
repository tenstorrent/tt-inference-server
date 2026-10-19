# SPDX-License-Identifier: Apache-2.0
"""Bounded unscored trace-serving smoke; no task or accuracy substitution."""
import json
import time
import urllib.request
from pathlib import Path


def run_probe(server, context):
    if server.model != 'google/gemma-4-31B-it':
        raise ValueError('Unexpected model')
    root = Path(context.output_dir) / 'UNSCORED-trace-serving-smoke'
    root.mkdir(parents=True, exist_ok=False)
    endpoint = server.url_with_port.rstrip('/')
    headers = {'Content-Type': 'application/json'}
    if server.auth_token:
        headers['Authorization'] = 'Bearer ' + server.auth_token

    def call(index, length):
        # Distinct synthetic token streams exercise runtime mechanics only.
        prompt = [2] + [1000 + ((i + index * 7) % 97) for i in range(length - 1)]
        body = {'model': server.model, 'prompt': prompt, 'max_tokens': 16,
                'temperature': 1.0, 'top_p': 0.95, 'top_k': 20,
                'return_token_ids': True, 'seed': 9472 + index}
        started = time.time()
        req = urllib.request.Request(endpoint + '/v1/completions', data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=600) as response:
            result = json.load(response)
        choice = result['choices'][0]
        assert choice['finish_reason'] in ('stop', 'length')
        assert result['usage']['prompt_tokens'] == length
        assert 0 < result['usage']['completion_tokens'] <= 16
        assert choice.get('token_ids'), 'Missing output token accounting'
        assert len(choice['token_ids']) == result['usage']['completion_tokens']
        row = {'index': index, 'input_length': length, 'start': started, 'end': time.time(), 'request': body, 'response': result}
        (root / f'call-{index:02d}.json').write_text(json.dumps(row, indent=2) + '\n')
        return row

    rows = [call(0, 129), call(1, 8449)]
    lengths = [31, 33, 127, 129, 225, 1023, 1025, 2047, 4097, 6657, 8192, 257, 511, 1024, 128, 65]
    rows.extend(call(index, length) for index, length in enumerate(lengths, start=2))
    rows.append(call(18, 129))
    summary = {'unscored_policy_probe': True, 'status': 'completed', 'request_count': len(rows),
               'output_tokens': sum(r['response']['usage']['completion_tokens'] for r in rows),
               'scope': 'Synthetic API/lifetime/sampling mechanics only; no quality or scored-agent claim. Server source/trace/allocation logs require separate acceptance.',
               'chunked_prompt_tokens': 8449}
    path = root / 'summary.json'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    return summary, path
