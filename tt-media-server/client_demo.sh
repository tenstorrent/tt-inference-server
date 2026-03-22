#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

MAX_TOKENS=${1:-128}
PORT=${PORT:-8000}
SERVER=${SERVER:-http://localhost:$PORT}
API_KEY=${API_KEY:-your-secret-key}

python3 -u -c "
import requests, time, json, sys

server = '$SERVER'
api_key = '$API_KEY'
max_tokens = $MAX_TOKENS
headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

# Optional sampling overrides from env vars
sampling_overrides = {}
if '$TEMPERATURE' != '':
    sampling_overrides['temperature'] = float('$TEMPERATURE')
if '$REPETITION_PENALTY' != '':
    sampling_overrides['repetition_penalty'] = float('$REPETITION_PENALTY')

# Wait for server to be ready
print('Waiting for server...', end='', flush=True)
while True:
    try:
        resp = requests.get(f'{server}/tt-liveness', headers=headers, timeout=2)
        if resp.status_code == 200 and resp.json().get('model_ready'):
            break
    except (requests.ConnectionError, requests.Timeout, Exception):
        pass
    print('.', end='', flush=True)
    time.sleep(5)
print(f' ready!')

liveness = requests.get(f'{server}/tt-liveness', headers=headers).json()
model = liveness.get('model', 'unknown')
print(f'Model: {model} | Max tokens: {max_tokens}')
print()

while True:
    try:
        prompt = input('Prompt (q to quit): ')
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if prompt.strip().lower() == 'q':
        break
    if not prompt.strip():
        continue

    print('Response: ', end='', flush=True)

    start = time.perf_counter()
    ttft = None
    token_count = 0

    resp = requests.post(
        f'{server}/v1/completions',
        headers=headers,
        json={**{'model': model, 'prompt': prompt, 'max_tokens': max_tokens, 'stream': True}, **sampling_overrides},
        stream=True,
    )
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith('data: '):
            continue
        data = line.removeprefix('data: ')
        if data.strip() == '[DONE]':
            break
        chunk = json.loads(data)
        text = chunk['choices'][0].get('text', '')
        finish = chunk['choices'][0].get('finish_reason')
        if finish:
            break
        if text and ttft is None:
            ttft = time.perf_counter() - start
        if text and text != 'final_text':
            print(text, end='', flush=True)
            token_count += 1

    elapsed = time.perf_counter() - start
    ttft_str = f'TTFT: {ttft*1000:.0f}ms' if ttft else 'TTFT: n/a'
    print()
    print(f'[{token_count} tokens | {ttft_str} | {elapsed:.2f}s | {token_count/elapsed:.1f} tok/s]')
    print()
"
