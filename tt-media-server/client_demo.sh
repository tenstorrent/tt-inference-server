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

models_resp = requests.get(f'{server}/v1/models', headers=headers).json()
model = models_resp['data'][0]['id'] if models_resp.get('data') else 'unknown'
print(f'Model: {model} | Max tokens: {max_tokens}')
print()

conversation = []
print('(conversation history is maintained across turns, type \"new\" to reset)')
print()

while True:
    try:
        prompt = input('Prompt (q to quit): ')
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if prompt.strip().lower() == 'q':
        break
    if prompt.strip().lower() == 'new':
        conversation.clear()
        print('-- conversation reset --')
        print()
        continue
    if not prompt.strip():
        continue

    conversation.append({'role': 'user', 'content': prompt})

    print('Response: ', end='', flush=True)

    start = time.perf_counter()
    ttft = None
    token_count = 0
    assistant_text = ''
    usage = {}

    resp = requests.post(
        f'{server}/v1/chat/completions',
        headers=headers,
        json={**{'model': model, 'messages': conversation, 'max_tokens': max_tokens, 'stream': True}, **sampling_overrides},
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
        delta = chunk['choices'][0].get('delta', {})
        text = delta.get('content', '')
        finish = chunk['choices'][0].get('finish_reason')
        if finish and finish == 'error':
            err_msg = text.strip() if text else 'Unknown error'
            print(f'\n\033[91m[SERVER ERROR: {err_msg}]\033[0m', flush=True)
            break
        if text and ttft is None:
            ttft = time.perf_counter() - start
        if text:
            print(text, end='', flush=True)
            assistant_text += text
            token_count += 1
        if finish:
            usage = chunk.get('usage', {})
            break

    conversation.append({'role': 'assistant', 'content': assistant_text})

    elapsed = time.perf_counter() - start
    ttft_str = f'TTFT: {ttft*1000:.0f}ms' if ttft else 'TTFT: n/a'
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    print()
    print(f'[{token_count} tokens | {ttft_str} | {elapsed:.2f}s | {token_count/elapsed:.1f} tok/s | in:{prompt_tokens} out:{completion_tokens}]')
    print()
"
