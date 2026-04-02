#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Dispatches 4 fixed prompts via /v1/chat/completions (mimics tt-cloud-console).
# Runs 2 rounds to test prefix cache / KV cache bleed.
# Usage: ./client_demo_4prompts.sh [max_tokens] [stagger_seconds]

MAX_TOKENS=${1:-128}
STAGGER=${2:-0.5}
PORT=${PORT:-8000}
SERVER=${SERVER:-http://localhost:$PORT}
API_KEY=${API_KEY:-your-secret-key}

python3 -u -c "
import requests, time, json, sys, threading, os, shutil

server = '$SERVER'
api_key = '$API_KEY'
max_tokens = $MAX_TOKENS
stagger = $STAGGER
headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

prompts = [
    'Explain how penguins survive in Antarctica. Always use the word penguin in every sentence.',
    'Describe how submarine sonar works underwater. Always use the word submarine in every sentence.',
    'Explain how dinosaurs went extinct millions of years ago. Always use the word dinosaur in every sentence.',
    'Describe how chocolate is made from cacao beans. Always use the word chocolate in every sentence.',
]
n = len(prompts)

# Each slot tracks its conversation history (like the console does)
conversations = [[] for _ in range(n)]

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

model = os.environ.get('MODEL', 'unknown')
stagger_str = f' | Stagger: {stagger}s' if stagger > 0 else ' | No stagger'
print(f'Model: {model} | Max tokens: {max_tokens} | Concurrent: {n}{stagger_str}')
print(f'Endpoint: /v1/chat/completions (chat template, like console)')
print()

lock = threading.Lock()
buffers = [''] * n
metrics = [None] * n

def get_cols():
    return shutil.get_terminal_size((240, 24)).columns

def redraw():
    cols = get_cols()
    sys.stdout.write(f'\033[{n}A')
    for i in range(n):
        tag = f'[{i+1}] '
        max_text = cols - len(tag)
        text = buffers[i].replace('\n', ' ')
        if len(text) > max_text:
            text = text[:max_text - 3] + '...'
        sys.stdout.write(f'\r\033[K{tag}{text}\n')
    sys.stdout.flush()

def stream_request(idx, prompt):
    start = time.perf_counter()
    ttft = None
    token_count = 0

    # Build messages array like the console does: history + new user message
    messages = list(conversations[idx]) + [{'role': 'user', 'content': prompt}]

    try:
        resp = requests.post(
            f'{server}/v1/chat/completions',
            headers=headers,
            json={'model': model, 'messages': messages, 'max_tokens': max_tokens, 'stream': True},
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
            if finish:
                break
            if text and ttft is None:
                ttft = time.perf_counter() - start
            if text:
                token_count += 1
                with lock:
                    buffers[idx] += text
                    redraw()
    except Exception as e:
        with lock:
            buffers[idx] += f' [ERROR: {e}]'
            redraw()
        return

    # Append this exchange to conversation history (like console keeps context)
    conversations[idx].append({'role': 'user', 'content': prompt})
    conversations[idx].append({'role': 'assistant', 'content': buffers[idx]})

    elapsed = time.perf_counter() - start
    ttft_str = f'TTFT: {ttft*1000:.0f}ms' if ttft else 'TTFT: n/a'
    if ttft and token_count > 1:
        decode_time = elapsed - ttft
        tps = (token_count - 1) / decode_time
        tps_str = f'{tps:.1f} tok/s'
    else:
        tps_str = 'n/a tok/s'
    metrics[idx] = f'{token_count} tokens | {ttft_str} | {elapsed:.2f}s | {tps_str}'

def staggered_start(idx, prompt):
    if stagger > 0 and idx > 0:
        time.sleep(stagger * idx)
    stream_request(idx, prompt)

for loop in range(1, 3):
    print(f'========== Round {loop}/2 ==========')
    print()

    # Reset buffers (but keep conversation history for round 2)
    for i in range(n):
        buffers[i] = ''
        metrics[i] = None

    # Print N blank lines to reserve space for redraw
    for i in range(n):
        print(f'[{i+1}] ')

    threads = [threading.Thread(target=staggered_start, args=(i, prompts[i])) for i in range(n)]
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_elapsed = time.perf_counter() - wall_start

    # Print final results
    print()
    for i in range(n):
        tag = f'[{i+1}]'
        if metrics[i]:
            print(f'{tag} {metrics[i]}')
        else:
            print(f'{tag} no metrics (error?)')
    print(f'--- Wall time: {wall_elapsed:.2f}s ---')
    print()
    for i in range(n):
        print(f'=== [{i+1}] Full response ===')
        print(f'Prompt: {prompts[i]}')
        print(buffers[i].strip())
        print()
"
