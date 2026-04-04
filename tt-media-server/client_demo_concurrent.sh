#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

N=${1:-2}
MAX_TOKENS=${2:-128}
STAGGER=${3:-0}
PORT=${PORT:-8000}
SERVER=${SERVER:-http://localhost:$PORT}
API_KEY=${API_KEY:-your-secret-key}

python3 -u -c "
import requests, time, json, sys, threading, os, shutil

server = '$SERVER'
api_key = '$API_KEY'
max_tokens = $MAX_TOKENS
n = $N
stagger = $STAGGER
headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

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
stagger_str = f' | Stagger: {stagger}s' if stagger > 0 else ''
print(f'Model: {model} | Max tokens: {max_tokens} | Concurrent: {n}{stagger_str}')
print()

lock = threading.Lock()
# Each stream accumulates its text; we redraw N lines in place
buffers = [''] * n
metrics = [None] * n

def get_cols():
    return shutil.get_terminal_size((120, 24)).columns

def redraw():
    \"\"\"Redraw all N stream lines in place. Caller must hold lock.\"\"\"
    cols = get_cols()
    # Move cursor up N lines, redraw each, end on the line after the last
    sys.stdout.write(f'\\033[{n}A')
    for i in range(n):
        tag = f'[{i+1}/{n}] '
        max_text = cols - len(tag)
        text = buffers[i].replace('\\n', ' ')
        if len(text) > max_text:
            text = text[:max_text - 3] + '...'
        sys.stdout.write(f'\\r\\033[K{tag}{text}\\n')
    sys.stdout.flush()

def stream_request(idx, prompt):
    start = time.perf_counter()
    ttft = None
    token_count = 0

    try:
        resp = requests.post(
            f'{server}/v1/completions',
            headers=headers,
            json={'model': model, 'prompt': prompt, 'max_tokens': max_tokens, 'stream': True},
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
                token_count += 1
                with lock:
                    buffers[idx] += text
                    redraw()
    except Exception as e:
        with lock:
            buffers[idx] += f' [ERROR: {e}]'
            redraw()
        return

    elapsed = time.perf_counter() - start
    ttft_str = f'TTFT: {ttft*1000:.0f}ms' if ttft else 'TTFT: n/a'
    if ttft and token_count > 1:
        decode_time = elapsed - ttft
        tps = (token_count - 1) / decode_time
        tps_str = f'{tps:.1f} tok/s'
    else:
        tps_str = 'n/a tok/s'
    metrics[idx] = f'{token_count} tokens | {ttft_str} | {elapsed:.2f}s | {tps_str}'

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

    # Reset buffers
    for i in range(n):
        buffers[i] = ''
        metrics[i] = None

    # Print N blank lines to reserve space for redraw
    for i in range(n):
        print(f'[{i+1}/{n}] ')

    def staggered_start(idx, prompt):
        if stagger > 0 and idx > 0:
            time.sleep(stagger * idx)
        stream_request(idx, prompt)

    threads = [threading.Thread(target=staggered_start, args=(i, prompt)) for i in range(n)]
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_elapsed = time.perf_counter() - wall_start

    # Print final results below the streaming area
    print()
    for i in range(n):
        tag = f'[{i+1}/{n}]'
        if metrics[i]:
            print(f'{tag} {metrics[i]}')
        else:
            print(f'{tag} no metrics (error?)')
    print(f'--- Wall time: {wall_elapsed:.2f}s ---')
    print()
"
