#!/bin/bash
# Server Stress Test
# Simulates N concurrent users having independent conversations.
#
# Usage:
#   ./test_stress.sh [server_url] [num_clients] [mode]
#
# Modes:
#   light   - 5 turns, short prompts, 64 max tokens (quick sanity check)
#   medium  - 15 turns, mixed prompts, 128 max tokens (default)
#   heavy   - 50 turns, long prompts, 256 max tokens (stability soak)
#
# Examples:
#   ./test_stress.sh http://10.32.48.16:8000 4 light
#   ./test_stress.sh http://10.32.48.16:8000 8 heavy
#   ./test_stress.sh http://localhost:8000 4 medium

SERVER=${1:-http://localhost:8000}
NUM_CLIENTS=${2:-4}
MODE=${3:-medium}
API_KEY=${API_KEY:-your-secret-key}
RESULTS_DIR="/tmp/stress_test_$(date +%s)"
mkdir -p "$RESULTS_DIR"

case "$MODE" in
    light)  NUM_TURNS=5;  MAX_TOKENS=64;  DELAY=0 ;;
    medium) NUM_TURNS=15; MAX_TOKENS=128; DELAY=1 ;;
    heavy)  NUM_TURNS=50; MAX_TOKENS=256; DELAY=2 ;;
    *) echo "Unknown mode: $MODE (use light/medium/heavy)"; exit 1 ;;
esac

echo "============================================"
echo "Server Stress Test"
echo "============================================"
echo "Server:     $SERVER"
echo "Clients:    $NUM_CLIENTS"
echo "Mode:       $MODE ($NUM_TURNS turns, $MAX_TOKENS max tokens)"
echo "Results:    $RESULTS_DIR"
echo "============================================"
echo ""

# Check server is up
if ! curl -s "$SERVER/tt-liveness" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('model_ready') else 1)" 2>/dev/null; then
    echo "ERROR: Server not ready at $SERVER"
    exit 1
fi
echo "Server is ready. Starting stress test..."
echo ""

run_client() {
    local client_id=$1
    local outfile="$RESULTS_DIR/client_${client_id}.log"
    local statsfile="$RESULTS_DIR/client_${client_id}_stats.csv"

    python3 -u -c "
import requests, json, time, sys, random

server = '$SERVER'
api_key = '$API_KEY'
client_id = $client_id
num_turns = $NUM_TURNS
max_tokens = $MAX_TOKENS
delay = $DELAY
outfile = '$outfile'
statsfile = '$statsfile'
headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

# Diverse prompt pool — mix of easy and hard requests
prompts = [
    'Hello, how are you today?',
    'Tell me a short joke',
    'What is 42 * 17?',
    'Write a haiku about rain',
    'Explain photosynthesis in one sentence',
    'Name 5 countries in Europe',
    'What is the capital of Japan?',
    'Tell me a short story about a dog',
    'What are the primary colors?',
    'Describe the water cycle briefly',
    'Who wrote Romeo and Juliet?',
    'What is machine learning in simple terms?',
    'List 3 types of clouds',
    'Tell me a fun fact about the ocean',
    'What is the speed of light?',
    'Write a limerick about a cat',
    'Explain gravity to a child',
    'What are the planets in our solar system?',
    'Tell me about the Eiffel Tower',
    'What causes thunder?',
    'Summarize the plot of Cinderella',
    'What is pi?',
    'Name 3 programming languages',
    'Describe what a rainbow looks like',
    'What is the tallest mountain on Earth?',
    'Keep going with more details',
    'Tell me more about that',
    'Can you elaborate on the previous point?',
    'Give me another example',
    'What else can you tell me about this topic?',
]

# Long prompts for heavy mode stress
long_prompts = [
    'Write a detailed essay about the history of computing from the earliest mechanical calculators through modern quantum computers, covering key figures and breakthroughs',
    'Explain in great detail how a car engine works, including the four-stroke cycle, fuel injection, cooling systems, and exhaust processing',
    'Describe the complete process of how a bill becomes a law in the United States, including committee reviews, amendments, filibuster, and presidential action',
    'Give me a comprehensive overview of World War II, covering the major theaters of war, key battles, political alliances, and the aftermath',
    'Explain the entire lifecycle of a star from nebula formation through main sequence, red giant phase, and eventual death as white dwarf or supernova',
]

messages = [{'role': 'system', 'content': 'You are a helpful assistant. Keep responses concise.'}]
stats = []
errors = 0
total_tokens = 0

with open(outfile, 'w') as f, open(statsfile, 'w') as sf:
    sf.write('turn,ttft_ms,total_ms,tokens,tok_per_sec,error\n')
    f.write(f'=== Client {client_id} | {num_turns} turns | max_tokens={max_tokens} ===\n\n')

    for turn in range(num_turns):
        # Pick prompt
        if max_tokens >= 256 and random.random() < 0.3:
            prompt = random.choice(long_prompts)
        elif turn > 0 and random.random() < 0.3:
            prompt = random.choice(prompts[-5:])  # follow-up prompts
        else:
            prompt = random.choice(prompts[:25])

        messages.append({'role': 'user', 'content': prompt})

        start = time.perf_counter()
        ttft = None
        token_count = 0
        assistant_text = ''
        error = False

        try:
            resp = requests.post(
                f'{server}/v1/chat/completions',
                headers=headers,
                json={'model': 'unused', 'messages': messages, 'max_tokens': max_tokens, 'stream': True},
                stream=True,
                timeout=300,
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
                if finish == 'error':
                    error = True
                    errors += 1
                    break
                if text and ttft is None:
                    ttft = time.perf_counter() - start
                if text:
                    assistant_text += text
                    token_count += 1
                if finish:
                    break

        except Exception as e:
            error = True
            errors += 1
            assistant_text = f'[ERROR: {e}]'

        elapsed = time.perf_counter() - start
        tok_s = token_count / elapsed if elapsed > 0 and token_count > 0 else 0
        ttft_ms = ttft * 1000 if ttft else 0
        total_tokens += token_count

        messages.append({'role': 'assistant', 'content': assistant_text})

        f.write(f'[Turn {turn+1}] User: {prompt[:80]}...\n' if len(prompt) > 80 else f'[Turn {turn+1}] User: {prompt}\n')
        f.write(f'[Turn {turn+1}] Assistant ({token_count} tok, {elapsed:.1f}s, {tok_s:.1f} t/s): {assistant_text[:200]}\n')
        if error:
            f.write(f'[Turn {turn+1}] *** ERROR ***\n')
        f.write('\n')

        sf.write(f'{turn+1},{ttft_ms:.0f},{elapsed*1000:.0f},{token_count},{tok_s:.1f},{1 if error else 0}\n')

        # Occasionally reset conversation to avoid hitting max_model_len
        if len(messages) > 20:
            messages = [messages[0]]  # keep system prompt
            f.write('[--- conversation reset to avoid max length ---]\n\n')

        if delay > 0:
            time.sleep(random.uniform(0, delay))

    # Summary
    f.write(f'\n=== SUMMARY ===\n')
    f.write(f'Turns: {num_turns}\n')
    f.write(f'Errors: {errors}\n')
    f.write(f'Total tokens: {total_tokens}\n')

status = 'PASS' if errors == 0 else f'FAIL ({errors} errors)'
print(f'Client {client_id}: {num_turns} turns, {errors} errors, {total_tokens} tokens — {status}')
" 2>&1
}

# Launch all clients in parallel
START_TIME=$(date +%s)
PIDS=()
for i in $(seq 1 $NUM_CLIENTS); do
    run_client $i &
    PIDS+=($!)
done

# Wait for all
for pid in "${PIDS[@]}"; do
    wait $pid
done
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================"
echo "RESULTS"
echo "============================================"
echo "Duration: ${ELAPSED}s"
echo ""

# Aggregate stats
TOTAL_ERRORS=0
TOTAL_TOKENS=0
for f in "$RESULTS_DIR"/client_*_stats.csv; do
    errs=$(tail -n +2 "$f" | awk -F, '{sum+=$6} END {print sum+0}')
    toks=$(tail -n +2 "$f" | awk -F, '{sum+=$4} END {print sum+0}')
    TOTAL_ERRORS=$((TOTAL_ERRORS + errs))
    TOTAL_TOKENS=$((TOTAL_TOKENS + toks))
done

echo "Total tokens generated: $TOTAL_TOKENS"
echo "Aggregate throughput:   $(echo "$TOTAL_TOKENS $ELAPSED" | awk '{printf "%.1f tok/s", $1/$2}')"
echo "Total errors:           $TOTAL_ERRORS"
echo ""

if [ "$TOTAL_ERRORS" -eq 0 ]; then
    echo "✅ ALL CLIENTS PASSED — server is stable"
else
    echo "❌ $TOTAL_ERRORS ERRORS DETECTED — check logs in $RESULTS_DIR/"
fi
echo ""
echo "Logs: $RESULTS_DIR/"
echo "Stats: $RESULTS_DIR/client_*_stats.csv"
