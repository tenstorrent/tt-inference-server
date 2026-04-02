#!/bin/bash
# KV Cache Bleed Detection Test
# Runs 4 concurrent multi-turn conversations with distinct topics.
# Each conversation uses unique keywords that should NEVER appear in other conversations.
# Detects cross-contamination by checking for foreign keywords in responses.
#
# Usage: ./test_kv_cache_bleed.sh [server_url] [num_turns]
# Example: ./test_kv_cache_bleed.sh http://10.32.48.16:8000 10

SERVER=${1:-http://localhost:8000}
NUM_TURNS=${2:-10}
API_KEY=${API_KEY:-your-secret-key}
RESULTS_DIR="/tmp/kv_bleed_test_$(date +%s)"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "KV Cache Bleed Detection Test"
echo "============================================"
echo "Server:    $SERVER"
echo "Turns:     $NUM_TURNS per conversation"
echo "Results:   $RESULTS_DIR"
echo "============================================"
echo ""

# 4 topics with unique marker words that should never cross-contaminate
# Each topic has: name, system_prompt, prompts_pattern, detection_keywords
run_conversation() {
    local id=$1
    local topic=$2
    local system_msg=$3
    local keyword=$4
    shift 4
    local prompts=("$@")
    local outfile="$RESULTS_DIR/conv_${id}_${topic}.log"

    python3 -u -c "
import requests, json, time, sys

server = '$SERVER'
api_key = '$API_KEY'
topic = '$topic'
keyword = '$keyword'
conv_id = $id
num_turns = $NUM_TURNS
outfile = '$outfile'
headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

prompts = [
$(for p in "${prompts[@]}"; do echo "    '$p',"; done)
]

messages = [{'role': 'system', 'content': '$system_msg'}]
all_responses = []

with open(outfile, 'w') as f:
    f.write(f'=== Conversation {conv_id}: {topic} (keyword: {keyword}) ===\n\n')

    for turn in range(num_turns):
        prompt = prompts[turn % len(prompts)]
        messages.append({'role': 'user', 'content': prompt})

        resp = requests.post(
            f'{server}/v1/chat/completions',
            headers=headers,
            json={'model': 'unused', 'messages': messages, 'max_tokens': 128, 'stream': False},
            timeout=120,
        )

        if resp.status_code != 200:
            f.write(f'[Turn {turn+1}] ERROR: {resp.status_code} {resp.text[:200]}\n')
            break

        data = resp.json()
        assistant_text = data['choices'][0]['message']['content']
        messages.append({'role': 'assistant', 'content': assistant_text})
        all_responses.append(assistant_text)

        f.write(f'[Turn {turn+1}] User: {prompt}\n')
        f.write(f'[Turn {turn+1}] Assistant: {assistant_text}\n\n')

    # Write summary
    f.write(f'\n=== SUMMARY ===\n')
    f.write(f'Topic: {topic}\n')
    f.write(f'Keyword: {keyword}\n')
    f.write(f'Turns completed: {len(all_responses)}\n')

print(f'Conv {conv_id} ({topic}): {len(all_responses)} turns completed')
" 2>&1
}

# Topic 1: PENGUINS (keyword: penguin)
TOPIC1_SYSTEM="You are an expert on penguins. Only discuss penguins. Always mention the word penguin in every response."
TOPIC1_PROMPTS=(
    "Tell me about emperor penguins"
    "How do penguins survive the cold?"
    "What do penguins eat?"
    "Describe penguin mating rituals"
    "How fast can penguins swim?"
    "Tell me about baby penguins"
    "Where do penguins live?"
    "What are the biggest threats to penguins?"
    "How many species of penguins exist?"
    "Tell me a fun fact about penguins"
)

# Topic 2: VOLCANOES (keyword: volcano)
TOPIC2_SYSTEM="You are an expert on volcanoes. Only discuss volcanoes. Always mention the word volcano in every response."
TOPIC2_PROMPTS=(
    "Tell me about Mount Vesuvius volcano"
    "How do volcanoes erupt?"
    "What is volcanic lava made of?"
    "Describe the Ring of Fire volcanoes"
    "What was the biggest volcanic eruption?"
    "How do scientists predict volcanic eruptions?"
    "Tell me about underwater volcanoes"
    "What gases do volcanoes emit?"
    "How do volcanoes form islands?"
    "Tell me about dormant volcanoes"
)

# Topic 3: ORIGAMI (keyword: origami)
TOPIC3_SYSTEM="You are an expert on origami paper folding. Only discuss origami. Always mention the word origami in every response."
TOPIC3_PROMPTS=(
    "How do I make an origami crane?"
    "What paper is best for origami?"
    "Tell me about the history of origami"
    "What is modular origami?"
    "Describe complex origami techniques"
    "Who are famous origami artists?"
    "How is origami used in engineering?"
    "Tell me about origami mathematics"
    "What is wet-folding origami?"
    "Describe origami paper sizes"
)

# Topic 4: SUBMARINES (keyword: submarine)
TOPIC4_SYSTEM="You are an expert on submarines. Only discuss submarines. Always mention the word submarine in every response."
TOPIC4_PROMPTS=(
    "How do submarines dive underwater?"
    "Tell me about nuclear submarines"
    "What is life like on a submarine?"
    "How deep can submarines go?"
    "Describe submarine sonar systems"
    "Tell me about famous submarine battles"
    "How do submarines produce oxygen?"
    "What was the first submarine ever built?"
    "How do submarine crews communicate?"
    "Tell me about research submarines"
)

echo "Starting 4 concurrent conversations..."
echo ""

# Run all 4 conversations in parallel
run_conversation 1 "penguins" "$TOPIC1_SYSTEM" "penguin" "${TOPIC1_PROMPTS[@]}" &
PID1=$!
run_conversation 2 "volcanoes" "$TOPIC2_SYSTEM" "volcano" "${TOPIC2_PROMPTS[@]}" &
PID2=$!
run_conversation 3 "origami" "$TOPIC3_SYSTEM" "origami" "${TOPIC3_PROMPTS[@]}" &
PID3=$!
run_conversation 4 "submarines" "$TOPIC4_SYSTEM" "submarine" "${TOPIC4_PROMPTS[@]}" &
PID4=$!

# Wait for all to complete
wait $PID1 $PID2 $PID3 $PID4

echo ""
echo "============================================"
echo "BLEED DETECTION ANALYSIS"
echo "============================================"
echo ""

# Check each conversation for foreign keywords
BLEED_FOUND=0

check_bleed() {
    local file=$1
    local topic=$2
    local own_keyword=$3
    shift 3
    local foreign_keywords=("$@")

    echo "--- Conversation: $topic (owns: $own_keyword) ---"
    for fk in "${foreign_keywords[@]}"; do
        count=$(grep -i "Assistant:" "$file" | grep -ic "$fk" || true)
        if [ "$count" -gt 0 ]; then
            echo "  ⚠️  BLEED DETECTED: '$fk' found $count times in $topic responses!"
            grep -i "Assistant:" "$file" | grep -in "$fk" | head -3 | sed 's/^/      /'
            BLEED_FOUND=1
        fi
    done
    if [ "$BLEED_FOUND" -eq 0 ]; then
        echo "  ✓ No foreign keywords detected"
    fi
    echo ""
}

check_bleed "$RESULTS_DIR/conv_1_penguins.log" "penguins" "penguin" "volcano" "origami" "submarine"
check_bleed "$RESULTS_DIR/conv_2_volcanoes.log" "volcanoes" "volcano" "penguin" "origami" "submarine"
check_bleed "$RESULTS_DIR/conv_3_origami.log" "origami" "origami" "penguin" "volcano" "submarine"
check_bleed "$RESULTS_DIR/conv_4_submarines.log" "submarines" "submarine" "penguin" "volcano" "origami"

echo "============================================"
if [ "$BLEED_FOUND" -eq 1 ]; then
    echo "❌ BLEED DETECTED — see details above"
else
    echo "✅ NO BLEED DETECTED — all conversations isolated"
fi
echo "============================================"
echo "Full logs: $RESULTS_DIR/"
