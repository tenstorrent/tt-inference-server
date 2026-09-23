#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SD3.5 on BH Galaxy, 8 workers: fire 8 requests in parallel, each with a different prompt,
# negative prompt and seed, and check that they run concurrently on 8 different workers.
#
# Pass condition: 8 x HTTP 200, all 8 images decode, and the wall time for the batch is close to
# one image's run time (~4.6 s at 20 steps), not 8x. The per-image time reported is the worker's own
# "[run] executed in N seconds. SD35 inference" log line (the pipeline call: encode, denoise, VAE),
# read from the container log after the batch and paired to requests by completion order. Images are saved for eyeballing; a labelled contact
# sheet is written when Pillow is available.
#
# Prereq: the server was started with DEVICE_IDS listing eight (4,1) columns, e.g.
#   DEVICE_IDS="(0,4,12,8),(1,5,13,9),(2,6,14,10),(3,7,15,11),(27,31,23,19),(26,30,22,18),(25,29,21,17),(24,28,20,16)"
# and has logged "All workers ready".
#
# Usage: sd35_galaxy_8worker_batch_test.sh [PORT=8000] [STEPS=20]
# Env:   CONTAINER=tt-inference   container to poll for readiness and read [run] times from ("" to skip both)
#        OUT_DIR=./sd35_batch_<timestamp>   where responses and images are written
#        API_KEY=                 bearer token if the server runs without NO_AUTH=1
set -uo pipefail
PORT=${1:-8000}
STEPS=${2:-20}
CONTAINER=${CONTAINER-tt-inference}
OUT=${OUT_DIR:-./sd35_batch_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
AUTH=()
[ -n "${API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer $API_KEY")

# ---- readiness: /health is 200 as soon as the FIRST worker is ready, so wait for all 8 ----
if [ -n "$CONTAINER" ]; then
  echo -n "waiting for 'All workers ready' in $CONTAINER"
  ready=0
  for _ in $(seq 1 120); do
    # grep without -q: read the whole log so docker never sees a broken pipe under pipefail
    if docker logs "$CONTAINER" 2>&1 | grep "All workers ready" >/dev/null; then ready=1; echo " ready"; break; fi
    echo -n "."; sleep 5
  done
  [ $ready -eq 1 ] || { echo; echo "not all workers ready after 10 min; see: docker logs $CONTAINER"; exit 1; }
fi
echo "health: $(curl -s -o /dev/null -w '%{http_code}' "localhost:$PORT/health")"

# ---- 8 distinct jobs ----
PROMPTS=(
  "a red fox sitting in fresh snow at sunrise, photorealistic wildlife photo, 85mm"
  "a lighthouse on a rocky cliff in a violent storm at night, lightning, dramatic oil painting"
  "a steaming bowl of ramen on a wooden counter, overhead shot, food photography, warm light"
  "an astronaut riding a horse across a red martian desert, two moons in the sky, cinematic"
  "a cozy log cabin glowing at dusk in a snowy pine forest, soft falling snow, wide angle"
  "a neon-lit rainy street in tokyo at night, reflections on wet asphalt, 35mm film look"
  "a golden retriever puppy in a field of sunflowers, shallow depth of field, summer afternoon"
  "an ancient stone bridge over a misty river in autumn, watercolor on textured paper"
)
NEGS=("blurry, cartoon" "daylight, calm sea" "people, hands" "earth, trees" "daylight, summer" "daylight, empty street" "adult dog, night" "photorealistic, modern")
SEEDS=(11 22 33 44 55 66 77 88)

echo "firing 8 requests at once ($STEPS steps each)..."
SINCE=$(date -u +%Y-%m-%dT%H:%M:%S)
: > "$OUT/results.txt"
T0=$(date +%s.%N)
for i in 0 1 2 3 4 5 6 7; do
  (
    code=$(curl -s -o "$OUT/r$i.json" -w '%{http_code}' -X POST "localhost:$PORT/v1/images/generations" \
      -H 'Content-Type: application/json' "${AUTH[@]}" \
      -d "{\"prompt\":\"${PROMPTS[$i]}\",\"negative_prompt\":\"${NEGS[$i]}\",\"num_inference_steps\":$STEPS,\"seed\":${SEEDS[$i]}}")
    echo "$(date +%s.%N) $i ${SEEDS[$i]} $code" >> "$OUT/results.txt"
  ) &
done
wait
WALL=$(echo "$(date +%s.%N) - $T0" | bc)

# ---- per-image [run] time from the worker logs (pipeline call: encode + denoise + VAE) ----
# The log line carries no request id, so pair by completion order: the k-th response to return
# is matched with the k-th "[run] executed in" line written since the batch started.
RUNS=""
if [ -n "$CONTAINER" ]; then
  sleep 1  # let the last worker flush its log line
  RUNS=$(docker logs --since "$SINCE" "$CONTAINER" 2>&1 | grep -oE "\[run\] executed in [0-9.]+ seconds\. SD35 inference" | awk '{print $4}')
fi
printf "%-4s %-5s %-5s %s\n" "req" "seed" "http" "run (s)"
sort -n "$OUT/results.txt" | awk -v runs="$(echo "$RUNS" | tr '\n' ' ')" 'BEGIN{n=split(runs,r," ")} {printf "%-4s %-5s %-5s %s\n", $2, $3, $4, (NR<=n ? sprintf("%.2f", r[NR]) : "-")}'
n=$(echo "$RUNS" | grep -c .)
if [ "$n" -gt 0 ]; then
  echo "$RUNS" | awk -v wall="$WALL" '{s+=$1; if(min==""||$1<min)min=$1; if($1>max)max=$1} END{printf "run min/mean/max: %.2f / %.2f / %.2f s   wall for all 8: %.2fs\n", min, s/NR, max, wall}'
  [ "$n" -ne 8 ] && echo "note: expected 8 [run] lines, found $n (another request may have overlapped the batch)"
else
  printf "wall for all 8: %.2fs   (no [run] lines read: CONTAINER unset or log unavailable)\n" "$WALL"
fi

# ---- decode images; contact sheet if Pillow is present ----
python3 - "$OUT" "$STEPS" "$(IFS=,; echo "${SEEDS[*]}")" <<'EOF'
import base64, json, sys
out, steps, seeds = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
ok, raw = 0, []
for i in range(8):
    try:
        data = base64.b64decode(json.load(open(f"{out}/r{i}.json"))["images"][0])
        open(f"{out}/img{i}_steps{steps}_seed{seeds[i]}.jpg", "wb").write(data); raw.append(data); ok += 1
    except Exception as e:
        raw.append(None); print(f"r{i}: no image ({e})")
print(f"{ok}/8 images decoded -> {out}/img*.jpg")
try:
    import io
    from PIL import Image, ImageDraw
    tiles = []
    for i, d in enumerate(raw):
        im = Image.open(io.BytesIO(d)).convert("RGB") if d else Image.new("RGB", (1024, 1024), (40, 40, 40))
        t = im.resize((480, 480)); ImageDraw.Draw(t).rectangle([0, 0, 480, 26], fill=(0, 0, 0))
        ImageDraw.Draw(t).text((6, 6), f"#{i}  steps={steps}  seed={seeds[i]}", fill=(255, 255, 255)); tiles.append(t)
    sheet = Image.new("RGB", (4 * 480, 2 * 480))
    for i, t in enumerate(tiles): sheet.paste(t, ((i % 4) * 480, (i // 4) * 480))
    sheet.save(f"{out}/contact_sheet.jpg", quality=88); print(f"labelled sheet -> {out}/contact_sheet.jpg")
except ImportError:
    print("Pillow not installed; skipped contact sheet (pip install pillow to enable)")
sys.exit(0 if ok == 8 else 1)
EOF
