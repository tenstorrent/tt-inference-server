---
name: tts-client-bench
description: Run the client-side TTS-2 staged load test (warmup, concurrency ladder, first-chunk-vs-prompt-size probe) against a deployed /v1/audio/speech endpoint with real voice IDs, then make the client-style charts, optionally overlaying several endpoints (direct host vs ngrok vs tt-console). Use when asked to benchmark the Inworld TTS endpoint from the client side, reproduce the customer's charts, compare direct vs ngrok vs console latency, or regenerate the FC/throughput charts.
argument-hint: "[endpoint=<host[:port]>] [label=<name>] [levels=1,2,...,600]"
---

# TTS client bench (staged test + charts)

Everything lives in `scripts/tts_client_bench/` on this branch
(`ljovanovic/tts-benchmarking_tool_backup`):

| file | what it does |
|---|---|
| `tts_staged_harness.py` | the client test: `run` (3 stages -> `staged.json` + table), `report` (table + charts from a json) |
| `compare_runs.py` | overlays several `staged.json` runs: 3 charts + one table |
| `tts_load_harness.py`, `tts_load_report.py` | the request/stream code the staged harness uses (HTTPS on port 443) |
| `datasets/tts_staged_2026-10-27.json` | the replayed traffic: texts, voice sizes, per-level ramp, probe list |

Python 3.10+, stdlib only for the run. Charts need matplotlib:
`uv run --with matplotlib python ...` works on exabox without installing anything.

## 1. Ask for the endpoint first

Take it from the arguments if given. Otherwise ask in **one** AskUserQuestion
call: the endpoint, and a short label for it (used in chart legends). The
endpoint can be any of these three:

| kind | example | port | key |
|---|---|---|---|
| **direct host** | `bh-glx-110-a02u08` (the decoder host where the server container runs) | the published container port, usually `8000` (bare metal script maps `8010`) | `your-secret-key` |
| **ngrok** | `inworld-tts-3-direct.n.cloud.tenstorrent.com` | `443` | ask |
| **console** | `console.tenstorrent.com` | `443` | the `sk-tt-...` console key - ask, never commit or log it |

Port 443 switches the harness to HTTPS; anything else is plain HTTP. The path
is always `/v1/audio/speech`.

Also ask (with these defaults) only if the user did not say:
- levels: `1,2,4,8,16,32,64,128,256,512,600`
- compare against an earlier run? (a path to its `staged.json`)

## 2. Where to run it

Run from a host **in the same cluster as the server** (exabox, 110-A), so the
numbers are the server plus the endpoint path, not your home network. On
exabox, the login node is fine for small levels; for 512/600 workers use a
compute node of a job you hold:

```bash
srun --jobid=<JOB> --overlap -N1 bash -c "ulimit -u 8192; ulimit -n 65536; <command>"
```

600 workers are 600 threads plus sockets, so the ulimits matter.

## 3. Test data: real voice IDs

Requests send `{"text": ..., "speech_ids": [...]}`. The voice prompt is a list
of codec speech IDs (50 per second of audio, codebook 0..65535).

- `/data/ljovanovic/tts-voice-ids/trace_voices.json`: 53 **real** voice prompts
  (speech ID lists) taken from the captured client traffic, one per voice size
  the traffic used (177 ... 1264), plus 2000/3000/4500/6000 for the probe.
- `/data/ljovanovic/tts-voice-ids/speech_ids.json`: the per-size cut the harness
  actually sends (`{"<size>": [ids...]}`), built from the file above. Pass this
  folder as `--voice-dir` and every run reuses exactly the same voices.
- Without real IDs the harness falls back to seeded random IDs: the prompt
  length is right, but the model can't clone a random "voice", so more outputs
  run to the 120 s cap and the numbers look worse than reality. Don't use
  random IDs for anything you share.

If `/data/ljovanovic` is not reachable, copy that folder somewhere and point
`--voice-dir` at the copy (or rebuild with `--voice-ids <trace_voices.json>`
into an empty `--voice-dir`).

## 4. Run (about 35 min for the full ladder)

```bash
cd scripts/tts_client_bench
D=/data/$USER/ttbench-runs/staged-<label>-$(date +%Y%m%d-%H%M%S); mkdir -p $D
nohup python3 -u tts_staged_harness.py run \
  --host <HOST> --port <PORT> --key <KEY> \
  --levels 1,2,4,8,16,32,64,128,256,512,600 \
  --voice-dir /data/ljovanovic/tts-voice-ids \
  --out $D/staged.json --chart none > $D/run.log 2>&1 &
```

The three stages, in order:

1. **warmup**: 3 rounds of 12 simultaneous requests, each round drains before
   the next. Burns off the cold start (the first request on a cold server
   thread can take 10+ s). Not scored.
2. **ladder**: closed loop at each level for 120 s. Workers join with the same
   ~2 s ramp the real clients had; each level drains fully before the next.
   This gives req/s and first-chunk (FC) P50/P95/P99 per concurrency.
3. **probe**: one request at a time over every voice size, cancelled after 2
   chunks. This is FC vs prompt length (text + voice tokens), with no load.

Poll `run.log`: one `[cN] done` line per level, then the table. `err` must be 0.
Tell the user if a level shows errors, `120s %` above 0, or `runaway %` above a few %.

## 5. Charts

One run (3 charts next to the json: `_throughput_fc`, `_concurrency_fc`, `_probe_fc_isl`):

```bash
uv run --with matplotlib python tts_staged_harness.py report --results $D/staged.json
```

Several endpoints on the same charts (what we sent for direct vs ngrok on 2026-11-25):

```bash
uv run --with matplotlib python compare_runs.py \
  --run direct=<dir1>/staged.json --run ngrok=<dir2>/staged.json \
  --out <outdir>/direct_ngrok
# -> direct_ngrok_1_throughput_vs_fc.png, _2_fc_vs_concurrency.png, _3_fc_vs_prompt_size.png
```

It also prints one table (req/s, FC p50, FC p95 per level per run). Paste
that table with the charts.

## 6. Report

Give the user: endpoint + label, the run folder, the FC P50/P95 table, the
chart paths, and one or two lines on what changed vs the comparison run.
Targets to read against: FC P50 under 100 ms, P95 under 125 ms.
