# MiniMax-H3 stress kit

Scripts for stress testing the MiniMax-H3 video deployments (FL2VA and Ref2VA) the way
customers will use them: many requests at once, mixed shapes, realistic mistakes, long
prompts, odd keyframes. Everything is plain Python 3 (standard library only); the input
pack in `assets/` is included, so nothing has to be downloaded or generated first.

| Name | Default URL | What it does |
|---|---|---|
| `fl2va` | https://sjc3-h3-fl2va-lightx2v-d23.n.cloud.tenstorrent.com (SJC3 quad D23) | first/last keyframe + prompt → video with audio |
| `ref2va` | https://sjc3-h3-ref2va-lightx2v-c12.n.cloud.tenstorrent.com (SJC3 quad C12) | reference images/videos/audio + prompt → video with audio |

These two are the **test** deployments. The AI& quads (Q1 to Q4) are customer-facing: do not
point this kit at them without the deployment owner's agreement.

## Read this before running

- **Runs serialize on the mesh.** Each deployment generates one video at a time, about 12 jobs
  a minute. Anyone else using the same deployment during your run will wait behind you, and
  their traffic will inflate your numbers. Say in the team channel when you start and stop.
- **One run per deployment at a time.** `run.sh` takes a lock (`/tmp/h3_stress.lock`) so two
  runs on the same machine cannot overlap. It cannot see runs from other machines.
- **Customer-like mode can take the deployment down.** The server keeps a lifetime error
  counter per worker; a few requests that pass admission but fail inside the model can
  trip it, and the whole quad then needs a redeploy (details under "The worker error budget").
  The default mode is careful and will not do that. Only use `--inject all` when the owner is
  fine with a possible redeploy.
- **The deployments take their own key**, not the console.tenstorrent.com key. Keys rotate
  on redeploy.

## Setup (five minutes)

```bash
unzip h3-stress-kit.zip && cd h3-stress-kit      # or git clone / copy the folder
python3 --version                                # 3.9 or newer
cp env.example .env && $EDITOR .env              # set H3_API_KEY (and URLs if they changed)
source .env
python3 check.py
```

`check.py` prints one line per server for liveness and one for the key:

```
fl2va   liveness HTTP 200  ready=True queue=0 canary=healthy  worker error_count=0 restart_count=0
fl2va   auth     HTTP 404  ('key OK')
ref2va  liveness HTTP 405
ref2va  auth     HTTP 404  ('key OK')
```

- `liveness HTTP 200 ready=True` is what you want. `HTTP 405 "Model is not ready"` means the
  server is up but the model is still warming up (or stuck after a restart); wait, or ask in
  the channel. `503` is usually the ngrok tunnel or the server with no worker.
- `auth HTTP 404 ('key OK')` is correct (the check asks for a job id that does not exist; a
  404 proves the key was accepted). `401` means the key is wrong or rotated.
- `error_count` is the worker error budget already spent. If it is 4 or 5, do not run the
  `edge` phase (`--skip edge`) and tell the owner.

Pillow is only needed if you want to rebuild `assets/` (`pip install -r requirements.txt`).

## What you can run

### 1. The standard stress run: `run.sh`

```bash
./run.sh 30                                   # 30-minute run against both deployments, careful mode
./run.sh 30 --servers fl2va                   # one deployment only
./run.sh 10 --workers 2,2 --burst 6,4         # lighter
./run.sh 30 --servers fl2va --inject all --max-worker-errors 999   # customer-like, no guard (see above)
tail -f results/*/stress.log                  # watch it
```

`run.sh MINUTES [h3_stress.py options]` runs `h3_stress.py` unattended. One run walks six
phases whose lengths are fractions of `--minutes` (baseline 20 %, validate 7 %, sustain 48 %,
burst 10 %, recovery 10 %, edge 5 %):

| Phase | What happens | Question it answers |
|---|---|---|
| baseline | every `core` case once per server, one at a time | unloaded latency |
| validate | every `reject` and `probe` case, then cancel, double download, delete, unknown id | admission control and job lifecycle |
| sustain | `--workers` closed-loop clients per server drawing weighted random cases; a mistake request every `--neg-every` s; a status-poll hammer every `--hammer-every` s; `GET /v1/videos/jobs` every `--list-every` s | steady state under load |
| burst | `--burst` jobs submitted within one second, then drained | queueing and fairness |
| recovery | the baseline cases again | did the load leave the server slower or broken |
| edge | 7000-char prompts, durations outside 4 to 15 s | the risky shapes, run last |

A monitor thread samples `/tt-liveness` on both servers every 10 s (queue depth, model
ready, canary state, worker error and restart counts).

Options you are likely to touch (`python3 h3_stress.py --help` has all of them):

| Option | Default | Meaning |
|---|---|---|
| `--minutes` | 30 | total wall-clock budget |
| `--servers` | `fl2va,ref2va` | which deployments to drive |
| `--workers` | `4,3` | concurrent clients in sustain, per server (fl2va,ref2va) |
| `--burst` | `12,8` | burst sizes, per server |
| `--only ID,ID` | | restrict every phase to these case ids (or prefixes) |
| `--skip PHASE,...` | | skip phases, e.g. `--skip edge` |
| `--inject safe|all` | `safe` | which mistake shapes the sustain injector may send (see error budget) |
| `--max-worker-errors N` | 2 | stop sending risky shapes to a server whose error_count rose by N during the run |
| `--keep sample|all|none` | `sample` | which output MP4s to keep |
| `--no-delete-after` | | keep jobs on the server instead of deleting them after download |
| `--out DIR` | `results/` | where to write |

### 2. The input-shape probe: `budget_probe.py`

Sends one request per unusual-but-legal input shape, one at a time, and reads
`/tt-liveness` before and after each so you can see exactly which shape, if any, was counted
against the worker error budget. FL2VA only.

```bash
python3 budget_probe.py --list                     # the 28 shapes
python3 budget_probe.py --stage probe              # just the probe, 5 to 10 minutes when idle
python3 budget_probe.py --stage all --recovery-wait 900   # probe, then the budget stage
python3 budget_probe.py --only IMG-exif-rot6,PROMPT       # a subset
```

Shapes: unusual image encodings (16-bit grey, palette, RGBA, 1-bit, grey JPEG, EXIF-rotated,
GIF, TIFF, BMP, CMYK JPEG), extreme aspect ratios (4:1, 1:4, mixed first/last frames), image
URLs, explicit canvas sizes, long prompts (3000, 12,000, 20,000 chars, emoji only) and odd
seeds (-1, 2^64-1, 10^30).

The `budget` stage only runs if the probe stage found a shape that was counted. It then checks
whether good jobs reset the counter, repeats the counted shape up to `--threshold` (default 5,
the server's `max_worker_restart_count`) to see where the server stops, and waits up to
`--recovery-wait` seconds for it to come back. **This stage is designed to take the deployment
down**; run it only with the owner's agreement.

On 2026-10-12 the CMYK JPEG shape took D23 down and it did not come back by itself, so it
now runs last on purpose. To leave it out, pass `--only` with the ids you do want (prefixes
work, but `IMG-` would match it too).

### 3. Waiting for a deployment to come back: `recovery_watch.py`

```bash
python3 recovery_watch.py fl2va                 # prints each state change, exits on READY
python3 recovery_watch.py fl2va watch.jsonl     # also logs every sample
```

Handy after an outage or a redeploy, or to chain a run: `python3 recovery_watch.py fl2va && ./run.sh 30 --servers fl2va`.

### 4. Reports

Every run writes `results/<timestamp>/report.md` and `summary.json` on its own. To
re-render or to make a shareable page:

```bash
python3 report.py results/20261011_185953                                   # re-render report.md
python3 build_html_report.py results/20261011_185953 results/20261011_192611 \
    --label 20261011_185953='careful, 4 in flight' --label 20261011_192611='customer-like' \
    --intro intro.md -o report.html                                         # one page for several runs
python3 build_html_report.py results/<run> --incident results/<run>:fl2va -o report.html   # a run in which fl2va died
```

`report.html` is self-contained (inline SVG charts) and opens in any browser.

## Reading the results

`results/<timestamp>/`:

| File | Content |
|---|---|
| `report.md`, `summary.json` | headline numbers, latency by phase, baseline vs recovery, per case, admission control, lifecycle, control plane, server health, throughput per minute, errors, findings |
| `results.jsonl`, `results.csv` | one row per request: timings, HTTP code, job id, status trail, MP4 duration/tracks/size/hash, in-flight count and server queue depth at submit, pass/fail and why |
| `monitor.jsonl` | the `/tt-liveness` samples |
| `stress.log` | the live log |
| `out/*.mp4` | sampled outputs: the first good clip per case plus every bad one |
| `probe.jsonl` | (budget_probe only) liveness before and after each shape |

Timings: `queue_s` is from 202 to the first non-queued status; `gen_s` from there to
completed; `e2e_s` from POST sent to MP4 verified, so it includes upload and download. A
request expected to succeed passes when: 202, job completed, download 200, the MP4 has a
video and an audio track, and its duration is within 0.6 s of what was asked.

Numbers to compare against (FL2VA, 2026-10-11 build, idle): a 5 s clip generates in about
3.3 s, 10 s in about 6 s, 15 s in about 9 s; the server handles about 12 jobs a minute, one
at a time, strictly FIFO; extra clients only add queue time (about 4 s per job ahead).

## The worker error budget

Each deployment keeps a lifetime `error_count` per worker (`/tt-liveness` → `worker_info`).
A request that passes admission but fails inside the model adds one. When it exceeds
`max_worker_restart_count` (5) the scheduler marks the worker dead; on these quads that has
meant the whole deployment going offline until someone redeploys it. The counter does not
reset while the deployment runs.

What the harness does about it:

- `--inject safe` (default): the sustain injector only sends mistake shapes that the server
  already refused at admission during the validate phase, so the run cannot spend the budget
  by accident.
- `--max-worker-errors N` (default 2): stop sending any negative, probe or previously failing
  case to a server whose `error_count` rose by N during the run.
- Risky cases are tagged `edge` and run last. Check `error_count` with `check.py` first.
- `--inject all --max-worker-errors 999` turns all of that off. That is what real clients
  look like; use it knowingly.

History: on 2026-10-10 a keyframe with a side under 256 px and a blank prompt each counted
and took both quads down at error_count 6. The 2026-10-11 FL2VA build rejects both at
admission (422) and a customer-like run with 52 injected mistakes left the counter at 0. On
2026-10-12 a CMYK JPEG keyframe took D23 down (a different path: the container exited and
the JobSet recreated all four pods, which then never finished warmup).

## Inputs

`assets/` is synthetic and deterministic (fixed seeds), built by `make_assets.py`: keyframes
in many sizes, aspect ratios and encodings; WAV audio of 2 to 15 s plus too-short and
too-long negatives; five reference videos generated by the FL2VA server; prompts of every
length including blank and emoji; `prompts/bank.json` with 16 prompts drawn at random by
cases marked `"prompt": "bank"`. Rebuild with `python3 make_assets.py` (needs Pillow);
`--force` overwrites, `--seed-videos` regenerates the reference clips through the FL2VA
server (needs it up).

`cases.json` is the request catalogue for `h3_stress.py`: about 45 positive and 40
negative/probe shapes across both servers. Each entry names its server, task, duration,
prompt, media, expectation (`ok`, `reject`, `probe`), sustain weight and tags (`core` =
baseline and recovery, `edge` = run last, `det` = run twice to check seed determinism). Add a
case by adding an entry. `extra` is merged over the payload last, and a string `"@asset:NAME"`
inside it is replaced by that file's base64, which is how malformed bodies are built.

The probe shapes for `budget_probe.py` are defined in that script (`SHAPES`), not in
`cases.json`.

## API notes (as of the 2026-10-11 build)

- Async API: `POST /v1/videos/generations/i2v` (FL2VA) or `/ref2va` → `202 {id}` → poll
  `GET /v1/videos/generations/{id}` → `GET .../{id}/download`. `Authorization: Bearer <key>`.
- Fields the server reads: `prompt`, `image_prompts` (with `frame_pos` 0 or -1), `duration`
  (whole seconds, 4 to 15), `seed`, `aspect_ratio`, `height`, `width`. `negative_prompt` and
  `num_inference_steps` are rejected as unknown fields. `duration_seconds` is ignored.
- Keyframe base64 is capped at 10,000,000 chars. Sides under 256 px and blank prompts are
  refused at admission. A 3:1 keyframe and a text-only FL2VA request are accepted.
- `GET /v1/videos/jobs` returns every job with its full base64 inputs (tens of MB), so the
  harness deletes each job after downloading its MP4 (`--no-delete-after` keeps them).
- The ngrok tunnel occasionally returns a 503 of its own (about 1 in 700) while the server is
  fine. `ERR_NGROK_3004` for a minute or two means the server pod is being replaced.
- Liveness: `GET /tt-liveness` without auth. `405 "Model is not ready"` is the warmup gate,
  not an auth problem.

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `401 Invalid or missing API Key` everywhere | key rotated on redeploy | ask for the new key, `export H3_API_KEY=...` |
| `405 Model is not ready` for a long time | model warming up, or stuck after a pod restart | `python3 recovery_watch.py fl2va`; if it stays 405 for more than an hour, tell the owner |
| `503` on every request | no worker, or ngrok | check `/tt-liveness`; wait a minute and retry |
| `another stress run holds /tmp/h3_stress.lock` | a previous run is still going, or crashed | `ps aux \| grep h3_stress`; `rmdir /tmp/h3_stress.lock` if stale |
| `ModuleNotFoundError: PIL` | only `make_assets.py` needs it | `pip install pillow`, or just use the included `assets/` |
| `404 Video job not found` while polling | the job was deleted (by the harness after download, or by a pod restart) | expected after a delete; after a restart it means the job was lost |
| run takes much longer than `--minutes` | jobs hanging up to `--job-timeout` (600 s) | lower `--job-timeout`, or check whether the server is stuck |

## Previous rounds

- 2026-10-10, both deployments, first run and the outage: https://claude.ai/artifact/LSVDh7spM6pgiyX68hpWo8
- 2026-10-11, FL2VA after the redeploy, careful and customer-like: https://claude.ai/artifact/X8QpM8cXyitAtG4qbUdPeQ
- 2026-10-12, FL2VA input-shape probe: `budget_probe.py --stage all`; the CMYK JPEG shape took D23 down (report pending)
- Notes on the worker error budget in the server code: https://claude.ai/artifact/CSAHeaNLhoday6cGgyRwNg
- How the deployment is wired (repos, images, cluster, quad): https://claude.ai/artifact/DN6CiPCRcWd8uk9hLrhQ8V

## Files

| File | Purpose |
|---|---|
| `check.py` | pre-flight: liveness, key, error budget |
| `run.sh` | unattended stress run with a lock; builds assets if missing |
| `h3_stress.py` | the stress harness (six phases, monitor, results) |
| `budget_probe.py` | one request per unusual input shape, with the error budget read around each |
| `recovery_watch.py` | poll liveness until ready |
| `report.py` | `results/<run>` → `report.md` + `summary.json` |
| `build_html_report.py` | one or more runs → self-contained `report.html` |
| `make_assets.py` | rebuilds `assets/` |
| `cases.json` | request catalogue for `h3_stress.py` |
| `assets/` | keyframes, audio, reference videos, prompts (72 MB, included) |
| `env.example` | the three environment variables |
| `requirements.txt` | Pillow, only for `make_assets.py` |
