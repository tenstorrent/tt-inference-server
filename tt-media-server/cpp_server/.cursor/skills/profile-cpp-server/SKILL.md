---
name: profile-cpp-server
description: Capture on-CPU and off-CPU flamegraphs of the running tt_media_server_cpp main (Drogon) and worker processes using Linux perf + Brendan Gregg's FlameGraph. Use when the user asks to profile the C++ server, find a performance bottleneck, identify slow code paths, capture a flamegraph, or investigate CPU usage / lock contention in BlazeRunner or the decode scheduler.
disable-model-invocation: true
---

# Profile cpp_server (flamegraph)

## Purpose

Produce a single self-contained SVG per process that shows where the C++ server is spending or losing CPU. Two views, complementary:

- **On-CPU** — what each thread is *executing*. Wider boxes = more CPU samples. Finds hot functions and tight loops.
- **Off-CPU** — what each thread is *waiting for* (mutex, condvar, sleep, I/O). Wider boxes = more blocking events at that call path. Finds lock contention and missed wakeups.

For `tt_media_server_cpp` you usually want both: the worker's `BlazeRunner::step()` polls `try_pop_response` in a loop (`src/runtime/runners/blaze_runner/blaze_runner.cpp` around lines 200-218). An on-CPU view alone can't distinguish "hot lock under load" from "thread asleep waiting for the lock" — the off-CPU view is what proves contention.

## Prerequisites

```bash
# perf binary (matched to running kernel)
sudo apt install -y linux-tools-$(uname -r) linux-tools-generic

# FlameGraph scripts (the script downloads these on first run; manual:)
git clone --depth 1 https://github.com/brendangregg/FlameGraph

# Binary needs frame info — already true in default builds (Release with -g,
# not stripped). Verify:
file cpp_server/build/tt_media_server_cpp | grep "not stripped"

# Tokenizer must be present for any non-trivial backend:
ls cpp_server/tokenizers/deepseek-ai/DeepSeek-R1-0528/tokenizer.json
# (build.sh fetches it; download manually with wget from HF if running ad-hoc)
```

`perf record` typically needs to run as root. On most hosts the user can lower the restriction once:

```bash
sudo sysctl -w kernel.perf_event_paranoid=1 kernel.kptr_restrict=0
```

Inside Docker `/proc/sys/kernel` is usually read-only — just prefix `perf` calls with `sudo`. Both scripts below already do.

## Workflow — convenience scripts (preferred)

The two wrapper scripts at the repo root (`cpp_server/`) auto-detect the main and worker PIDs via `pgrep`, capture all of them in parallel, and render SVGs:

```bash
# On-CPU (where CPU time goes). Default: every cpp_server process, 30s.
./flamegraph-capture.sh                 # main + every worker
./flamegraph-capture.sh main 60         # main only, 60s
./flamegraph-capture.sh worker 30       # workers only
./flamegraph-capture.sh 12345 20        # one specific PID

# Off-CPU (where threads block). Same shape.
./flamegraph-capture-offcpu.sh
```

Output: `cpp_server/bench_results/flamegraph[_offcpu]_<timestamp>/<name>.svg`. Open in any browser. Click any frame to zoom; the search box (top-right) highlights symbols matching a regex.

For the CI-style reproducible capture (fixed workload, four SVGs per run):

```bash
cd cpp_server
python3 -m pytest tests/ci/test_perf_flamegraph.py -v -s
# Outputs in tests/ci/_artifacts/test_perf_flamegraph/
#   oncpu_main.svg, oncpu_worker0.svg
#   offcpu_main.svg, offcpu_worker0.svg
#   summary.txt, server.log
```

The pytest module requires `./build.sh --blaze` + `LLM_DEVICE_BACKEND=mock_pipeline` (it starts the server itself).

## Workflow — raw perf commands

Use these when the scripts aren't checked in, the user wants to vary parameters (sampling frequency, custom events, attaching to a different binary), or you need to debug a script failure.

1. **Identify target PIDs.** The main Drogon process is the one *without* `--worker`; workers are spawned as children.

   ```bash
   pgrep -af tt_media_server_cpp        # see all
   MAIN=$(pgrep -af tt_media_server_cpp | grep -v -- '--worker' | awk '{print $1}' | head -1)
   WORKER=$(pgrep -af 'tt_media_server_cpp.*--worker' | awk '{print $1}' | head -1)
   echo "MAIN=$MAIN  WORKER=$WORKER"
   ```

2. **Capture on-CPU samples.** 99 Hz × DWARF unwinding is the standard recipe — frequent enough to be statistically meaningful, not so frequent it perturbs the workload. Run in parallel for both processes during steady-state load.

   ```bash
   sudo perf record -F 99 --call-graph dwarf -o main.data   -p $MAIN   -- sleep 30 &
   sudo perf record -F 99 --call-graph dwarf -o worker.data -p $WORKER -- sleep 30 &
   wait
   ```

3. **Capture off-CPU samples.** `-e cs` (software context-switch event) records a stack every time the thread is taken off-CPU — blocking on a mutex/condvar shows up as a stack ending in `pthread_mutex_lock` / `pthread_cond_wait`. Counts events, not durations (true duration-weighted off-CPU needs BPF / `bcc/offcputime`, which is not available in our containers).

   ```bash
   sudo perf record -e cs --call-graph dwarf -o main_off.data   -p $MAIN   -- sleep 30 &
   sudo perf record -e cs --call-graph dwarf -o worker_off.data -p $WORKER -- sleep 30 &
   wait
   ```

4. **Render to SVG.** `perf script` decodes `.data` into folded stacks; `stackcollapse-perf.pl` rolls duplicates into counts; `flamegraph.pl` renders. The `sed` step folds the long `[[kernel.kallsyms]]` chains (caused by `kptr_restrict` in containers) into a single readable `[kernel]` frame.

   ```bash
   sudo chown $USER:$USER *.data
   FG=./build/_deps/flamegraph     # path to cloned FlameGraph repo

   perf script -i main.data | "$FG/stackcollapse-perf.pl" \
       | sed 's/\(;\[\[kernel\.kallsyms\]\]\)\+/;[kernel]/g' \
       | "$FG/flamegraph.pl" --title "main on-CPU" > main.svg

   perf script -i worker.data | "$FG/stackcollapse-perf.pl" \
       | sed 's/\(;\[\[kernel\.kallsyms\]\]\)\+/;[kernel]/g' \
       | "$FG/flamegraph.pl" --title "worker on-CPU" > worker.svg
   ```

   For the off-CPU SVGs add `--colors=io --countname=switches` so they're visually distinguishable.

5. **(Optional) Drive load.** A flamegraph of an idle process is mostly Tracy / metrics / epoll noise. Push representative traffic during the capture window — for `LLM_DEVICE_BACKEND=mock_pipeline` builds, a simple concurrent POST loop against `/v1/chat/completions` works (see `cpp_server/tests/ci/test_perf_flamegraph.py` for a self-contained example using `requests` + `ThreadPoolExecutor`). Start load first, sleep a few seconds so it ramps up, then start `perf record`.

## Interpreting the flamegraph

- **x-axis is NOT time.** Each column is one stack; columns are sorted alphabetically so identical stacks merge into a single wide box. Width = relative cost (samples or events).
- **y-axis is the call stack.** Bottom frame = thread entry point; top frame = the leaf that was on-CPU (or where the thread blocked) at sample time.
- **Use search (top-right)** to highlight all frames matching a regex — e.g. `pthread_mutex_lock`, `Scheduler::`, `Drogon`. Highlighted total appears in the search status.
- **Compare on-CPU and off-CPU side-by-side** for the same symbol: a function wide in *both* views means the lock is both expensive when running *and* contended enough to put threads to sleep. That's the textbook signature of a hot contended lock.
- **`[[vdso]]` as a top leaf** usually means `clock_gettime` is being called in a tight loop. Worth tracking down the parent frame.
- **`[unknown]` frames** mean missing debug info. The default build has them resolved; if you see many, check `file build/tt_media_server_cpp | grep -i stripped`.

## Common pitfalls

- **`perf record` runs but the SVG is empty** — the target process was idle. Drive load (step 5 above) and re-capture.
- **All stacks show `[unknown]`** — binary was stripped, or rebuilt without `-g`. Fix the build.
- **Kernel frames repeat 10+ times in every stack** — `kptr_restrict=1`. Apply the `sed` fold from step 4, or run `sudo sysctl -w kernel.kptr_restrict=0` on a non-container host.
- **`perf record` fails with `Permission denied`** — `perf_event_paranoid` is restrictive. Prefix with `sudo` (works inside Docker) or lower the sysctl on bare metal.
- **Worker has very few samples but is clearly busy** — process actually has multiple threads and most are blocked. Switch to the off-CPU script.
- **Tracy threads (`Tracy_Sampling`, `Tracy_Profiler`, `Tracy_DXT1`, `Tracy_Symbol_Wo`) dominate the off-CPU view** — Tracy was compiled in. They're harmless on idle-block, but their on-CPU contribution is real. Either ignore them via the search box, or rebuild without `--tracy` for a clean baseline.

## Reporting

When summarizing findings to the user:

- State the workload (concurrency, duration, backend) — flamegraph numbers are meaningless without it.
- Cite each finding as `function → child_frame (N samples or events)`, using the exact symbol from the folded output (`sort -t' ' -k2 -nr <file>.folded | head`).
- For each finding, name the file and code path it points at (e.g. `tt_llm_engine::scheduler::decode::DecodeScheduler::Impl::handle_api_requests` → `src/decode_scheduler/...` in `tt-llm-engine`).
- Distinguish CPU cost from contention: on-CPU-only = expensive computation; on-CPU + off-CPU = contended lock; off-CPU-only = waiting on external/external event.
- Attach the SVG paths so the user can open them directly.
