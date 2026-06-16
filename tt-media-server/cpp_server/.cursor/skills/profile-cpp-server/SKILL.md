---
name: profile-cpp-server
description: Capture on-CPU and off-CPU flamegraphs of the running tt_media_server_cpp main (Drogon) and worker processes using Linux perf + Brendan Gregg's FlameGraph. Use when the user asks to profile the C++ server, find a performance bottleneck, identify slow code paths, capture a flamegraph, or investigate CPU usage / lock contention in BlazeRunner or the decode scheduler.
disable-model-invocation: true
---

# Profile cpp_server (flamegraph)

## Purpose

For each process (main Drogon + every worker), produce a `.folded` stack-collapsed profile and a self-contained `.svg` flamegraph that show where the C++ server is spending or losing CPU. The preferred way to view is to drag-drop the `.folded` file into [speedscope.app](https://www.speedscope.app/) (real search, sandwich view, time-order view); the SVG is a quick-look fallback that opens in any browser without internet. Two profile flavors, complementary:

- **On-CPU** — what each thread is *executing*. Wider boxes = more CPU samples. Finds hot functions and tight loops.
- **Off-CPU** — what each thread is *waiting for* (mutex, condvar, sleep, I/O). Wider boxes = more blocking events at that call path. Finds lock contention and missed wakeups.

For `tt_media_server_cpp` you usually want both: the worker's `BlazeRunner::step()` polls `try_pop_response` in a loop (`src/runtime/runners/blaze_runner/blaze_runner.cpp` around lines 200-218). An on-CPU view alone can't distinguish "hot lock under load" from "thread asleep waiting for the lock" — the off-CPU view is what proves contention.

## End-to-end workflow (do these in order)

All paths below are relative to `tt-media-server/cpp_server/`.

1. **Build the server with the BlazeRunner code path enabled.** Without `--blaze`, the runner registry falls back to `MOCK` and you'll be profiling the wrong code.

   ```bash
   cd tt-media-server/cpp_server
   ./build.sh --blaze
   ```

   Verify the binary is fresh and not stripped:

   ```bash
   file build/tt_media_server_cpp | grep "not stripped"
   ls build/tt_media_server_cpp
   ```

2. **Start the server with the mock pipeline backend, and stash its PID for step 6.** Tokenizer files must be in place (`./build.sh --blaze` fetches them; for an ad-hoc download see `build.sh:200`). Capturing `$!` immediately is the simplest way to make sure cleanup later kills the right process and not some other `tt_media_server_cpp` (`pkill -f` is too broad — see the warning in step 6).

   ```bash
   LLM_DEVICE_BACKEND=mock_pipeline ./build/tt_media_server_cpp \
       -p 8000 -h 127.0.0.1 -t 4 > server.log 2>&1 &
   SERVER_PID=$!
   echo "SERVER_PID=$SERVER_PID"

   # Wait for readiness.
   until curl -sf http://127.0.0.1:8000/tt-liveness \
            | grep -q '"model_ready":true'; do sleep 1; done

   # Confirm Blaze runner (not the MOCK fallback) was selected.
   grep -E "Creating Blaze runner|RunnerRegistry" server.log | head
   # Expect: "[RunnerRegistry] Creating Blaze runner (pipeline_manager)"
   # If you see "No factory registered for (llm, mock_pipeline); falling back
   # to MOCK", the binary was built without --blaze — go back to step 1.
   ```

3. **Drive load and capture.** A flamegraph of an idle process is mostly Tracy / metrics / epoll noise — push traffic during the capture window. The script then handles `perf record`, `stackcollapse-perf`, `flamegraph.pl`, and kernel-frame folding.

   Wrap load + capture as a single helper so the load generator covers the **whole** window (capture + the 3 s warmup) and is killed when the capture is done. Running both captures from a single long-lived load generator is brittle: if the first capture takes longer than expected, the load runs out before the second one starts (the symptom is a low-sample off-CPU profile).

   ```bash
   # Repeatable: drive load only for the capture window, then kill it.
   capture_with_load() {
       local capture_cmd="$1"     # e.g. "./flamegraph-capture.sh all 30"
       local duration="$2"        # capture seconds, must match capture_cmd
       local stop=$(( $(date +%s) + duration + 5 ))   # +5s safety margin

       LOAD_STOP=$stop python3 -c '
   import concurrent.futures, os, time, requests
   url = "http://127.0.0.1:8000/v1/chat/completions"
   headers = {"Authorization": "Bearer your-secret-key"}
   payload = {"model":"llm","messages":[{"role":"user","content":"x "*32}],
              "max_tokens":256,"stream":False}
   stop = int(os.environ["LOAD_STOP"])
   with concurrent.futures.ThreadPoolExecutor(16) as ex:
       fs = [ex.submit(lambda: requests.post(url, headers=headers, json=payload, timeout=60))
             for _ in range(16)]
       while time.time() < stop:
           done, fs = concurrent.futures.wait(fs, timeout=0.05,
                                              return_when=concurrent.futures.FIRST_COMPLETED)
           fs = list(fs) + [ex.submit(lambda: requests.post(url, headers=headers, json=payload, timeout=60))
                            for _ in done]
   ' &
       local load_pid=$!
       sleep 3                                          # let load ramp up
       eval "$capture_cmd"
       kill "$load_pid" 2>/dev/null || true
       wait "$load_pid" 2>/dev/null || true
   }

   capture_with_load "./flamegraph-capture.sh all 30" 30          # on-CPU
   capture_with_load "./flamegraph-capture-offcpu.sh all 30" 30   # off-CPU
   ```

   Each output dir contains a `<name>.folded` + `<name>.svg` per profiled process.

4. **View the result.** Preferred: open https://www.speedscope.app/ in a browser and drag-drop a `.folded` file from the output dir — real search, sandwich view (per-function callers/callees), and time-order view (real timeline). Fallback: open the matching `.svg` in any browser for a self-contained quick look.

5. **Analyze the results.** See the **Analyze the results** section below for the procedural recipe (concrete shell commands).

6. **Stop the server.** Always do this — leftover servers hold port 8000, the `/tt_worker_metrics` shared-memory segment, and the memory-management IPC queues. The next run will fail or, worse, profile a stale binary.

   ```bash
   # Preferred: use the PID stashed in step 2.
   if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
       kill -TERM "$SERVER_PID"
       for _ in $(seq 1 10); do
           kill -0 "$SERVER_PID" 2>/dev/null || break
           sleep 1
       done
       kill -KILL "$SERVER_PID" 2>/dev/null || true
   fi

   # Fallback when SERVER_PID is gone (new shell, lost variable). Only kill
   # processes whose /proc/PID/exe is the actual binary — pgrep -f / pkill -f
   # would also match any shell, editor, or `pgrep tt_media_server_cpp` whose
   # argv contains the binary name.
   for pid in $(pgrep -f tt_media_server_cpp); do
       [ "$(readlink "/proc/$pid/exe" 2>/dev/null)" = \
         "$(realpath build/tt_media_server_cpp)" ] && kill -TERM "$pid"
   done
   sleep 2
   for pid in $(pgrep -f tt_media_server_cpp); do
       [ "$(readlink "/proc/$pid/exe" 2>/dev/null)" = \
         "$(realpath build/tt_media_server_cpp)" ] && kill -KILL "$pid" 2>/dev/null || true
   done

   # Verify shutdown: no live binary, port 8000 free, /tt_worker_metrics gone.
   pgrep -af tt_media_server_cpp || echo "OK: no server processes"
   ss -tlnp 2>/dev/null | grep ':8000 ' || echo "OK: port 8000 free"
   ls /dev/shm/tt_worker_metrics 2>/dev/null || echo "OK: shm cleared"
   ```

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

## Frame pointers (enabled in the Release build)

The build adds `-fno-omit-frame-pointer` to global CXX flags (top-level `CMakeLists.txt`), so every in-tree TU keeps `rbp` as a frame-pointer register. Both capture scripts use `perf record --call-graph fp` to walk stacks through this chain — no DWARF unwinding needed.

The trade-off, for reference:

- Runtime cost on this workload: under 1% (mutex / IPC / HTTP-framing dominated, not register-pressure-bound numeric loops; numeric compute runs on TT-Metal devices). Fedora 38+ and Ubuntu 24.04+ ship distro-wide with this flag for the same reason.
- Capture-time win, measured on this binary: `perf.data` ~6–30× smaller (worker: 3.7 MB → 560 KB; main: 4.8 MB → 129 KB), capture overhead ~100× smaller (1 ring-buffer wake vs 187 for the same 15 s window), and `[unknown]` frames went from 24 to 2.
- Selective opt-out is available if a specific TU regresses measurably: `__attribute__((optimize("omit-frame-pointer")))`.

`--call-graph dwarf` is still available as a fallback (e.g. profiling a binary built without FP). When walking through libraries like `libstdc++` / `libc` / `libpthread` that aren't compiled with frame pointers, the `fp` walk stops there and lists the library as a frame — caller context above usually still resolves because the chain continues through user code. If you see truncated stacks at library boundaries that matter, fall back to `dwarf` for that capture.

## Workflow — convenience scripts (preferred)

The two wrapper scripts at the repo root (`cpp_server/`) auto-detect the main and worker PIDs by checking `/proc/<pid>/exe` against the binary (so shell wrappers, load generators, and `pgrep` itself never get picked), capture all of them in parallel, and produce both a `.folded` file (for speedscope) and a `.svg` (quick look) per process:

```bash
# On-CPU (where CPU time goes). Default: every cpp_server process, 30s.
./flamegraph-capture.sh                 # main + every worker
./flamegraph-capture.sh main 60         # main only, 60s
./flamegraph-capture.sh worker 30       # workers only
./flamegraph-capture.sh 12345 20        # one specific PID

# Off-CPU (where threads block). Same shape.
./flamegraph-capture-offcpu.sh
```

Outputs in `cpp_server/bench_results/flamegraph[_offcpu]_<timestamp>/`:
- `<name>.folded` — drag-drop into https://www.speedscope.app/ for the modern UI. Preferred.
- `<name>.svg` — self-contained flamegraph; open in any browser. Click to zoom, search box highlights symbols. Useful when you want a single file you can attach to a PR comment or chat.

## Workflow — raw perf commands

Use these when the scripts aren't checked in, the user wants to vary parameters (sampling frequency, custom events, attaching to a different binary), or you need to debug a script failure.

1. **Identify target PIDs.** The main Drogon process is the one *without* `--worker`; workers are spawned as children. Use `/proc/<pid>/exe` to filter — `pgrep -af tt_media_server_cpp | grep -v -- '--worker'` looks like it works but also matches a parent shell, a load-generator script, or even `pgrep` itself whenever the binary name appears in argv. That's how a capture can target a wrapper PID and return zero samples.

   ```bash
   pgrep -af tt_media_server_cpp        # see all matches (incl. wrappers)

   # Only PIDs whose executable IS the binary, not bash/python wrappers.
   server_pids() {
       local pid exe
       for pid in $(pgrep -f tt_media_server_cpp 2>/dev/null); do
           exe="$(readlink "/proc/$pid/exe" 2>/dev/null)"
           [[ "$exe" == */tt_media_server_cpp ]] && echo "$pid"
       done
   }
   is_worker() { tr '\0' ' ' < "/proc/$1/cmdline" | grep -q -- '--worker'; }

   MAIN=$(for p in $(server_pids); do is_worker "$p" || { echo "$p"; break; }; done)
   WORKER=$(for p in $(server_pids); do is_worker "$p" && { echo "$p"; break; }; done)
   echo "MAIN=$MAIN  WORKER=$WORKER"
   ```

   `flamegraph-capture.sh` / `flamegraph-capture-offcpu.sh` already use exactly this filter, so the wrapper scripts will pick the right PIDs on their own. The snippet above is for running `perf` by hand.

2. **Capture on-CPU samples.** 99 Hz × frame-pointer stack walking — frequent enough to be statistically meaningful, not so frequent it perturbs the workload (FP capture is essentially free; you can safely push to `-F 999` if needed). Run in parallel for both processes during steady-state load.

   ```bash
   sudo perf record -F 99 --call-graph fp -o main.data   -p $MAIN   -- sleep 30 &
   sudo perf record -F 99 --call-graph fp -o worker.data -p $WORKER -- sleep 30 &
   wait
   ```

   Use `--call-graph dwarf` instead if profiling a binary that wasn't built with `-fno-omit-frame-pointer`. See the **Frame pointers** section for the trade-off.

3. **Capture off-CPU samples.** `-e cs` (software context-switch event) records a stack every time the thread is taken off-CPU — blocking on a mutex/condvar shows up as a stack ending in `pthread_mutex_lock` / `pthread_cond_wait`. Counts events, not durations (true duration-weighted off-CPU needs BPF / `bcc/offcputime`, which is not available in our containers).

   ```bash
   sudo perf record -e cs --call-graph fp -o main_off.data   -p $MAIN   -- sleep 30 &
   sudo perf record -e cs --call-graph fp -o worker_off.data -p $WORKER -- sleep 30 &
   wait
   ```

4. **Render.** `perf script` decodes `.data` into stacks; `stackcollapse-perf.pl` rolls duplicates into counts (output suitable for both speedscope and flamegraph.pl); `flamegraph.pl` renders the quick-look SVG. The `sed` step folds the long `[[kernel.kallsyms]]` chains (caused by `kptr_restrict` in containers) into a single readable `[kernel]` frame.

   ```bash
   sudo chown $USER:$USER *.data
   FG=./build/_deps/flamegraph     # path to cloned FlameGraph repo

   perf script -i main.data | "$FG/stackcollapse-perf.pl" \
       | sed 's/\(;\[\[kernel\.kallsyms\]\]\)\+/;[kernel]/g' \
       | tee main.folded \
       | "$FG/flamegraph.pl" --title "main on-CPU" > main.svg

   perf script -i worker.data | "$FG/stackcollapse-perf.pl" \
       | sed 's/\(;\[\[kernel\.kallsyms\]\]\)\+/;[kernel]/g' \
       | tee worker.folded \
       | "$FG/flamegraph.pl" --title "worker on-CPU" > worker.svg
   ```

   Drag-drop `*.folded` into https://www.speedscope.app/ for the interactive view; open `*.svg` in a browser for a quick look. For the off-CPU SVGs add `--colors=io --countname=switches` so they're visually distinguishable.

5. **(Optional) Drive load.** A flamegraph of an idle process is mostly Tracy / metrics / epoll noise. Push representative traffic during the capture window — for `LLM_DEVICE_BACKEND=mock_pipeline` builds, the inline `python3 -c '...'` snippet in the end-to-end workflow step 3 above works. Start load first, sleep a few seconds so it ramps up, then start `perf record`.

## How to view and read the result

Two viewers, same underlying data (the `.folded` file is the source of truth; the `.svg` is rendered from it):

**Speedscope (preferred).** Open https://www.speedscope.app/ and drag-drop the `.folded` file. The page is client-side JS — your profile data does not leave the browser. Three views worth knowing:
- **Left Heavy** (default) — the aggregated flamegraph, equivalent to the SVG.
- **Sandwich** — pick a function and see *all* its callers above and callees below. Best view for "where is `pthread_mutex_lock` being called from?".
- **Time Order** — chronological timeline of samples. The x-axis finally means time. Use range-select to zoom into a slice.
- Keyboard: `1/2/3` switches view; `Cmd/Ctrl+F` searches across all views.

**SVG flamegraph (quick look).** Open the `.svg` in any browser. Useful when you want a single-file artifact or no internet.
- **x-axis is NOT time.** Each column is one stack; columns are sorted alphabetically so identical stacks merge into a single wide box. Width = relative cost (samples or events).
- **y-axis is the call stack.** Bottom frame = thread entry point; top frame = the leaf that was on-CPU (or where the thread blocked) at sample time.
- **Search box (top-right)** highlights all frames matching a regex.

## Analyze the results

Speedscope is for humans; the agent should drive analysis from the `.folded` files (the same input speedscope uses) — they're sorted-counts text and easy to grep/sort. Each line is `frame1;frame2;...;leaf COUNT`.

1. **List the hottest call paths per process.** Largest counts first. Cap to ~15 lines per file — anything below that is usually noise.

   ```bash
   cd bench_results/flamegraph_<ts>     # on-CPU run
   for f in *.folded; do
       echo "=== $f (on-CPU, top 15) ==="
       sort -t' ' -k2 -nr "$f" | head -15
   done

   cd ../flamegraph_offcpu_<ts>          # off-CPU run
   for f in *.folded; do
       echo "=== $f (off-CPU, top 15) ==="
       sort -t' ' -k2 -nr "$f" | head -15
   done
   ```

2. **Find contended locks (the most actionable signal).** A symbol that shows up wide in *both* the on-CPU and off-CPU views is a textbook contended lock: expensive when held, and threads sleep waiting for it.

   ```bash
   # Symbols that appear in BOTH on-CPU and off-CPU folded files for the worker.
   ON=bench_results/flamegraph_<ts>/worker0.folded
   OFF=bench_results/flamegraph_offcpu_<ts>/worker0.folded
   grep -oE "tt[a-zA-Z_:<>]+|DecodeScheduler::[A-Za-z_]+|BlazeRunner::[A-Za-z_]+" "$ON"  | sort -u > /tmp/on.syms
   grep -oE "tt[a-zA-Z_:<>]+|DecodeScheduler::[A-Za-z_]+|BlazeRunner::[A-Za-z_]+" "$OFF" | sort -u > /tmp/off.syms
   comm -12 /tmp/on.syms /tmp/off.syms | head -30
   ```

   For any symbol in the intersection, check its leaf frame: if it ends in `pthread_mutex_lock` in off-CPU and `pthread_mutex_unlock` in on-CPU, that's a hot contended mutex — the most common single optimization target in `BlazeRunner` / `DecodeScheduler`.

3. **Check symbol resolution quality.** Unresolved frames hide findings.

   ```bash
   for f in **/*.folded; do
       total=$(wc -l < "$f")
       unk=$(grep -c "\[unknown\]" "$f" || true)
       echo "$f: $unk / $total stacks contain [unknown]"
   done
   ```

   Expected: < 2% on the main process, ~0% on the worker. Higher means the binary was built stripped or without `-g` — go back to step 1 of the workflow.

4. **Look for suspicious leaf frames.** A few that almost always indicate a real bug:

   - **`[[vdso]]` as a top-15 leaf** — `clock_gettime` called in a tight loop. Find the parent: `grep '\[\[vdso\]\] [0-9]' <file>.folded | head`.
   - **Constructor symbols in the hot path** (e.g. `SamplingParams::SamplingParams`) — per-iteration object construction; usually fixable by hoisting or moving to a member.
   - **`spdlog::sinks::ansicolor_sink::log → write`** wide on the main process — synchronous logging at INFO is bottlenecking request handling. Lower log level or switch to an async sink.
   - **`prometheus::detail::CKMSQuantiles::insert`** wide on `DrogonIoLoop` — histogram updates are expensive; consider dropping high-cardinality labels or moving to summary-only.
   - **Tracy_Sampling / Tracy_Profiler / Tracy_DXT1 / Tracy_Symbol_Wo** in the top of an off-CPU view — Tracy was compiled in. Harmless in off-CPU (those threads are mostly asleep), but if they show up wide *on-CPU* the build needs to drop `--tracy`.

5. **Map symbols back to source.** For any finding, find the file it lives in so the user can act:

   ```bash
   # Cross-repo grep (tt-media-server + tt-llm-engine + tt-metal):
   grep -rn "DecodeScheduler::Impl::handle_api_requests" \
       tt-media-server tt-llm-engine 2>/dev/null | head -5
   ```

   For BlazeRunner / decode scheduler frames, the source is in `tt-llm-engine` (the nested submodule under `tt-media-server/cpp_server/tt-llm-engine/`), not in `tt-media-server` itself.

## Common pitfalls

- **`perf record` runs but the output is empty** — the target process was idle. Drive load (step 3 of the end-to-end workflow) and re-capture.
- **Capture announces `main:<pid>` but the resulting profile is empty / has no `tt_media_server` frames** — the resolver picked a shell wrapper or load-generator whose argv contains `tt_media_server_cpp`. Run `readlink /proc/<pid>/exe` on the announced PID; if it points to `bash` or `python`, your script is the old version that doesn't filter by `/proc/<pid>/exe`. Update both `flamegraph-capture*.sh` to use the `list_server_pids` helper (see the **Workflow — raw perf commands** step 1 snippet) or pass the main PID explicitly: `./flamegraph-capture.sh <pid> 30`.
- **All stacks show `[unknown]`** — binary was stripped, or rebuilt without `-g`. Fix the build.
- **Kernel frames repeat 10+ times in every stack** — `kptr_restrict=1`. Apply the `sed` fold from step 4, or run `sudo sysctl -w kernel.kptr_restrict=0` on a non-container host.
- **`perf record` fails with `Permission denied`** — `perf_event_paranoid` is restrictive. Prefix with `sudo` (works inside Docker) or lower the sysctl on bare metal.
- **Worker has very few samples but is clearly busy** — process actually has multiple threads and most are blocked. Switch to the off-CPU script.
- **Tracy threads (`Tracy_Sampling`, `Tracy_Profiler`, `Tracy_DXT1`, `Tracy_Symbol_Wo`) dominate the off-CPU view** — Tracy was compiled in. They're harmless on idle-block, but their on-CPU contribution is real. Either ignore them via the search box, or rebuild without `--tracy` for a clean baseline.
- **Next run fails with port-in-use, `shm_open` errors, or "queue already exists"** — a previous server was not shut down. Run step 6 of the workflow against the leftover PID. `pkill -f tt_media_server_cpp` is a hammer, not a fix — it can kill an unrelated shell whose argv contains the binary name; prefer the `/proc/<pid>/exe`-filtered loop in step 6.
- **Second capture (off-CPU) has far fewer samples than the first** — the load generator timed out before the second capture started. Use the `capture_with_load` helper in step 3 so each capture has its own load window.

## Reporting

When summarizing findings to the user:

- State the workload (concurrency, duration, backend) — flamegraph numbers are meaningless without it.
- Cite each finding as `function → child_frame (N samples or events)`, using the exact symbol from the folded output (`sort -t' ' -k2 -nr <file>.folded | head`).
- For each finding, name the file and code path it points at (e.g. `tt_llm_engine::scheduler::decode::DecodeScheduler::Impl::handle_api_requests` → `src/decode_scheduler/...` in `tt-llm-engine`).
- Distinguish CPU cost from contention: on-CPU-only = expensive computation; on-CPU + off-CPU = contended lock; off-CPU-only = waiting on external/external event.
- Attach both the `.folded` paths (so the user can drag-drop into speedscope.app) and the `.svg` paths (for a quick look in any browser).
