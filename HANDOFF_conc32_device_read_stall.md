# Handoff: forge b32 conc=32 device-read stall (engine hangs, 0 tokens)

## Summary

Under concurrent load (e.g. 32 concurrent streaming requests), a forge single-chip LLM served via tt-media-server **intermittently wedges**: the vLLM EngineCore stops producing tokens entirely (0 output), streaming requests sit in-flight until the client times out, and the server becomes unresponsive. It is **not** an OOM, **not** a compile, and **not** a config problem — it is a **tt-metal fast-dispatch completion-queue read that never completes**: the host enqueues a device→host readback and blocks in `FDMeshCommandQueue::read_completion_queue` waiting for a completion entry the device never posts. Reproduced live (multiple times) on a warm, healthy Qwen3-8B server with a clean backtrace (below), on the **shipping chunk-1024 / gmu-0.35 production config**. It is an upstream tt-metal / tt-xla runtime bug; not fixable by chunk size or GPU_MEMORY_UTILIZATION.

## Update 2026-07-05 (overnight): reliable repro + trigger localized to the driver

New findings that supersede/sharpen the notes below (all on the local editable tt-xla `kmabee/llm_integration_july3`, no CI):

- **Reliable, not ~1/20, at high concurrency.** Full-model Qwen3-8B via tt-media-server (`launch_qwen3_8b_ci.sh`, conc=64) hung on the **first** conc=64 batch in **2/2** runs. Both gdb-confirmed real stalls (178 threads, **zero active MLIR/lowering frames** — all compile pools parked idle; only thread blocked is the `from_device → copy_completion_queue_data_into_user_space → read_completion_queue` chain). The earlier "~1 in 20+" was at lower/varied concurrency; **hang probability scales with concurrency.**
- **The trigger is HOW tt-media-server drives the engine, not the model/config/sampling.** Same model / additional_config / device / conc=64, three drivers:
  | driver | result |
  |---|---|
  | tt-media-server (custom `device_worker_dynamic_batch`) | **HANGS reliably** |
  | stock `vllm serve` (vLLM-native scheduling) | 60 continuous conc=64 runs **clean** |
  | in-process `AsyncLLMEngine`, synchronized waves | 200 conc=64 waves **clean** |
  tt-media-server's worker does `task_queue.get_many(max=32)` → fires a **burst** of up to 32 concurrent `engine.generate()` → keeps pulling more *while those run* = **continuous overlapping bursts** of admission. vLLM-native scheduling admits smoothly one-at-a-time; the wave script drains between batches. Leading hypothesis: **bursty simultaneous admission** → many concurrent prefill input-relayout `from_device` readbacks overlapping decode → the completion-queue race.
- **Additional ruled-out triggers:** `seed=None` and `repetition_penalty=1.0` (the exact hung SamplingParams: `temperature=0.6, top_p=1.0, top_k=0, repetition_penalty=1.0, seed=None, n=1`). Single-layer (`num_hidden_layers=1`) ran 200 conc=64 waves clean → the stall is **model-depth / compute-time sensitive** AND driver-pattern sensitive.
- **Compile-time note correction:** an earlier ">20 min single-layer compile" observation was a `TTXLA_LOGGER_LEVEL=DEBUG` artifact (full-IR dump per graph, ~20× slowdown). Without DEBUG, single-layer compiles in ~65s. The serving config compiles ~76 graph shapes (`num_tokens{1,128,256,512,1024} × num_reqs{1,32} × all_greedy{T,F} × apply_grammar{T,F} × prefix_chunk{T,F}`).
- **CI intermittency explained (e.g. Falcon3-7B):** identical dispatches, 1 pass / 3 fail — the pass ran its full ~57min suite (benchmarks + gpqa) without tripping the stall; the 3 fails hit it, produced 0 tokens, never recovered, and dead-hung ~6h until the workflow-timeout cancelled them (`Run tests` step, ~7h, container logs purged in the abort cascade). Same fingerprint as the confirmed Qwen3-8B stall. CI eval/benchmark concurrency sits in the *intermittent* zone; conc=64 is *near-deterministic*.

### Experiments run today (what was tried / narrowed down)

All Qwen3-8B, production config (b32 / max_model_len 40960 / chunk 1024 / gmu 0.35 / device sampling / trace / b1-prefill / BFP8), single P150, unless noted.

| driver / setup | scale | result |
|---|---|---|
| **tt-media-server** (`launch_qwen3_8b_ci.sh`) conc=64, full model | 2 runs | **HANG both — reliable, first conc=64 batch** (gdb: real stall) |
| stock `vllm serve` conc=64 continuous, full model | 60 runs (~7.7k req) | clean |
| stock `vllm serve` **eval-mimic** (OSL 1024 ignore_eos + rep_penalty 1.1 + seed + top_k) | 10 runs | clean |
| in-process `AsyncLLMEngine`, **synchronized waves**, full model | 200 waves | clean |
| in-process `AsyncLLMEngine`, synchronized waves, **single-layer** | 200 waves | clean |
| in-process `AsyncLLMEngine`, **bursty continuous** (pre-warmed), full model | 20,000 req | clean |
| in-process `AsyncLLMEngine`, **cold-start bursty** (first burst compiles under load), full model | in progress | (see progress log) |
| chunked-prefill qualification sweep, all 5 models, chunk 1024 | 5 models | 5/5 PASS, **no OOM** (config is sound) |

Takeaway: the stall reproduces **only through tt-media-server's driver**. Stock vLLM (native scheduling) and synchronized waves — even at full model, long generation, eval-style sampling, and tens of thousands of requests — do **not** reproduce it. So the trigger is the **admission/driving pattern**, not the model, config, sampling, generation length, or raw device concurrency (device batch is capped at b32=max_num_seqs in every case).

### Fastest reliable repro (commands)
```
# serve (from tt-xla venv, chip 0):
cd ~/tt-xla && source venv/activate && \
  /path/to/tt-inference-server/tt-media-server/launch_qwen3_8b_ci.sh
# once a single request returns text, attack (continuous conc=64) — hangs on the first batch:
OPENAI_API_KEY=your-secret-key ~/tt-xla/venv/bin/vllm bench serve --backend openai-chat \
  --endpoint /v1/chat/completions --model Qwen/Qwen3-8B --host 127.0.0.1 --port 8019 \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --max-concurrency 64 --num-prompts 128 --extra-body '{"truncate_prompt_tokens":"1024","max_tokens":128}'
```
Verify a real stall (not a slow compile): `gdb -p <VLLM::EngineCore pid> -batch -ex "thread apply all bt"` → **zero active MLIR/lowering frames** + a thread in `read_completion_queue`/`from_device`. Recover: `pkill -9 -f "VLLM::EngineCore"; fuser -k 8019/tcp; tt-smi -r`.

### Pure-tt-xla repro status (no tt-media-server)
Not yet achieved via stock vLLM. Scripts + live status in `~/tt-xla/HANG_4521_REPRO_README.md` (`serve_forge_qwen3_8b.sh`, `tt_xla_hang_repro.py` with `HANG_DRIVE=continuous`/`HANG_SKIP_WARMUP=1`). Open question being tested: whether **concurrent compile + execute** (a new graph shape compiling while other requests run — which is what the first tt-media-server batch does) is the missing ingredient, vs. something specific to tt-media-server's multiprocessing worker/IPC path.

### Recommended next steps for the assignee (updated)
- **Upstream tt-metal, actionable frame:** `buffer_dispatch::copy_completion_queue_data_into_user_space` (under `FDMeshCommandQueue::read_completion_queue`) blocks forever during a `from_device` readback. Ask whether the fast-dispatch completion queue can drop/lose a completion (or deadlock) when many device→host readbacks are enqueued concurrently.
- **Reproduce it minimally at the tt-metal level** with the pattern that triggers it: **bursty, overlapping concurrent `enqueue_read`/`from_device`** (many admitted near-simultaneously), ideally with a graph being programmed/compiled at the same time — this matches tt-media-server's `device_worker_dynamic_batch` (`get_many(32)` → burst of `generate()` → keep pulling). Smooth one-at-a-time admission (vLLM-native) does NOT trip it.
- **Confirm whether concurrent first-compile is required** (cold-start test) — if so, the fix likely lives in the compile/dispatch interleave, not steady-state decode.
- **Interim (tt-inference-server side):** it is now reproducible on `main`/nightly (all 5 forge models incl. Falcon3-7B); expect nightly forge-LLM runs red until the upstream fix. Capping served concurrency below the trip point is the only local mitigation and is not desirable for production.

## Symptom

- conc=32 streaming batch → all 32 requests log `Starting streaming` (32 admitted), then the engine goes idle (`Worker health check: 0 dead workers found` every 30s) with **zero** generation and **0 completions**, until the client aborts at its read timeout.
- **Low-probability / intermittent**, not reliably triggered: ~20 consecutive conc=64 runs (16 plain + 4 with a mid-stream client kill) passed cleanly, then a later back-to-back run hung. Rough rate ≈ 1 in 20+ runs. Running conc=32/64 `vllm bench serve` back-to-back and just repeating is the repro; a mid-stream client abort is **not** required (the abort-poison hypothesis was tested and disproven — 4/4 abort cycles passed).
- After the hang the EngineCore process is **alive but blocked** (`Sl` state, not spinning — it is a blocked completion-queue read, not compute). `tt-smi -s` still returns healthy telemetry during the wedge (ARCCLK ticking, `DDR_STATUS 0x55555555`, heartbeat advancing, ~61 W) — so the chip/ARC is alive; only the fast-dispatch completion queue is wedged. Recovery needs `kill -9` the EngineCore + `tt-smi -r`.

## Root cause (captured live via gdb)

The stuck worker thread is blocked reading a tensor **off** the device during executable execution (a `from_device` / `.cpu()` in the graph-execute path), waiting on a fast-dispatch completion-queue entry that never arrives:

```
xla::PjRtCApiLoadedExecutable::ExecutePortable
 -> tt::pjrt::FlatbufferLoadedExecutableInstance::execute -> prepareInputTensor
  -> tt::pjrt::PjrtTensor::ensure_layout -> tt::runtime::toLayout
   -> tt::runtime::ttnn::LayoutConverter::convertTensorLayout / handleDeviceInputLayoutNoTypecast
    -> ttnn::operations::core::from_device -> tt::tt_metal::Tensor::cpu -> tt::tt_metal::cpu
     -> tt::tt_metal::enqueue_read_tensor -> MeshCommandQueueBase::enqueue_read
      -> MeshCommandQueueBase::enqueue_read_shards_nolock
       -> FDMeshCommandQueue::finish_nolock -> FDMeshCommandQueue::read_completion_queue
        -> copy_buffer_data_to_user_space -> copy_completion_queue_data_into_user_space   <-- BLOCKED
           (buffer_dispatch::copy_completion_queue_data_into_user_space — waiting on a
            device completion entry that is never posted)
```

The vLLM main thread is correspondingly parked in `torch_xla XLATensor::ToTensor -> LazyGraphExecutor::DeviceLocker::Barrier` (a `condition_variable::wait`) waiting on that device op. The tt-metal/LLVM compile threadpools are idle (`tf::Executor::_wait_for_task`, `libTTMLIRCompiler`) — confirming this is **not** a compile. The deepest frame (`read_completion_queue` / `copy_completion_queue_data_into_user_space`) is the actionable one: a device→host DMA readback was enqueued but its completion is never signalled.

## Where it was originally found (CI)

tt-shield "On dispatch" (workflow 154042663), forge-vllm-plugin, p150, against branch `kmabee/issue_4496_forge_llm_production_settings.testing` (production-like settings: b32, high seq len, chunked prefill, b1-prefill):

- Qwen3-8B release — FAILED: https://github.com/tenstorrent/tt-shield/actions/runs/28696213435
- Falcon3-7B-Instruct release — FAILED: https://github.com/tenstorrent/tt-shield/actions/runs/28696216594
- Llama-3.2-3B-Instruct release — FAILED: https://github.com/tenstorrent/tt-shield/actions/runs/28697131183
- Qwen3-4B release — PASSED (smaller model, did not trip it): https://github.com/tenstorrent/tt-shield/actions/runs/28697130201

In CI the symptom surfaces as a benchmark **streaming timeout** (per-chunk `request_processing_timeout_seconds`, 3000s) with the model producing no tokens. Reference: an equivalent Falcon3-7B nightly WITHOUT the production settings passed (https://github.com/tenstorrent/tt-shield/actions/runs/28685775494), i.e. the trigger correlates with the higher-concurrency / production config, not the model per se.

## Local reproduction

Environment: QB2 P150 box (chip 0), tt-xla local editable venv, branch `kmabee/llm_integration_july3`. CI wheel equivalent = tt-forge-version `f631d5b1279d0a0f334f0afcc6e4e519bc155461`.

1. Launch the server (from the tt-xla venv, pinned to chip 0, port 8019):

```
cd /home/kmabee/tt-xla && source venv/activate && \
/home/kmabee/tt-inference-server/tt-media-server/launch_qwen3_8b_ci.sh
```

Config it serves (Qwen3-8B forge): b32, max_model_len 40960, PREFILL_CHUNK_SIZE=1024, GPU_MEMORY_UTILIZATION=0.35, opt=1, device sampling, trace, b1-prefill (MIN_NUM_SEQS=1 / PREFILL_BATCH_THRESHOLD=16), bfp8 weights+KV. (The b32/chunk-2048 variant `launch_qwen3_8b_b32_chunk2048.sh` reproduces the same class of hang; it additionally needs gmu<=0.15 to avoid a separate warmup DRAM-OOM.)

2. Wait until a real generate returns text (not just a 200 on /v1/models) so the server is **warm** — otherwise the first large-ISL request pays a one-time cold compile that a short client timeout can misread as a hang (see the false-alarm note below). Then drive conc=32/64 streaming **repeatedly in a loop** — it is ~1 in 20+, so a few runs won't do it; expect to run it ~20–40× back-to-back:

```
for i in $(seq 1 40); do
  OPENAI_API_KEY=your-secret-key timeout 400 /home/kmabee/tt-xla/venv/bin/vllm bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --model Qwen/Qwen3-8B --host 127.0.0.1 --port 8019 \
    --dataset-name random --random-input-len 1024 --random-output-len 128 \
    --max-concurrency 32 --num-prompts 64 \
    --extra-body '{"truncate_prompt_tokens":"1024","max_tokens":128}' \
    > /tmp/run_$i.log 2>&1
  grep -q "Successful requests" /tmp/run_$i.log && echo "run $i PASS" || { echo "run $i HANG"; break; }
done
```

Hang = the run makes no progress (0 tokens); the server log shows `Starting streaming` ×32 then only `Worker health check` lines (32 admitted, 0 completed). A completed run prints a normal `Serving Benchmark Result` (e.g. 64/64, ~42s, high TTFT ~33s but no hang).

**False-alarm to exclude:** a *short* client `timeout` (e.g. `timeout 300`) on a **cold** server can abort during the one-time first-request MLIR/kernel compile and look like a hang — that is NOT this bug. The real stall is: server already warm, 0 tokens indefinitely, and the gdb signature below (`read_completion_queue`). Always grep the serve log for `TT_FATAL`/`Out of Memory` first to rule out OOM.

3. When wedged, confirm the root cause: `gdb -p <live EngineCore pid> -batch -ex "thread apply all bt"` (pick the non-zombie `VLLM::EngineCore`) and look for the `read_completion_queue` / `enqueue_read_shards_nolock` / `from_device` chain above; compile threadpools should be idle. Recover with `pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "uvicorn main:app"` then `tt-smi -r` (the FD queue stays wedged until the process is killed + device reset).

## What is ruled out

- Not OOM: no `Out of Memory` / `TT_FATAL` in the serve log at hang time (grep the serve log — do this first).
- Not compile: compile threadpools idle; no `Compiling graph` / IR at request time with `TTXLA_LOGGER_LEVEL=DEBUG`.
- Not chunk-size / gmu specific: **confirmed live on the shipping chunk-1024 / gmu-0.35 config** (and also seen on chunk-2048 / gmu-0.15); changing them does not fix it.
- Once warm and not wedged, the config serves correctly (conc=32/64 completes 64/64 — 20+ clean runs before one hung), so this is a rare runtime concurrency/queue race, not a bad config.
- Not the v2 refactor: the stall is in the tt-metal serving path (EngineCore), which v2 orchestration does not touch. (Separately, a short client `timeout 300` on a cold server produced *false* hangs earlier — a tooling artifact, not a server/v2 timeout; the server's real request bound is `request_processing_timeout_seconds`=3000s.)

## Suggested next steps for the assignee

- File upstream (tt-metal) with the backtrace above: `FDMeshCommandQueue::read_completion_queue` / `buffer_dispatch::copy_completion_queue_data_into_user_space` blocking forever during a `from_device` readback under concurrent execution — a device→host DMA whose completion entry is never posted. Ask whether the fast-dispatch completion queue can drop/lose a completion (or deadlock) under concurrent in-flight requests. This is the actionable frame; `enqueue_read_shards_nolock` is just the caller.
- It is a **low-probability race** (~1 in 20+ conc=32/64 runs), independent of a client abort (abort-poison disproven). To characterize: loop the repro many times and correlate with total in-flight count / a specific graph shape; consider a tt-metal-level stress test of concurrent `enqueue_read` if reproducing through vLLM is too slow.
- Interim mitigation to evaluate (not a fix): cap server concurrency below the failure point, and/or ensure the client/benchmark timeout tolerates it — but the underlying stuck read must be fixed upstream.

## Key artifacts / paths

- Server launchers: `tt-media-server/launch_qwen3_8b_ci.sh`, `tt-media-server/launch_qwen3_8b_b32_chunk2048.sh` (+ `launch_falcon3_7b_ci.sh`).
- Streaming timeout knob: `tt-media-server/config/constants.py` `request_processing_timeout_seconds` (3000s); per-chunk wait in `model_services/base_service.py`.
- On a fresh box, run the tt-xla sanity tests first to confirm the stack is healthy (torch add ~20s, vLLM OPT-gen ~2min); a slow/hung warmup usually means the device needs `tt-smi -r`.
