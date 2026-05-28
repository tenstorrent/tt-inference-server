# decode_scheduler_tui

Standalone TUI that drives `tt_llm_engine::scheduler::decode::DecodeScheduler`
using the same five request types `blaze_runner.cpp` uses
(`ALLOCATE` / `SUBMIT` / `CONTINUE` / `STOP` / `EVICT`). Lets you exercise the
decode scheduler against a deployed model — or a pure-software pipeline
simulator — without booting the inference server.

## Build

Built only when `ENABLE_BLAZE=ON`. First configure pulls FTXUI via
FetchContent; subsequent builds are fast.

```bash
cmake -S cpp_server -B cpp_server/build -DENABLE_BLAZE=ON
cmake --build cpp_server/build --target decode_scheduler_tui -j
```

Binary lands at `cpp_server/build/decode_scheduler_tui`.

## Run against the pipeline simulator (no hardware)

Sim mode constructs `PipelineSimulatorConfig`; useful for validating scheduler
state machines, command flow, and your own test scripts before touching a
device.

```bash
./build/decode_scheduler_tui --backend=sim --sim-decode-token=42
```

Then in the TUI:

```
> alloc                           # scheduler assigns slot 0
> submit 0 1,2,3,4 max=20 ignore_eos
> dump                            # see scheduler diagnostics in the event log
> evict 0
```

Useful sim flags:

| Flag                       | Default | Meaning                                              |
|----------------------------|---------|------------------------------------------------------|
| `--sim-num-stages=N`       | 64      | Pipeline depth (devices × stages)                    |
| `--sim-stage-us=N`         | 44      | µs per stage tick                                    |
| `--sim-decode-token=ID`    | EMPTY   | Force a fixed output token (EMPTY → use mock model)  |
| `--sim-accept-rate=F`      | 1.0     | Spec-decode accept rate (ignored if decode-token set)|

## Run against a deployed real model

Socket mode connects to a `tt-llm-engine` pipeline that's already serving on
H2D / D2H socket descriptors — typically a model you started separately. Build
must have `TtLlmEngine::Full` available (i.e. `tt-metal` is in scope at
configure time), otherwise socket mode fails at runtime.

```bash
./build/decode_scheduler_tui \
    --backend=socket \
    --h2d=tt_llm_h2d \
    --d2h=tt_llm_d2h \
    --connect-timeout-ms=30000
```

Socket flags:

| Flag                       | Default       | Meaning                                  |
|----------------------------|---------------|------------------------------------------|
| `--h2d=NAME`               | tt_llm_h2d    | Host→device socket descriptor id         |
| `--d2h=NAME`               | tt_llm_d2h    | Device→host socket descriptor id         |
| `--connect-timeout-ms=N`   | 30000         | Initial connect timeout                  |
| `--deepseek-md`            | off           | Use the DeepSeek-MD wire format          |

The descriptor names must match what the model side is publishing — these are
the same values `tt::config::blazeSocketDescriptorPrefix()` derives in the
production server.

## Scheduler params (both backends)

| Flag             | Default | Maps to                       |
|------------------|---------|-------------------------------|
| `--max-users=N`  | 64      | `SchedulerParams::max_users`  |
| `--max-seq-len=N`| 131072  | `SchedulerParams::max_seq_len`|
| `--eos-token=ID` | 1       | `SchedulerParams::eos_token`  |

## Commands

| Command                                       | Effect                                                                                  |
|-----------------------------------------------|-----------------------------------------------------------------------------------------|
| `alloc`                                       | Push `ALLOCATE`; scheduler picks a slot and returns it in the event log                 |
| `submit <slot> <t1,t2,..> [flags]`            | Push `SUBMIT` with raw prompt token IDs                                                 |
| `continue <slot> <t1,t2,..> [flags] [pos=N]`  | Push `CONTINUE` (resume an existing slot with more tokens)                              |
| `stop <slot>`                                 | Push `STOP` — cancel an in-flight generation                                            |
| `evict <slot>`                                | Push `EVICT` — release the slot                                                         |
| `dump`                                        | Stream `DecodeScheduler::dump_diagnostics` into the event log                           |
| `clear`                                       | Clear the event log pane                                                                |
| `help`                                        | Show in-TUI help                                                                        |
| `quit` / `Esc`                                | Exit                                                                                    |

Sampling flags for `submit` / `continue`:

```
max=N        max_new_tokens             temp=F       sampling temperature
top_p=F      top-p                      top_k=N      top-k (-1 = disabled)
pos=N        position_id (continue)     stop=t1,t2   stop token ids
ignore_eos   set ignore_eos             spec         enable speculative decode
disagg       mark as disaggregated decode
```

Tokens are raw integer IDs — no tokenizer is bundled. Encode externally
(e.g. with the model's tokenizer in Python) and paste the comma-separated IDs.

## Layout at a glance

```
 Decode Scheduler TUI   backend=sim(...)        RUNNING  next_req_id=N  prefill_q=N  decode_staging=N

 Slots: # State InFlight Tok Pos Gen Max EvP StP Spec a/r        (live, all max_users rows that ever became active)
 Recent tokens: slot N: id id id id ...                          (ring buffer per slot)
 Event log: [+s.ms] pushed SUBMIT ...                            (newest at bottom; capped at 500 entries)
 > <command input>
```

Slot state colors: green=DECODE, yellow=PREFILL, blue=COMPLETE, gray=INACTIVE.
The numbers in the slot table come straight from the scheduler's query API
(`get_user_state`, `get_in_flight_count`, `get_tokens_generated`,
`get_current_position`, `get_generation`, `get_max_new_tokens`,
`get_evict_pending`, `get_stop_pending`, `get_spec_accepts`,
`get_spec_rejects`) so the view is whatever the scheduler thinks is true.
