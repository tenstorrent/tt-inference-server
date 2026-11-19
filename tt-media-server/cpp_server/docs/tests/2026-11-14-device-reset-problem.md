# Problem: the server cannot open the device without `tt-smi -r`

Date: 2026-11-14
Machine: T3K (8 Wormhole chips), `tt_media_server_cpp` with `MODEL_RUNNER=tt-wan2.2`.

This is an analysis of a problem that came up during testing. Some conclusions are **proven** by logs and code, others are **hypotheses**. Each claim is marked accordingly.

## 1. Symptom

At server start the worker crashes during warmup, before it receives any request:

```
TT_FATAL: Read unexpected run_mailbox value from core 21-16 (assert.hpp:104)
TT_FATAL: Read unexpected run_mailbox value from core 25-16 (assert.hpp:104)
Mesh device initialization failed
Worker 0 CRASHED: VideoIpcRunner warmup failed
```

Only a chip reset (`tt-smi -r`) helps. After the reset the server starts normally.

## 2. What the error means (proven)

The error comes from tt-metal, `tt_metal/llrt/llrt.cpp` (`check_if_riscs_on_specified_core_done`). Before launching a program the host reads the **go message** (`run_mailbox`) of a core and expects one of two values: `RUN_MSG_DONE (0x0)` or the state it asks for itself (`RUN_MSG_GO 0x80`). For anything else it raises `TT_FATAL`.

The value the core actually had is visible in an older log of the Python server (from 2026-11-06):

```
Read unexpected run_mailbox value: 0x40 (expected 0x80 or 0x0)
```

`0x40` is `RUN_MSG_INIT` (`hw/inc/hostdev/dev_msgs.h`). It means the firmware on that core stayed in its initial state and never reached `DONE`. In other words, a leftover of a previous session that did not finish initialization or shutdown cleanly remained on the chip.

Cores `21-16` and `25-16` have y=16, which are (hypothesis, from the shape of the coordinates) translated coordinates of the **ethernet cores** of a Wormhole chip. On a T3K the 8 chips are connected by ethernet links, and the fabric runs on exactly those cores.

## 3. Why a reset helps

`tt-smi -r` resets the chips at hardware level: it returns the firmware and the state of every core to the start. The leftover of the old session disappears and the host again reads the expected values. The chip itself does not know that its state is stale, and tt-metal does not do its own reset when opening the device. So the stale value is treated as an error rather than something to clean up.

## 4. When the problem occurred

| Date and time | Situation before the start | Result |
|---|---|---|
| 2026-11-06 | older Python server, earlier run | same error (`0x40`), cores `21-16` and `25-16` |
| 2026-11-14 00:37 | state left from earlier sessions, first start | `run_mailbox` error |
| 2026-11-14 00:40 | start right after a worker crashed in the middle of initialization | `run_mailbox` error |
| 2026-11-14 00:50 | start after `tt-smi -r` | works |
| 2026-11-14 (after the tests) | start after a SIGTERM shutdown (a segfault in the log) | `run_mailbox` error |
| 2026-11-14 (after the cancel test) | SIGTERM shutdown, then `tt-smi -r` right away | works (the reset was done before the start, so this proves nothing about how clean the chips are after a shutdown) |

Important: the same error occurs with the Python server (06.11.), so it is **not specific to the C++ service**.

## 5. Possible causes

### 5.1 The runner never disables the fabric (hypothesis, strong lead)

`DiTRunner._configure_fabric` ([dit_runners.py:140](../../../tt_model_runners/dit_runners.py#L140)) calls `ttnn.set_fabric_config(FABRIC_1D, ...)` when opening. Closing (`BaseMetalDeviceRunner.close_device`) only calls `ttnn.close_mesh_device(...)`. `ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)` is never called.

For comparison, tt-metal's own `conftest.py` has a function `reset_fabric(...)` that sets `DISABLED` after every test, with a comment that even this is "potentially unstable". The only case in which the DiT runner sets the fabric back to `DISABLED` is when opening the mesh fails (`_initialize_mesh_device`).

If the fabric stays enabled when the mesh is closed, the ERISC cores can be left in a state the next initialization does not expect. That would match the fact that exactly the ethernet cores are affected.

### 5.2 Segfault when the worker exits (proven to exist, cause unknown)

When the server is shut down (log from 01:21):

```
Device 0: Closing mesh device...
Device 0: Successfully closed mesh device
[SingleProcessWorker] Worker 0 killed by signal: Segmentation fault
[Main] Server shutdown complete (graceful)
```

The mesh is closed cleanly, and right after that the worker process crashes. If the crash interrupts the final cleanup of the driver or runtime, the device can be left in an unexpected state. The server still logs "graceful", so the problem is not obvious. The segfault was seen on **both** clean shutdowns that were logged in this session (2 of 2), so it does not look random. The cause is not known yet.

### 5.3 Interruption in the middle of initialization (proven as a trigger)

When the worker is killed or crashes while loading the model or opening the mesh, initialization does not reach `DONE`. That left `0x40` on the cores. Worker shutdown in the code is: `SIGTERM`, a short wait, then `SIGKILL` (`single_process_worker.cpp`, around lines 207 to 217). A worker in the middle of initialization does not react to `SIGTERM` until it finishes (in `main.cpp` the signal only sets a flag), so `SIGKILL` cuts it off mid-work.

### 5.4 Server or worker crashes in general

Any crash or `kill -9` (including when the parent dies, since the worker gets `PR_SET_PDEATHSIG` / `SIGTERM`) leaves the device without cleanup. This is the general case from which the previous points follow.

### 5.5 Possible contribution: several processes (hypothesis, not checked)

Several runs had been made on the machine before (an older Python server, various experiments). If some process stayed alive or held a chip, that can also leave state behind. This time no process held `/dev/tenstorrent/*` (checked with `lsof`, which can miss processes from other containers).

## 6. Possible solutions

Ordered from the most reliable to the cheapest.

| # | Solution | What it addresses | Cost |
|---|---|---|---|
| 1 | In `close_device` (or in the runner `cleanup`) call `ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)` after `ttnn.close_mesh_device` | point 5.1 | small Python change; must check it does not introduce new instability (see the comment in tt-metal's conftest) |
| 2 | Capture a backtrace of the exit segfault (core dump or `gdb`) and fix it | point 5.2 | needs one deliberate shutdown with core dumps enabled; the fix depends on the cause |
| 3 | Let a worker in the middle of initialization be interrupted cleanly, or lengthen the wait before `SIGKILL` | point 5.3 | changes in `main.cpp` and `single_process_worker.cpp`; hard because of long tt-metal calls |
| 4 | Automatic `tt-smi -r` before the start (script, systemd `ExecStartPre`) | everything above, as a workaround | about a minute extra at start; hides the real cause; the reset stops any other process using the chips |
| 5 | On a `run_mailbox` error during warmup, reset automatically and retry once | everything above | complex in the supervisor; a reset from the application needs privileges |
| 6 | Ask the tt-metal team whether opening the device should recognize `0x40` and reset the core itself | cause on the tt-metal side | depends on them |

Recommendation: try solution 1 first (cheapest, and it has support in tt-metal's conftest), collect data for solution 2 in parallel, and until then use solution 4 for manual starts.

## 7. How to verify the hypotheses

1. **Fabric not disabled (5.1):** add `ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)` after `close_mesh_device`, then repeat the cycle start, test, clean shutdown, start without a reset. If the `run_mailbox` error no longer appears, the hypothesis is confirmed.
2. **Segfault (5.2):** run the server with `ulimit -c unlimited` or under `gdb`, shut it down with SIGTERM and read the worker backtrace. Also check whether it happens on every shutdown, not just once.
3. **Interruption during initialization (5.3):** deliberately interrupt the server in the middle of loading and confirm that the next start fails with the `run_mailbox` error (already seen at 00:40, but not repeated on purpose).
4. **Ethernet cores (hypothesis from section 2):** check in tt-metal whether translated coordinates `x-16` correspond to ethernet cores on Wormhole.

## 8. Open questions

- Whether the exit segfault happens on every shutdown. Two of two shutdowns in this session had it (1:21 and 14:37), but that is a small sample and no backtrace was captured.
- Whether a device that was closed cleanly and without the segfault would leave the chips clean (that is, whether the cause is 5.1 or 5.2).
- Whether the same happens on other machines (n150, n300), or whether it is tied to the T3K and the fabric.
