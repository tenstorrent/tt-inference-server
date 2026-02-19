Readme

## Tracy profiling (C++ server)

We use [Tenstorrent’s Tracy](https://github.com/tenstorrent/tracy) with the Tracy GUI built from the same repo. The GUI version we use is **0.10.0**.

### What Tracy enables

- **Zones** – Instrumented regions (e.g. `API::completions`, `Scheduler::start`, `Worker::process_task`) to see where time is spent.
- **Plots** – Numeric series over time (e.g. `pending_tasks`).
- **Memory profiling** – Allocation/free tracking (when built with Tracy alloc hooks).
- **Multi-process** – Main process on port 8086, workers on 8087, 8088, … (each process can be connected separately in the GUI).

### Building the Tracy GUI

Build the **profiler** (GUI) from the [Tenstorrent Tracy repo](https://github.com/tenstorrent/tracy):

```bash
git clone https://github.com/tenstorrent/tracy.git
cd tracy/profiler/build/unix
make release
```

The GUI binary is `Tracy-release` (or similar) in that build directory. GUI version should match the client; we use **0.10.0**.

### Using the GUI when the server runs on a remote machine

1. Install the Tracy GUI on your **local** machine (build from repo above).
2. On the **remote** machine, the C++ server is built with Tracy and listens on:
   - **8086** – main process
   - **8087, 8088, …** – worker processes
3. Forward those ports from remote to local, e.g.:
   ```bash
   ssh -L 8086:localhost:8086 -L 8087:localhost:8087 user@remote-host
   ```
4. On your **local** machine, open the Tracy GUI and connect to **localhost:8086** (and optionally **localhost:8087**, etc.). Traffic goes over SSH, so the GUI talks to the Tracy client on the remote server.

### Launch configurations (VS Code)

| Use case | Launch config | Notes |
|----------|----------------|-------|
| **Run C++ server with Tracy** | **C++ Server [CodeLLDB + Tracy]** | Builds `build-tracy/`, runs the server with Tracy enabled; connect GUI to localhost:8086 (and 8087 for workers). |
| **Run C++ server without Tracy** | **C++ Server [CodeLLDB]** | Builds `build/`, runs the server with no Tracy instrumentation. |
| **Capture Tracy to a file** | **Tracy: Capture to file** | Runs the Tracy **capture** tool: connects to port 8086 and writes a capture to `cpp_server/capture.tracy` for 60 seconds (`-s 60`). Use **-f** in the config to overwrite an existing file. Start the **C++ server with Tracy** first, then run this config; afterward open `capture.tracy` in the Tracy GUI. |

Typical workflow for “capture to file”:

1. Start **C++ Server [CodeLLDB + Tracy]** (server listening on 8086).
2. Run **Tracy: Capture to file** (capture connects to 8086 and writes `capture.tracy`).
3. Open `cpp_server/capture.tracy` in the Tracy GUI to inspect the capture offline.
