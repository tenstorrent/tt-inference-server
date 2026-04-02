## Tracy profiling (C++ server)

We use [Tenstorrent’s Tracy](https://github.com/tenstorrent/tracy) with the Tracy GUI built from the same repo. The GUI version we use is **0.10.0**.

### What Tracy enables

- **Zones** – Instrumented regions (e.g. `API::chat_completions`, `Scheduler::start`, `Worker::process_task`) to see where time is spent.
- **Plots** – Numeric series over time (e.g. `pending_tasks`).
- **Memory profiling** – Allocation/free tracking (when built with Tracy alloc hooks).
- **Lock profiling** – Mutex hold/wait times for `TracyLockable(std::mutex, ...)` (scheduler, embedding service/controller, model runner). Lock events appear on the thread timeline; in the GUI, **Options → Locks** lists locks and **Draw locks** shows them on the timeline. Uncheck **Only contended** to see locks used by a single thread (otherwise only contended locks are shown).
- **Multi-process** – Main process on port 8086 (Tracy started in `register_services()` via `TracyStartMainProcess()`). Workers are started by fork+exec and each starts Tracy on 8087, 8088, … (connect to each in the GUI).

### Building with Tracy

```bash
./build.sh --tracy
```

The binary is at `./build/tt_media_server_cpp` as usual.

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
You can use a Homebrew formula that builds Tenstorrent’s Tracy.
```bash
brew tap-new /path/to/your/tracy
cp /path/to/your/tracy.rb $(brew --repository)/Library/Taps/path/to/your/homebrew-tracy/Formula/tracy.rb
brew install /path/to/your/homebrew-tracy/tracy
```
Replace `/path/to/your/tracy.rb` with the path to your formula file (e.g. from your Tenstorrent Tracy clone). After install, run the GUI with `tracy` if it was linked.
2. On the **remote** machine, the C++ server is built with Tracy and listens on:
   - **8086** – main process
   - **8087, 8088, …** – worker processes
3. Forward those ports from remote to local, e.g.:
   ```bash
   ssh -p <PORT> -L 8086:localhost:8086 -L 8087:localhost:8087 $USER@<remote-host>
   ```
   **Finding `<PORT>` when the server runs in Docker:** run `docker container ls` and check the port mapping: the host port mapped to the container’s port 22 is the one to use as `<PORT>` in the `ssh -p` command.
4. On your **local** machine, open the Tracy GUI and connect to **localhost:8086** (and optionally **localhost:8087**, etc.). Traffic goes over SSH, so the GUI talks to the Tracy client on the remote server.

### Launch configurations (VS Code)

| Use case | Launch config | Notes |
|----------|----------------|-------|
| **Run C++ server with Tracy** | **C++ Server [CodeLLDB + Tracy]** | Builds `build-tracy/`, runs the server with Tracy enabled; connect GUI to localhost:8086 (and 8087 for workers). |
| **Run C++ server without Tracy** | **C++ Server [CodeLLDB]** | Builds `build/`, runs the server with no Tracy instrumentation. |

### Capturing a profile

Start the server with Tracy enabled first, then capture using either method — both write to `capture.tracy` for offline viewing in the Tracy GUI.

**Command line** (supports custom duration and port):

```bash
./tracy-capture.sh [SECONDS] [PORT]   # defaults: 60s, port 8086
./tracy-capture.sh 10 8087            # 10 seconds, worker 0
```

**VS Code**: run the **Tracy: Capture to file** launch config (captures port 8086 for 60s).

### Tips

- **Many thread rows** – The timeline can show many thread rows; those are the request-handling threads.
- **Capture scope** – The "Capture to file" config connects to the main process (8086) only. Worker data (8087, 8088, …) is not included unless you run separate captures for those ports.
- **Debugging** – When using **C++ Server [CodeLLDB + Tracy]**, you can set the launch config to use `"initCommands": ["settings set target.process.follow-fork-mode parent"]` so the debugger doesn’t follow worker processes after fork (avoids stopping in worker code when you intend to stay in the main process).
