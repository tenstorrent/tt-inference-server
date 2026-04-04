# Async Slot Allocation: Sync-to-Async Transition

## Background

The TT Media Server uses a two-process architecture: a **main process** (Drogon HTTP server) and a **worker process** (pipeline runner). They communicate via IPC shared-memory queues. The worker manages 64 KV cache slots on a hardware device. Slots are not freed after each request — they persist for multi-turn conversations, evicted only via LRU when a threshold is reached.

Under heavy load (1K+ concurrent requests), the server would **deadlock**. With the mock pipeline (instant token generation) it worked; with the socket pipeline (real hardware latency), it froze.

## The Original Synchronous Design

### Flow

1. HTTP request arrives on a Drogon IO thread
2. `resolveSession()` called synchronously:
   - Existing session: look up slot ID, return
   - New session: `createSession()` → `requestSlotIdFromMemoryManager()`
3. `requestSlotIdFromMemoryManager()`:
   - Creates a `std::promise<uint32_t>`
   - Pushes an ALLOCATE task to the IPC request queue via **blocking `push()`**
   - Returns a `std::future` — the IO thread calls `future.wait_for()` in a retry loop
4. A background drain thread reads results from the IPC result queue and fulfills the promise
5. IO thread unblocks, continues with the allocated slot

### Why It Deadlocked

Drogon uses an event-loop-per-thread model. Each IO thread runs a single event loop. Blocking an IO thread means:
- No new HTTP requests can be accepted on that thread
- No `queueInLoop` callbacks can be processed on that thread
- No SSE streaming data can be sent on that thread

With 1024 concurrent requests and a 64-capacity IPC queue, IO threads would block on `push()` waiting for queue space. The drain thread couldn't make progress fast enough because it was single-threaded. Result: cascading blocks across all IO threads → server freeze.

## The Async Design

### Core Idea

Never block a Drogon IO thread. Instead:

1. IO thread submits an async allocation request and **returns immediately**
2. The drain thread picks up the result and posts a callback back to the original IO thread's event loop
3. The IO thread processes the callback when it's ready, continuing with the request

### New Components

#### `PendingAsyncSession` (session_manager.hpp)

Bundles all state needed to complete an async allocation:

```cpp
struct PendingAsyncSession {
    domain::Session session;
    std::string sessionId;
    std::function<void(domain::Session)> onSuccess;
    std::function<void(const std::string&)> onError;
    trantor::EventLoop* callerLoop;  // IO thread to post callback to
    int attemptsRemaining;
    bool inFlight;
};
```

#### `createSessionAsync()` (session_manager.cpp)

Replaces the synchronous `createSession()` path for the hot (chat completions) flow:

- Creates a `PendingAsyncSession` with success/error callbacks
- Calls `sendAsyncAllocRequest()` to push to the IPC queue
- Returns immediately — the IO thread is free

#### `sendAsyncAllocRequest()` (session_manager.cpp)

Sends the IPC allocation request non-blockingly:

- Generates a unique task ID
- **Inserts into `pendingAsyncAllocations` map BEFORE pushing** (prevents race with drain thread)
- Calls `tryPush()` (non-blocking) instead of `push()` (blocking)
- If queue is full: removes the pending entry, schedules a retry (50ms delay) or invokes `onError`
- If push succeeds: entry is already in the map, drain thread will find it

#### `resolveSessionAsync()` (llm_controller.cpp)

Replaces the synchronous `resolveSession()`:

- **Existing session**: `acquireSessionSlot()` atomically looks up slot + sets in-flight. Calls `onResolved` synchronously.
- **New session**: calls `createSessionAsync()` with wrapped `onResolved`/`onError` callbacks. IO thread returns immediately.

#### `acquireSessionSlot()` (session_manager.cpp)

Atomic slot acquisition for existing sessions — prevents a TOCTOU race where a session could be evicted between lookup and marking as in-flight:

```cpp
uint32_t SessionManager::acquireSessionSlot(const std::string& sessionId) {
    uint32_t result = INVALID_SLOT_ID;
    sessions.modify(sessionId, [&result](domain::Session& s) {
        s.updateActivityTime();
        s.setInFlight(true);
        result = s.getSlotId();
    });
    return result;
}
```

### Controller Changes

Both `chatCompletions` (non-streaming) and `handleStreaming` were refactored to use `resolveSessionAsync`. The entire post-resolution logic (LLM submission, response building, error handling) moves into the `onResolved` callback:

```cpp
resolveSessionAsync(
    request,
    [this, request, cb](SessionInfo info) {
        // All request processing happens here, on the IO thread,
        // but only AFTER the slot is allocated asynchronously
        auto completion = service->submitRequest(std::move(*request));
        // ... build and send response ...
        (*cb)(resp);
    },
    [cb](const std::string& error) {
        // Allocation failed after all retries
        auto resp = /* 503 Service Unavailable */;
        (*cb)(resp);
    });
```

### Drain Thread Changes

The drain thread (`drainResultQueue`) was modified to:

1. **Batch-drain results**: reads ALL available results in a tight loop before sleeping, instead of one-per-iteration. This prevents the memory result queue from backing up and blocking the worker.

2. **Handle async results**: when a result arrives, looks up `pendingAsyncAllocations` by task ID:
   - **Success**: sets slot on session, inserts into session map, posts `onSuccess` to the caller's event loop
   - **Failure with retries left**: adds to retry queue (500ms delay)
   - **Failure, retries exhausted**: posts `onError` to the caller's event loop

3. **Process retry queue**: periodically re-sends allocation requests for pending retries. Collects ready items under the lock, releases the lock, then sends — avoiding recursive mutex deadlock.

4. **Process dealloc queue**: retries any deferred deallocation requests that couldn't be pushed earlier.

## Bugs Fixed Along the Way

### 1. TOCTOU Race on Session In-Flight Status

**Problem**: A session could be created but not yet marked as in-flight. During the gap, LRU eviction could evict the session, destroying the user's KV cache.

**Fix**: `createSession()` now accepts `inFlight` parameter and sets it before insertion. `acquireSessionSlot()` atomically marks in-flight during lookup.

### 2. Recursive Mutex Deadlock

**Problem**: `processRetryQueue()` held `retryMutex` while calling `sendAsyncAllocRequest()`. When `tryPush` failed, `sendAsyncAllocRequest` tried to re-acquire `retryMutex` to add to the retry queue. `std::mutex` is not recursive → deadlock.

**Fix**: Collect ready items into a local vector under the lock, release the lock, then call `sendAsyncAllocRequest` on each.

### 3. Insert-After-Push Race

**Problem**: `sendAsyncAllocRequest` called `tryPush()` first, then `pendingAsyncAllocations.insert()`. If the drain thread processed the result between those two calls, it couldn't find the pending entry → callback lost → request hung forever.

**Fix**: Insert into `pendingAsyncAllocations` BEFORE pushing to the IPC queue. If push fails, remove the entry. The result can never arrive before the entry exists.

### 4. Blocking `push()` in `sendDeallocRequest`

**Problem**: `evictOldSessions()` is called from `createSessionAsync()` on a Drogon IO thread. It called `sendDeallocRequest()` which used blocking `push()`. If the IPC queue was full, the IO thread blocked.

**Fix**: Changed to `tryPush()`. If the queue is full, the dealloc is deferred to a `deallocQueue` processed by the drain thread.

### 5. Single-Result Drain Loop Bottleneck

**Problem**: The drain thread read one result per loop iteration:

```
processRetryQueue()     // may send N alloc requests
processDeallocQueue()
tryPop ONE result       // reads 1
sleep 1ms
```

Under heavy load, the memory result queue filled to capacity (512). The worker blocked on `resultQueue->push()`, which blocked the task queue, which blocked IO threads.

**Fix**: Batch-drain all available results before sleeping:

```cpp
while (memoryResultQueue->tryPop(result)) {
    handleMemoryResult(result);
}
```

## Configuration Changes

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `MEMORY_QUEUE_CAPACITY` | 64 | 512 | Match the order of magnitude of the task queue (1024) |
| `MAX_RETRIES` | N/A (sync) | 30 | Allow sufficient time for slot recycling under high contention |
| Queue-full retry delay | N/A | 50ms | Short delay to avoid busy-spinning on a full request queue |
| Alloc-failure retry delay | N/A | 500ms | Allow time for LRU eviction to free slots |

## Architecture Summary

```
                    Drogon IO Threads
                    ┌─────────────────┐
  HTTP request ──►  │ resolveSession   │
                    │   Async()        │
                    │                  │
                    │ onResolved() ◄───┼──── callback via queueInLoop
                    │   submit to      │
                    │   task queue     │
                    └────────┬────────┘
                             │ createSessionAsync()
                             ▼
                    ┌─────────────────┐
                    │ Memory Request   │  tryPush (non-blocking)
                    │ Queue (IPC shm)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Worker Process   │  step(): getMemoryRequest → handleResponse
                    │ (Pipeline Runner)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Memory Result    │  batch tryPop (non-blocking)
                    │ Queue (IPC shm)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Drain Thread     │  handleMemoryResult()
                    │ (SessionManager) │  processRetryQueue()
                    │                  │  processDeallocQueue()
                    └─────────────────┘
                             │
                             │ callerLoop->queueInLoop(onSuccess/onError)
                             ▼
                    Back to IO thread
```

**Key invariant**: No Drogon IO thread ever blocks on an IPC operation. All IPC interactions from IO threads use `tryPush()`, with retry logic handled by the drain thread.
