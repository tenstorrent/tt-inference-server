// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "runtime/runners/blaze_runner/scheduler_interface.hpp"

namespace tt::runners::blaze {

namespace sch = tt_llm_engine::scheduler;

struct MockPrefillSchedulerConfig {
  std::chrono::milliseconds prefillLatency{0};
  uint32_t prefillChunkSize = 128;
};

struct MockDecodeSchedulerConfig {
  // Depth of the transformer pipeline the mock stands in for. Time-to-first-
  // token is a pipeline-fill cost, not a per-chunk prefill cost: the prompt
  // streams through the stages one chunk per stage-time and the first token
  // pops out once the final chunk clears the last stage.
  uint32_t numPipelineStages = 1;
  // Time a single token spends in one pipeline stage. Prefill chunks can stream
  // through back-to-back (the whole prompt is known up front), so this sets the
  // pipeline-*fill* latency: one stage-time per chunk plus the pipeline depth.
  std::chrono::microseconds stageLatency{0};
  // Prefill compute per chunk, added to time-to-first-token on top of the
  // fill above. stageLatency is a single token's per-stage cost, but a prefill
  // chunk pushes thousands of tokens through each stage,
  // which for a real model dwarfs one token's traversal.
  // This is the term that makes TTFT realistic (and scale with prompt length,
  // since it is charged per chunk); it mirrors mock_pipeline's
  // PrefillMockConfig chunk_latency. 0 keeps TTFT at the bare pipeline-fill
  // (~stageLatency-scale).
  std::chrono::milliseconds prefillChunkLatency{0};
  uint32_t prefillChunkSize = 128;
  uint32_t decodeTokenId = 0;
  // Per-slot spacing between decode tokens. Decode is autoregressive - token
  // N+1 cannot enter the pipeline until token N has traversed all stages - so a
  // single sequence produces one token per full traversal, i.e.
  // numPipelineStages * stageLatency (e.g. 64 * 44us = 2816us). The familiar
  // "one token per stage-time" figure is the aggregate rate once many slots
  // interleave in the pipeline (numPipelineStages slots busy => one token per
  // stageLatency across all of them), not a per-slot rate. The config builder
  // derives this from numPipelineStages * stageLatency; keeping it as its own
  // field lets tests dial the decode spacing independently of the fill.
  std::chrono::microseconds decodeTokenLatency{0};
  // Max absolute wobble added to each decode interval, drawn uniformly from
  // [-decodeTokenJitter, +decodeTokenJitter]. Mean-preserving, so throughput
  // and mean TPOT are unchanged; its only job is to stop every slot from
  // ticking in lockstep. Without it, identical-length requests started together
  // stay phase-locked for the whole run, finish in synchronized clumps, and
  // spike the TTFT tail as a burst of requests all contend for slots at once.
  // The jitter accumulates over a generation (a random walk), so completions
  // drift apart and slots free in a trickle instead of a clump. 0 = perfect
  // metronome (the phase-locked behaviour). It stands in for the desync a real
  // system gets for free (variable output lengths, batch effects, scheduling),
  // which this fixed-length synthetic benchmark lacks.
  std::chrono::microseconds decodeTokenJitter{0};
};

namespace detail {

// Shared slot pool + response/output queues for mock schedulers.
class MockSchedulerCore {
 public:
  explicit MockSchedulerCore(uint32_t maxUsers) : slotInUse(maxUsers, false) {
    freeSlots.reserve(maxUsers);
    for (uint32_t i = 0; i < maxUsers; ++i) {
      freeSlots.push_back(i);
    }
  }

  void start() { running = true; }

  void stop() {
    running = false;
    responses.clear();
    outputs.clear();
    freeSlots.clear();
    for (uint32_t i = 0; i < slotInUse.size(); ++i) {
      slotInUse[i] = false;
      freeSlots.push_back(i);
    }
  }

  bool isRunning() const { return running; }

  bool handleAllocate(const sch::ISRequest& request) {
    sch::SchedulerResponse response{};
    response.request_id = request.request_id;
    response.request_type = sch::RequestType::ALLOCATE;
    if (freeSlots.empty()) {
      response.slot_id = sch::INVALID_SLOT;
      response.error_code = sch::request_error::kNoFreeSlot;
    } else {
      const uint32_t slotId = freeSlots.back();
      freeSlots.pop_back();
      slotInUse[slotId] = true;
      response.slot_id = slotId;
      response.error_code = sch::request_error::kOk;
    }
    responses.push_back(response);
    return true;
  }

  bool handleEvictOrStop(const sch::ISRequest& request) {
    freeSlot(request.slot_id);
    responses.push_back(makeAck(request));
    return true;
  }

  void pushResponse(sch::SchedulerResponse response) {
    responses.push_back(response);
  }

  void pushOutput(sch::OutputMessage output) { outputs.push_back(output); }

  bool tryPopResponse(sch::SchedulerResponse& response) {
    if (responses.empty()) {
      return false;
    }
    response = responses.front();
    responses.pop_front();
    return true;
  }

  bool tryPopOutput(sch::OutputMessage& output) {
    if (outputs.empty()) {
      return false;
    }
    output = outputs.front();
    outputs.pop_front();
    return true;
  }

  // Drop every output queued for `slotId`. Used by EVICT/STOP so a token the
  // emitter produced just before cancellation can't outlive the slot - the
  // runner drains scheduler responses before outputs, so the EVICT/STOP ack
  // would otherwise flip the slot to IDLE/FREE first and the stale output
  // would then either trip "unexpected token for IDLE slot" or, worse, be
  // misattributed to whoever re-allocates the slot.
  void purgeOutputsForSlot(uint32_t slotId) {
    outputs.erase(std::remove_if(outputs.begin(), outputs.end(),
                                 [&](const sch::OutputMessage& o) {
                                   return o.slot_id == slotId;
                                 }),
                  outputs.end());
  }

 private:
  static sch::SchedulerResponse makeAck(const sch::ISRequest& request) {
    return sch::SchedulerResponse{
        .request_id = request.request_id,
        .slot_id = request.slot_id,
        .error_code = sch::request_error::kOk,
        .request_type = request.type,
    };
  }

  void freeSlot(uint32_t slotId) {
    if (slotId < slotInUse.size() && slotInUse[slotId]) {
      slotInUse[slotId] = false;
      freeSlots.push_back(slotId);
    }
  }

  bool running = false;
  std::vector<bool> slotInUse;
  std::vector<uint32_t> freeSlots;
  std::deque<sch::SchedulerResponse> responses;
  std::deque<sch::OutputMessage> outputs;
};

}  // namespace detail

class MockPrefillScheduler final : public IPrefillScheduler {
 public:
  explicit MockPrefillScheduler(uint32_t maxUsers,
                                MockPrefillSchedulerConfig cfg = {})
      : core(maxUsers), cfg(std::move(cfg)) {}

  void start() override { core.start(); }
  void stop() override { core.stop(); }

  bool push_request(const sch::ISRequest& request) override {
    if (!core.isRunning()) {
      return false;
    }

    switch (request.type) {
      case sch::RequestType::ALLOCATE:
        return core.handleAllocate(request);
      case sch::RequestType::EVICT:
      case sch::RequestType::STOP:
        return core.handleEvictOrStop(request);
      case sch::RequestType::SUBMIT: {
        if (cfg.prefillLatency.count() > 0) {
          std::this_thread::sleep_for(
              cfg.prefillLatency *
              (request.tokens.size() / cfg.prefillChunkSize + 1));
        }
        sch::OutputMessage output{};
        output.slot_id = request.slot_id;
        output.prefill_complete = true;
        output.real_pos = static_cast<uint32_t>(request.tokens.size());
        output.request_id = request.request_id;
        core.pushOutput(output);
        return true;
      }
      case sch::RequestType::CONTINUE:
        core.pushResponse(sch::SchedulerResponse{
            .request_id = request.request_id,
            .slot_id = request.slot_id,
            .error_code = sch::request_error::kMalformedTokenStream,
            .request_type = sch::RequestType::CONTINUE,
        });
        return true;
    }
    return true;
  }

  bool try_pop_response(sch::SchedulerResponse& response) override {
    return core.tryPopResponse(response);
  }

  bool try_pop_output(sch::OutputMessage& output) override {
    return core.tryPopOutput(output);
  }

 private:
  detail::MockSchedulerCore core;
  MockPrefillSchedulerConfig cfg;
};

// Asynchronous mock decode scheduler.
//
// push_request() returns immediately for SUBMIT/CONTINUE after enqueuing a
// PendingJob; a single background emitter thread paces token emission per slot
// at decodeTokenLatency intervals (first token deferred by the pipeline-fill
// latency computed in enqueueJobLocked).
// This keeps the runner thread free to service cancels, memory requests, and
// other slots' outputs while a long generation is in flight - which is the
// point of having a mock scheduler for bottleneck testing in the first place
// (a synchronous emitter would just serialize everything on the runner).
//
// EVICT/STOP synchronously drop any pending emission for the affected slot,
// mirroring real hardware: once a slot is yanked, no further tokens come out.
class MockDecodeScheduler final : public IDecodeScheduler {
 public:
  explicit MockDecodeScheduler(uint32_t maxUsers,
                               MockDecodeSchedulerConfig cfg = {})
      : core(maxUsers), cfg(std::move(cfg)) {}

  ~MockDecodeScheduler() override { stop(); }

  MockDecodeScheduler(const MockDecodeScheduler&) = delete;
  MockDecodeScheduler& operator=(const MockDecodeScheduler&) = delete;
  MockDecodeScheduler(MockDecodeScheduler&&) = delete;
  MockDecodeScheduler& operator=(MockDecodeScheduler&&) = delete;

  void start() override {
    std::lock_guard<std::mutex> lock(mutex);
    core.start();
    if (!emitter.joinable()) {
      stopRequested = false;
      emitter = std::thread([this] { emitterLoop(); });
    }
  }

  void stop() override {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!emitter.joinable() && !core.isRunning()) {
        return;
      }
      stopRequested = true;
      pending.clear();
      core.stop();
    }
    cv.notify_all();
    if (emitter.joinable()) {
      emitter.join();
    }
  }

  bool push_request(const sch::ISRequest& request) override {
    bool wakeEmitter = false;
    bool result = true;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!core.isRunning()) {
        return false;
      }
      switch (request.type) {
        case sch::RequestType::ALLOCATE:
          result = core.handleAllocate(request);
          break;
        case sch::RequestType::EVICT:
        case sch::RequestType::STOP:
          cancelSlotLocked(request.slot_id);
          wakeEmitter = true;
          result = core.handleEvictOrStop(request);
          break;
        case sch::RequestType::SUBMIT:
        case sch::RequestType::CONTINUE:
          if (request.gen.max_new_tokens == 0) {
            // max_new_tokens == 0 is a no-op decode (e.g. a stray submit, or a
            // prefill-only path that routes through here). Don't enqueue: the
            // emitter's isLast = (emitted + 1 == maxTokens) check would never
            // fire for maxTokens == 0 and the slot would stream phantom tokens
            // until uint32_t wraparound. Emit a single terminal output instead
            // so the runner finalizes the slot cleanly.
            sch::OutputMessage terminal{};
            terminal.slot_id = request.slot_id;
            terminal.is_complete = true;
            terminal.tokens_generated = 0;
            terminal.position_id = request.position_id.value_or(
                static_cast<uint32_t>(request.tokens.size()));
            terminal.request_id = request.request_id;
            core.pushOutput(terminal);
          } else {
            enqueueJobLocked(request);
            wakeEmitter = true;
          }
          break;
        default:
          break;
      }
    }
    if (wakeEmitter) {
      cv.notify_all();
    }
    return result;
  }

  bool try_pop_response(sch::SchedulerResponse& response) override {
    std::lock_guard<std::mutex> lock(mutex);
    return core.tryPopResponse(response);
  }

  bool try_pop_output(sch::OutputMessage& output) override {
    std::lock_guard<std::mutex> lock(mutex);
    return core.tryPopOutput(output);
  }

  uint32_t get_spec_accepts(uint32_t /*slotId*/) const override { return 0; }
  uint32_t get_spec_rejects(uint32_t /*slotId*/) const override { return 0; }

 private:
  struct PendingJob {
    uint32_t slotId = sch::INVALID_SLOT;
    uint32_t requestId = 0;
    uint32_t maxTokens = 0;
    uint32_t emitted = 0;
    uint32_t basePosition = 0;
    std::chrono::steady_clock::time_point nextAt{};
  };

  void enqueueJobLocked(const sch::ISRequest& request) {
    PendingJob job;
    job.slotId = request.slot_id;
    job.requestId = request.request_id;
    job.maxTokens = request.gen.max_new_tokens;
    job.basePosition = request.position_id.value_or(
        static_cast<uint32_t>(request.tokens.size()));
    const uint32_t numChunks =
        static_cast<uint32_t>(request.tokens.size()) / cfg.prefillChunkSize + 1;
    const uint32_t fillStages = numChunks + cfg.numPipelineStages - 1;
    job.nextAt = std::chrono::steady_clock::now() +
                 cfg.stageLatency * fillStages +
                 numChunks * cfg.prefillChunkLatency;
    pending.push_back(job);
  }

  void cancelSlotLocked(uint32_t slotId) {
    pending.erase(
        std::remove_if(pending.begin(), pending.end(),
                       [&](const PendingJob& j) { return j.slotId == slotId; }),
        pending.end());
    // Drop any tokens the emitter already published for this slot
    core.purgeOutputsForSlot(slotId);
  }

  std::chrono::microseconds sampleJitterLocked() {
    const int64_t j = cfg.decodeTokenJitter.count();
    if (j <= 0) {
      return std::chrono::microseconds{0};
    }
    std::uniform_int_distribution<int64_t> dist(-j, j);
    return std::chrono::microseconds{dist(jitterRng)};
  }

  void emitterLoop() {
    std::unique_lock<std::mutex> lock(mutex);
    while (!stopRequested) {
      if (pending.empty()) {
        cv.wait(lock);
        continue;
      }

      // Snapshot "now" once per tick so a just-rescheduled job (especially
      // when decodeTokenLatency == 0) doesn't fire again in this iteration.
      const auto now = std::chrono::steady_clock::now();
      auto nextDeadline = std::chrono::steady_clock::time_point::max();

      for (auto it = pending.begin(); it != pending.end();) {
        if (it->nextAt <= now) {
          const bool isLast = (it->emitted + 1 == it->maxTokens);
          core.pushOutput(sch::OutputMessage{
              .slot_id = it->slotId,
              .token_id = cfg.decodeTokenId,
              .is_complete = isLast,
              .tokens_generated = it->emitted + 1,
              .position_id = it->basePosition + it->emitted,
              .request_id = it->requestId,
          });
          ++it->emitted;
          if (isLast) {
            it = pending.erase(it);
            continue;
          }
          // Add the mean-zero jitter so slots don't stay phase-locked. Clamp at
          // zero so an over-large jitter can't schedule a token in the past.
          auto interval = cfg.decodeTokenLatency + sampleJitterLocked();
          if (interval < std::chrono::microseconds{0}) {
            interval = std::chrono::microseconds{0};
          }
          it->nextAt = now + interval;
        }
        if (it->nextAt < nextDeadline) {
          nextDeadline = it->nextAt;
        }
        ++it;
      }

      // cv.wait_until atomically releases the lock, which lets consumers
      // (the runner thread polling try_pop_output / try_pop_response, or
      // push_request handing in a new SUBMIT) grab it. Even when the deadline
      // is in the past (latency-0 stress mode), the unlock-relock round trip
      // is a deliberate yield point that keeps the consumer from starving.
      if (pending.empty()) {
        cv.wait(lock);
      } else {
        cv.wait_until(lock, nextDeadline);
      }
    }
  }

  detail::MockSchedulerCore core;
  MockDecodeSchedulerConfig cfg;

  std::mutex mutex;
  std::condition_variable cv;
  std::vector<PendingJob> pending;
  bool stopRequested = false;
  std::thread emitter;
  std::mt19937_64 jitterRng{0xC0FFEEULL};
};

}  // namespace tt::runners::blaze
