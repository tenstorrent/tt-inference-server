// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
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
  // The transformer pipeline this mock stands in for: `numPipelineStages`
  // stages of `stageLatency` each. It is a single *shared* resource - it
  // accepts one token every stageLatency, and each token exits after traversing
  // all stages (numPipelineStages * stageLatency). BOTH prefill and decode
  // tokens flow through this one pipeline, so total token throughput (prefill +
  // decode) is capped at 1 / stageLatency (e.g. 44us -> ~22.7k tok/s) and the
  // two contend for that single budget - exactly like the hardware pipeline.
  uint32_t numPipelineStages = 1;
  std::chrono::microseconds stageLatency{0};
  // A slot injects up to this many prompt tokens before the scheduler rotates
  // to the next prefilling slot
  uint32_t prefillChunkSize = 24;
  uint32_t decodeTokenId = 0;
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

// Asynchronous mock decode scheduler backed by a single shared pipeline.
//
// A single background emitter thread runs a tiny discrete-event simulation of
// the transformer pipeline: every stageLatency it consumes tokens that have
// exited, injects one new token, then sleeps to the next event. BOTH prefill
// and decode tokens are injected into this one pipeline, so:
//   * total throughput (prefill + decode) is capped at 1 / stageLatency, and
//     prefill and decode contend for that single budget - just like hardware;
//   * decode is injected first, prefill fills the spare capacity, so a decode-
//     saturated pipeline starves prefill;
//   * a sequence's consecutive decode tokens are one full traversal apart
//     (autoregressive: token N+1 needs token N's result), while N interleaved
//     slots produce ~one token per stageLatency in aggregate;
//   * time-to-first-token = time to inject the whole prompt (round-robin, in
//     chunks of prefillChunkSize, competing for the shared pipeline) + one
//     traversal, so it scales with prompt length and with load.
//
// push_request() returns immediately after enqueuing/cancelling. EVICT/STOP
// synchronously drop the slot's job, its in-flight tokens, and queued outputs.
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
      inflight.clear();
      decodeCursor = 0;
      prefillCursor = 0;
      nextInjectAt = std::chrono::steady_clock::now();
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
      inflight.clear();
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
          cancelSlot(request.slot_id);
          // pending or in flight work may have been removed, so the
          // emitter should wake up an recompute.
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
            enqueueJob(request);
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
    uint32_t basePosition = 0;
    uint32_t prefillTokensToBeInjected = 0;
    uint32_t prefillInjected = 0;
    uint32_t chunkRemaining = 0;
    uint32_t outputsEmitted = 0;
    bool decodeReady = false;

    struct PrefillInjectionState {
      bool producesOutput = false;
      bool shouldRotateCursor = false;
    };

    bool hasPrefillToInject() const {
      return prefillInjected < prefillTokensToBeInjected;
    }

    bool hasDecodeToInject() const {
      return decodeReady && outputsEmitted < maxTokens;
    }

    void injectDecodeToken() {
      decodeReady = false;  // token in flight; can't inject next until exit
    }

    PrefillInjectionState injectPrefillToken(uint32_t prefillChunkSize) {
      ++prefillInjected;
      const bool isLastPrefillToken = !hasPrefillToInject();
      if (chunkRemaining > 0) --chunkRemaining;

      PrefillInjectionState injection{
          .producesOutput = isLastPrefillToken,
          .shouldRotateCursor = (chunkRemaining == 0 || isLastPrefillToken),
      };
      if (injection.shouldRotateCursor) {
        chunkRemaining = prefillChunkSize;
      }
      return injection;
    }
  };

  struct InFlight {
    // One token traversing the pipeline; exits (produces its result) at exitAt.
    std::chrono::steady_clock::time_point exitAt{};
    uint32_t slotId = sch::INVALID_SLOT;
    bool producesOutput = false;  // last prefill token can produce an output,
                                  // or a decode token, of course
  };

  std::chrono::microseconds totalLatency() const {
    return cfg.stageLatency *
           cfg.numPipelineStages;  // whole-pipeline traversal
  }

  // Backpressure bound: never more than a full pipeline of tokens in flight.
  uint32_t maxInFlight() const {
    return cfg.numPipelineStages == 0 ? 1u : cfg.numPipelineStages;
  }

  void enqueueJob(const sch::ISRequest& request) {
    PendingJob job;
    job.slotId = request.slot_id;
    job.requestId = request.request_id;
    job.maxTokens = request.gen.max_new_tokens;
    job.basePosition = request.position_id.value_or(
        static_cast<uint32_t>(request.tokens.size()));
    job.prefillTokensToBeInjected =
        static_cast<uint32_t>(request.tokens.size());
    job.chunkRemaining = cfg.prefillChunkSize;
    // With no prompt tokens there is no prefill to produce the first output, so
    // the slot may immediately inject its first decode token (cold decode).
    job.decodeReady = (job.prefillTokensToBeInjected == 0);
    pending.push_back(job);
  }

  void cancelSlot(uint32_t slotId) {
    pending.erase(
        std::remove_if(pending.begin(), pending.end(),
                       [&](const PendingJob& j) { return j.slotId == slotId; }),
        pending.end());
    inflight.erase(
        std::remove_if(inflight.begin(), inflight.end(),
                       [&](const InFlight& f) { return f.slotId == slotId; }),
        inflight.end());
    core.purgeOutputsForSlot(slotId);
  }

  PendingJob* findJobBySlot(uint32_t slotId) {
    for (auto& j : pending) {
      if (j.slotId == slotId) return &j;
    }
    return nullptr;
  }

  bool hasInjectableWork() const {
    for (const auto& j : pending) {
      if (j.hasPrefillToInject()) return true;
      if (j.hasDecodeToInject()) return true;
    }
    return false;
  }

  // Consume every token that has exited the pipeline by `now`, emitting an
  // output for each token that produces one
  void consumeExitedTokens(std::chrono::steady_clock::time_point now) {
    while (!inflight.empty() && inflight.front().exitAt <= now) {
      const InFlight e = inflight.front();
      inflight.pop_front();
      if (!e.producesOutput) continue;  // non-last prefill token: drained
      PendingJob* job = findJobBySlot(
          e.slotId);  // use raw pointer to mutate job later, safe since we did
                      // not allocate nothing with new
      if (!job) continue;  // slot was evicted while this token was in flight
      ++job->outputsEmitted;
      const bool isLast = (job->outputsEmitted == job->maxTokens);
      core.pushOutput(sch::OutputMessage{
          .slot_id = job->slotId,
          .token_id = cfg.decodeTokenId,
          .is_complete = isLast,
          .tokens_generated = job->outputsEmitted,
          .position_id = job->basePosition + job->outputsEmitted - 1,
          .request_id = job->requestId,
      });
      if (isLast) {
        const uint32_t slotId = job->slotId;
        pending.erase(std::remove_if(pending.begin(), pending.end(),
                                     [&](const PendingJob& j) {
                                       return j.slotId == slotId;
                                     }),
                      pending.end());
      } else {
        job->decodeReady = true;  // its next decode token may now be injected
      }
    }
  }

  // Choose the next token to inject: a decode-ready slot first (round-robin),
  // then prefill. Prefill round-robin injects up to prefillChunkSize tokens per
  // slot before rotating. prefillCursor is the ring start index
  PendingJob* selectNextInjection(bool& producesOutput) {
    const size_t n = pending.size();
    for (size_t i = 0; i < n; ++i) {
      const size_t idx = (decodeCursor + i) % n;
      PendingJob& j = pending[idx];
      if (j.hasDecodeToInject()) {
        j.injectDecodeToken();
        decodeCursor = (idx + 1) % n;
        producesOutput = true;
        return &j;
      }
    }
    for (size_t i = 0; i < n; ++i) {
      const size_t idx = (prefillCursor + i) % n;
      PendingJob& j = pending[idx];
      if (j.hasPrefillToInject()) {
        const PendingJob::PrefillInjectionState injection =
            j.injectPrefillToken(cfg.prefillChunkSize);
        producesOutput = injection.producesOutput;
        prefillCursor = injection.shouldRotateCursor ? (idx + 1) % n : idx;
        return &j;
      }
    }
    return nullptr;
  }

  // Inject every token whose scheduled slot has arrived, into the shared
  // pipeline.
  void injectDueTokens(std::chrono::steady_clock::time_point now) {
    if (nextInjectAt < now &&
        (inflight.empty() || nextInjectAt + totalLatency() <= now)) {
      // nextInjectAt is either long overdue so we avoid compensating for time
      // difference passed, or inflight is empty so no need to inject anything
      // If we did not set it, we would inject many tokens from old nextInjectAt
      // time until now, which could be millions of tokens.
      // This resets this value and we could have a clean start.
      nextInjectAt = now;
    }
    while (inflight.size() < maxInFlight() && nextInjectAt <= now) {
      if (pending.empty()) {
        // no pending tokens, so whatever comes next could be immediately
        // injected starting from now
        nextInjectAt = now;
        break;
      }
      bool producesOutput = false;
      PendingJob* chosen = selectNextInjection(producesOutput);
      if (chosen == nullptr) {
        nextInjectAt = now;
        break;
      }
      inflight.push_back(InFlight{
          .exitAt = nextInjectAt + totalLatency(),
          .slotId = chosen->slotId,
          .producesOutput = producesOutput,
      });
      nextInjectAt += cfg.stageLatency;
    }
  }

  void emitterLoop() {
    std::unique_lock<std::mutex> lock(mutex);
    while (!stopRequested) {
      if (pending.empty() && inflight.empty()) {
        cv.wait(lock);
        continue;
      }

      const auto now = std::chrono::steady_clock::now();
      consumeExitedTokens(now);
      injectDueTokens(now);

      // Sleep to the next event: the earliest exit, or (if injectable work is
      // waiting on the rate limiter and the pipeline has room) the next
      // allowed inject time. cv.wait_until releases the lock so consumers and
      // push_request can make progress.
      auto wake = std::chrono::steady_clock::time_point::max();
      if (!inflight.empty()) {
        wake = std::min(wake, inflight.front().exitAt);
      }
      if (hasInjectableWork() && inflight.size() < maxInFlight()) {
        wake = std::min(wake, std::max(now, nextInjectAt));
      }
      if (wake == std::chrono::steady_clock::time_point::max()) {
        cv.wait(lock);
      } else {
        cv.wait_until(lock, wake);
      }
    }
  }

  detail::MockSchedulerCore core;
  MockDecodeSchedulerConfig cfg;

  std::mutex mutex;
  std::condition_variable cv;
  std::vector<PendingJob> pending;
  std::deque<InFlight> inflight;
  std::chrono::steady_clock::time_point nextInjectAt{};
  // decode and prefill cursors track the next token to inject for each type:
  // prefill and decode. These make sure that we continue where we left off in
  // round robin fashion and to provide some fairness in pending jobs.
  size_t decodeCursor = 0;
  size_t prefillCursor = 0;
  bool stopRequested = false;
  std::thread emitter;
};

}  // namespace tt::runners::blaze
