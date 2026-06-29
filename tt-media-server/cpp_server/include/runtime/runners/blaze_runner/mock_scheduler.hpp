// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <thread>
#include <vector>

#include "config/settings.hpp"
#include "runtime/runners/blaze_runner/scheduler_interface.hpp"

namespace tt::runners::blaze {
namespace detail {

namespace sch = tt_llm_engine::scheduler;

// Shared slot pool + response/output queues for mock schedulers.
class MockSchedulerCore {
 public:
  explicit MockSchedulerCore(uint32_t maxUsers) : slotInUse_(maxUsers, false) {
    freeSlots_.reserve(maxUsers);
    for (uint32_t i = 0; i < maxUsers; ++i) {
      freeSlots_.push_back(i);
    }
  }

  void start() { running_ = true; }

  void stop() {
    running_ = false;
    responses_.clear();
    outputs_.clear();
    freeSlots_.clear();
    for (uint32_t i = 0; i < slotInUse_.size(); ++i) {
      slotInUse_[i] = false;
      freeSlots_.push_back(i);
    }
  }

  bool isRunning() const { return running_; }

  bool handleAllocate(const sch::ISRequest& request) {
    sch::SchedulerResponse response{};
    response.request_id = request.request_id;
    response.request_type = sch::RequestType::ALLOCATE;
    if (freeSlots_.empty()) {
      response.slot_id = sch::INVALID_SLOT;
      response.error_code = sch::request_error::kNoFreeSlot;
    } else {
      const uint32_t slotId = freeSlots_.back();
      freeSlots_.pop_back();
      slotInUse_[slotId] = true;
      response.slot_id = slotId;
      response.error_code = sch::request_error::kOk;
    }
    responses_.push_back(response);
    return true;
  }

  bool handleEvictOrStop(const sch::ISRequest& request) {
    freeSlot(request.slot_id);
    responses_.push_back(makeAck(request));
    return true;
  }

  void pushResponse(sch::SchedulerResponse response) {
    responses_.push_back(response);
  }

  void pushOutput(sch::OutputMessage output) { outputs_.push_back(output); }

  bool tryPopResponse(sch::SchedulerResponse& response) {
    if (responses_.empty()) {
      return false;
    }
    response = responses_.front();
    responses_.pop_front();
    return true;
  }

  bool tryPopOutput(sch::OutputMessage& output) {
    if (outputs_.empty()) {
      return false;
    }
    output = outputs_.front();
    outputs_.pop_front();
    return true;
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
    if (slotId < slotInUse_.size() && slotInUse_[slotId]) {
      slotInUse_[slotId] = false;
      freeSlots_.push_back(slotId);
    }
  }

  bool running_ = false;
  std::vector<bool> slotInUse_;
  std::vector<uint32_t> freeSlots_;
  std::deque<sch::SchedulerResponse> responses_;
  std::deque<sch::OutputMessage> outputs_;
};

}  // namespace detail

namespace sch = tt_llm_engine::scheduler;

class MockPrefillScheduler final : public IPrefillScheduler {
 public:
  explicit MockPrefillScheduler(uint32_t maxUsers)
      : core_(maxUsers),
        prefillLatency_(std::chrono::milliseconds(
            tt::config::mockPrefillLatencyMs())) {}

  void start() override { core_.start(); }
  void stop() override { core_.stop(); }

  bool push_request(const sch::ISRequest& request) override {
    if (!core_.isRunning()) {
      return false;
    }

    switch (request.type) {
      case sch::RequestType::ALLOCATE:
        return core_.handleAllocate(request);
      case sch::RequestType::EVICT:
      case sch::RequestType::STOP:
        return core_.handleEvictOrStop(request);
      case sch::RequestType::SUBMIT: {
        if (prefillLatency_.count() > 0) {
          std::this_thread::sleep_for(prefillLatency_);
        }
        sch::OutputMessage output{};
        output.slot_id = request.slot_id;
        output.prefill_complete = true;
        output.real_pos = static_cast<uint32_t>(request.tokens.size());
        output.request_id = request.request_id;
        core_.pushOutput(output);
        return true;
      }
      case sch::RequestType::CONTINUE:
        core_.pushResponse(sch::SchedulerResponse{
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
    return core_.tryPopResponse(response);
  }

  bool try_pop_output(sch::OutputMessage& output) override {
    return core_.tryPopOutput(output);
  }

 private:
  detail::MockSchedulerCore core_;
  std::chrono::milliseconds prefillLatency_;
};

class MockDecodeScheduler final : public IDecodeScheduler {
 public:
  explicit MockDecodeScheduler(uint32_t maxUsers)
      : core_(maxUsers),
        decodeTokenId_(tt::config::mockDecodeTokenId()),
        prefillLatency_(std::chrono::milliseconds(
            tt::config::mockPrefillLatencyMs())),
        decodeTokenLatency_(std::chrono::microseconds(
            tt::config::mockDecodeTokenLatencyUs())) {}

  void start() override { core_.start(); }
  void stop() override { core_.stop(); }

  bool push_request(const sch::ISRequest& request) override {
    if (!core_.isRunning()) {
      return false;
    }

    switch (request.type) {
      case sch::RequestType::ALLOCATE:
        return core_.handleAllocate(request);
      case sch::RequestType::EVICT:
      case sch::RequestType::STOP:
        return core_.handleEvictOrStop(request);
      case sch::RequestType::SUBMIT:
      case sch::RequestType::CONTINUE:
        emitDecodeTokens(request);
        return true;
      default:
        return true;
    }
  }

  bool try_pop_response(sch::SchedulerResponse& response) override {
    return core_.tryPopResponse(response);
  }

  bool try_pop_output(sch::OutputMessage& output) override {
    return core_.tryPopOutput(output);
  }

  uint32_t get_spec_accepts(uint32_t /*slotId*/) const override { return 0; }
  uint32_t get_spec_rejects(uint32_t /*slotId*/) const override { return 0; }

 private:
  void emitDecodeTokens(const sch::ISRequest& request) {
    if (prefillLatency_.count() > 0) {
      std::this_thread::sleep_for(prefillLatency_);
    }

    const uint32_t maxTokens = request.gen.max_new_tokens;
    const uint32_t basePosition =
        request.position_id.value_or(static_cast<uint32_t>(request.tokens.size()));

    for (uint32_t i = 0; i < maxTokens; ++i) {
      if (i > 0 && decodeTokenLatency_.count() > 0) {
        std::this_thread::sleep_for(decodeTokenLatency_);
      }
      const bool isLast = (i + 1 == maxTokens);
      core_.pushOutput(sch::OutputMessage{
          .slot_id = request.slot_id,
          .token_id = decodeTokenId_,
          .is_complete = isLast,
          .tokens_generated = i + 1,
          .position_id = basePosition + i,
      });
    }
  }

  detail::MockSchedulerCore core_;
  uint32_t decodeTokenId_;
  std::chrono::milliseconds prefillLatency_;
  std::chrono::microseconds decodeTokenLatency_;
};

}  // namespace tt::runners::blaze
