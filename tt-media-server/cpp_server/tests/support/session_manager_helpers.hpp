// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Shared SessionManager helpers for integration tests and benchmarks.

#pragma once

#include <trantor/net/EventLoop.h>

#include <future>
#include <optional>
#include <span>
#include <string>
#include <thread>

#include "services/session_manager.hpp"

namespace tt::test {

// Trantor requires an EventLoop to be created and run on the same thread.
struct TrantorLoopFixture {
  std::promise<trantor::EventLoop*> promise_;
  trantor::EventLoop* loop{nullptr};
  std::thread loopThread;

  TrantorLoopFixture() {
    auto future = promise_.get_future();
    loopThread = std::thread([this]() {
      trantor::EventLoop eventLoop;
      promise_.set_value(&eventLoop);
      eventLoop.loop();
    });
    loop = future.get();
  }

  ~TrantorLoopFixture() {
    if (loop) loop->quit();
    if (loopThread.joinable()) loopThread.join();
  }

  TrantorLoopFixture(const TrantorLoopFixture&) = delete;
  TrantorLoopFixture& operator=(const TrantorLoopFixture&) = delete;
};

struct GetSlotOutcome {
  std::optional<services::SlotAcquireResult> result;
  std::optional<std::string> error;
  bool rateLimited = false;
};

// Synchronous wrapper around SessionManager::getSlot.
// Requires a running TrantorLoopFixture event loop.
inline GetSlotOutcome callGetSlot(services::SessionManager& manager,
                                  trantor::EventLoop* loop,
                                  std::span<const uint32_t> promptTokenIds,
                                  services::GetSlotOptions opts) {
  std::promise<GetSlotOutcome> promise;
  auto future = promise.get_future();

  try {
    manager.getSlot(
        promptTokenIds, std::move(opts), loop,
        [&promise](services::SlotAcquireResult result) {
          promise.set_value({.result = std::move(result)});
        },
        [&promise](const std::string& error) {
          promise.set_value({.error = error});
        });
  } catch (const services::SessionInFlightException&) {
    return {.rateLimited = true};
  }

  return future.get();
}

}  // namespace tt::test
