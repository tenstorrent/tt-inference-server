// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Reusable background thread that auto-responds to memory ALLOCATE requests.
// Used by both integration tests (TestServer) and e2e tests (disaggregated).

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "domain/manage_memory.hpp"

namespace tt::test {

// Background thread that auto-acknowledges memory ALLOCATE requests with
// SUCCESS + slotId=0. Tests that need to inspect requests or inject custom
// responses can call setAutoRespond(false) to pause the responder.
//
// Template parameters allow use with both boost and in_memory queues.
template <typename MemoryRequestQueue, typename MemoryResultQueue>
class MemoryAutoResponder {
 public:
  MemoryAutoResponder(std::shared_ptr<MemoryRequestQueue> requestQueue,
                      std::shared_ptr<MemoryResultQueue> resultQueue)
      : requestQueue_(std::move(requestQueue)),
        resultQueue_(std::move(resultQueue)) {
    start();
  }

  ~MemoryAutoResponder() { stop(); }

  // Non-copyable, non-movable (owns a thread).
  MemoryAutoResponder(const MemoryAutoResponder&) = delete;
  MemoryAutoResponder& operator=(const MemoryAutoResponder&) = delete;
  MemoryAutoResponder(MemoryAutoResponder&&) = delete;
  MemoryAutoResponder& operator=(MemoryAutoResponder&&) = delete;

  // Toggle auto-respond behavior. When OFF, ALLOCATE requests are not consumed.
  void setAutoRespond(bool on) { autoRespond_.store(on); }

  bool isAutoResponding() const { return autoRespond_.load(); }

 private:
  void start() {
    thread_ = std::thread([this] {
      domain::ManageMemoryTask req{};
      while (!stopRequested_.load()) {
        if (!autoRespond_.load() || !requestQueue_->tryPop(req)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          continue;
        }
        if (req.action == domain::MemoryManagementAction::ALLOCATE) {
          domain::ManageMemoryResult res{};
          res.taskId = req.taskId;
          res.status = domain::ManageMemoryStatus::SUCCESS;
          res.slotId = 0;
          resultQueue_->push(res);
        }
        // DEALLOCATE / MOVE: no response expected by the default path.
      }
    });
  }

  void stop() {
    stopRequested_.store(true);
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  std::shared_ptr<MemoryRequestQueue> requestQueue_;
  std::shared_ptr<MemoryResultQueue> resultQueue_;
  std::thread thread_;
  std::atomic<bool> autoRespond_{true};
  std::atomic<bool> stopRequested_{false};
};

}  // namespace tt::test
