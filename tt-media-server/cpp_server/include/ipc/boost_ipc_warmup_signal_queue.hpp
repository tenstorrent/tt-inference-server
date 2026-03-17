// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <boost/interprocess/ipc/message_queue.hpp>
#include <memory>
#include <string>

#include "ipc/warmup_signal_queue.hpp"

namespace tt::ipc {

constexpr const char* WARMUP_SIGNALS_QUEUE_NAME = "tt_warmup_signals";
constexpr size_t WARMUP_SIGNAL_MSG_SIZE = sizeof(int64_t);

/**
 * IWarmupSignalQueue implementation using Boost.Interprocess message_queue.
 */
class BoostIpcWarmupSignalQueue : public IWarmupSignalQueue {
 public:
  /** Open existing queue (worker side). */
  explicit BoostIpcWarmupSignalQueue(const std::string& name);

  /** Create queue (main process side). */
  BoostIpcWarmupSignalQueue(const std::string& name, size_t capacity);

  ~BoostIpcWarmupSignalQueue() override;

  void sendReady(int workerId) override;
  void receive(int& workerId) override;
  void remove() override;

  static void remove(const std::string& name);

 private:
  std::string name_;
  std::unique_ptr<boost::interprocess::message_queue> queue_;
};

}  // namespace tt::ipc
