// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <boost/interprocess/ipc/message_queue.hpp>
#include <memory>
#include <string>

#include "ipc/cancel_queue.hpp"

namespace tt::ipc {

/**
 * ICancelQueue implementation backed by a Boost.Interprocess message queue.
 *
 * One queue per worker. The main process creates the queue; the worker opens
 * it.
 */
class BoostIpcCancelQueue : public ICancelQueue {
 public:
  /** Create a new queue (main process). */
  BoostIpcCancelQueue(const std::string& name, size_t capacity);

  /** Open an existing queue (worker process). */
  explicit BoostIpcCancelQueue(const std::string& name);

  ~BoostIpcCancelQueue() override;

  void push(uint32_t taskId) override;
  void tryPopAll(std::vector<uint32_t>& out) override;
  void remove() override;

  /** Remove a named queue (cleanup helper). */
  static void removeByName(const std::string& name);

 private:
  std::string name_;
  std::unique_ptr<boost::interprocess::message_queue> queue_;
  std::vector<char> recv_buffer_;
};

}  // namespace tt::ipc
