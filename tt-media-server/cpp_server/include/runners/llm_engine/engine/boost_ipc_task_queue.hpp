// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <string>

#include <boost/interprocess/ipc/message_queue.hpp>

#include "llm_engine/engine/task_queue.hpp"

namespace llm_engine {

/**
 * ITaskQueue implementation backed by a Boost.Interprocess message queue.
 */
class BoostIpcTaskQueue : public ITaskQueue {
 public:
  BoostIpcTaskQueue(const std::string& name);

  void push(const Sequence& seq) override;
  Sequence* try_pop() override;
  bool empty() const override;

  /** Remove the named shared-memory queue (cleanup helper). */
  static void remove(const std::string& name);

 private:
  std::unique_ptr<boost::interprocess::message_queue> queue_;
  std::vector<char> send_buffer_;
  std::vector<char> recv_buffer_;
};

}  // namespace llm_engine
