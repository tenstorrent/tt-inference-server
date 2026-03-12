// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <mutex>
#include <string>

#include <boost/interprocess/ipc/message_queue.hpp>

#include "runners/llm_runner/task_queue.hpp"
#include "runners/llm_runner/config.hpp"

namespace tt::ipc {

/**
 * ITaskQueue implementation backed by a Boost.Interprocess message queue.
 */

class BoostIpcTaskQueue : public llm_engine::ITaskQueue {
 public:
  /** Reserve for task_id, block_table, and other Sequence fields besides token_ids_. */
  static constexpr size_t MAX_SEQUENCE_NON_TOKEN_BYTES = 4096;
  static constexpr size_t MAX_MSG_SIZE =
      llm_engine::Config::MAX_INPUT_TOKENS * sizeof(int64_t) + MAX_SEQUENCE_NON_TOKEN_BYTES;

  BoostIpcTaskQueue(const std::string& name);
  BoostIpcTaskQueue(const std::string& name, int size);
  ~BoostIpcTaskQueue();

  void push(const llm_engine::Sequence& seq) override;
  llm_engine::Sequence* receive() override;
  llm_engine::Sequence* try_pop() override;
  bool empty() const override;

  /** Remove the named shared-memory queue (cleanup helper). */
  static void remove(const std::string& name);

 private:
  std::unique_ptr<boost::interprocess::message_queue> queue_;
  std::mutex push_mutex_;
  std::vector<char> send_buffer_;
  std::vector<char> recv_buffer_;
};

}  // namespace tt::ipc
