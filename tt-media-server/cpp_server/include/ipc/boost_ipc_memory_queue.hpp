// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/streams/bufferstream.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "domain/manage_memory.hpp"

namespace tt::ipc {

namespace bi_ipc = boost::interprocess;

/**
 * Boost.Interprocess message queue for domain types that implement
 * serialize(std::ostream&) / static deserialize(std::istream&).
 */
template <typename MsgType, size_t MaxMsgSize>
class BoostIpcMemoryQueue {
 public:
  static constexpr size_t MAX_MSG_SIZE = MaxMsgSize;

  BoostIpcMemoryQueue(const std::string& name, int capacity)
      : send_buffer_(MAX_MSG_SIZE), recv_buffer_(MAX_MSG_SIZE) {
    bi_ipc::message_queue::remove(name.c_str());
    queue_ = std::make_unique<bi_ipc::message_queue>(
        bi_ipc::create_only, name.c_str(), capacity, MAX_MSG_SIZE);
  }

  ~BoostIpcMemoryQueue() {
    try {
      queue_.reset();
    } catch (const bi_ipc::interprocess_exception&) {
    }
  }

  BoostIpcMemoryQueue(const BoostIpcMemoryQueue&) = delete;
  BoostIpcMemoryQueue& operator=(const BoostIpcMemoryQueue&) = delete;

  void push(const MsgType& msg) {
    std::lock_guard<std::mutex> lock(push_mutex_);
    bi_ipc::obufferstream stream(send_buffer_.data(), send_buffer_.size());
    msg.serialize(stream);
    queue_->send(send_buffer_.data(), stream.tellp(), /*priority=*/0);
  }

  bool tryPop(MsgType& out) {
    bi_ipc::message_queue::size_type recv_size = 0;
    unsigned int priority = 0;
    if (!queue_->try_receive(recv_buffer_.data(), recv_buffer_.size(),
                             recv_size, priority)) {
      return false;
    }
    bi_ipc::ibufferstream stream(recv_buffer_.data(), recv_size);
    out = MsgType::deserialize(stream);
    return true;
  }

  static void remove(const std::string& name) {
    bi_ipc::message_queue::remove(name.c_str());
  }

 private:
  std::unique_ptr<bi_ipc::message_queue> queue_;
  std::mutex push_mutex_;
  std::vector<char> send_buffer_;
  std::vector<char> recv_buffer_;
};

constexpr size_t MEMORY_REQUEST_MAX_MSG_SIZE = 256;
constexpr size_t MEMORY_RESULT_MAX_MSG_SIZE = 4096;
constexpr int MEMORY_QUEUE_CAPACITY = 64;

inline constexpr const char* k_memory_request_queue_name = "tt_mem_requests";
inline constexpr const char* k_memory_result_queue_name = "tt_mem_results";

using MemoryRequestQueue =
    BoostIpcMemoryQueue<domain::ManageMemoryTask, MEMORY_REQUEST_MAX_MSG_SIZE>;
using MemoryResultQueue =
    BoostIpcMemoryQueue<domain::ManageMemoryResult, MEMORY_RESULT_MAX_MSG_SIZE>;

}  // namespace tt::ipc
