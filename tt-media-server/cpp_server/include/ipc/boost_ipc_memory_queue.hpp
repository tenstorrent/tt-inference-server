// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/streams/bufferstream.hpp>
#include <concepts>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "domain/manage_memory.hpp"

namespace tt::ipc {

namespace bi_ipc = boost::interprocess;

template <typename T>
concept Serializable =
    requires(const T& t, std::ostream& os, std::istream& is) {
      { t.serialize(os) } -> std::same_as<void>;
      { T::deserialize(is) } -> std::convertible_to<T>;
    };

/**
 * Boost.Interprocess message queue for domain types that implement
 * serialize(std::ostream&) / static deserialize(std::istream&).
 */
template <Serializable MsgType, size_t MaxMsgSize>
class BoostIpcMemoryQueue {
 public:
  static constexpr size_t MAX_MSG_SIZE = MaxMsgSize;

  BoostIpcMemoryQueue(const std::string& name, int capacity)
      : sendBuffer(MAX_MSG_SIZE), recvBuffer(MAX_MSG_SIZE) {
    bi_ipc::message_queue::remove(name.c_str());
    queue = std::make_unique<bi_ipc::message_queue>(
        bi_ipc::create_only, name.c_str(), capacity, MAX_MSG_SIZE);
  }

  // Open an existing queue (for clients)
  static std::unique_ptr<BoostIpcMemoryQueue> openExisting(
      const std::string& name) {
    try {
      auto q =
          std::unique_ptr<BoostIpcMemoryQueue>(new BoostIpcMemoryQueue(name));
      return q;
    } catch (const bi_ipc::interprocess_exception&) {
      return nullptr;
    }
  }

  ~BoostIpcMemoryQueue() {
    try {
      queue.reset();
    } catch (const bi_ipc::interprocess_exception&) {
    }
  }

  BoostIpcMemoryQueue(const BoostIpcMemoryQueue&) = delete;
  BoostIpcMemoryQueue& operator=(const BoostIpcMemoryQueue&) = delete;

  void push(const MsgType& msg, unsigned int priority = 0) {
    std::lock_guard<std::mutex> lock(pushMutex);
    bi_ipc::obufferstream stream(sendBuffer.data(), sendBuffer.size());
    msg.serialize(stream);
    queue->send(sendBuffer.data(), stream.tellp(), priority);
  }

  bool tryPop(MsgType& out) {
    bi_ipc::message_queue::size_type recv_size = 0;
    unsigned int priority = 0;
    if (!queue->try_receive(recvBuffer.data(), recvBuffer.size(), recv_size,
                            priority)) {
      return false;
    }
    bi_ipc::ibufferstream stream(recvBuffer.data(), recv_size);
    out = MsgType::deserialize(stream);
    return true;
  }

  static void remove(const std::string& name) {
    bi_ipc::message_queue::remove(name.c_str());
  }

 private:
  // Private constructor for open_only mode (used by openExisting)
  explicit BoostIpcMemoryQueue(const std::string& name)
      : sendBuffer(MAX_MSG_SIZE), recvBuffer(MAX_MSG_SIZE) {
    queue = std::make_unique<bi_ipc::message_queue>(bi_ipc::open_only,
                                                    name.c_str());
  }

  std::unique_ptr<bi_ipc::message_queue> queue;
  std::mutex pushMutex;
  std::vector<char> sendBuffer;
  std::vector<char> recvBuffer;
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
