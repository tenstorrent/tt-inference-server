// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/streams/bufferstream.hpp>
#include <concepts>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "domain/manage_memory.hpp"
#include "utils/logger.hpp"

namespace tt::ipc {

namespace bi_ipc = boost::interprocess;

template <typename T>
concept Serializable =
    requires(const T& t, std::ostream& os, std::istream& is) {
      { t.serialize(os) } -> std::same_as<void>;
      { T::deserialize(is) } -> std::convertible_to<T>;
    };

template <typename T>
concept IpcSerializable = Serializable<T> || (std::is_trivially_copyable_v<T> &&
                                              !std::is_pointer_v<T>);

/**
 * Generic Boost.Interprocess message queue.
 *
 * Supports two message flavours at compile time:
 *   - Serializable types: use serialize(ostream) / static deserialize(istream).
 *   - Trivially-copyable types (uint32_t, int64_t, ...): raw memcpy, no
 *     stream overhead.
 */
template <IpcSerializable MsgType, size_t MaxMsgSize>
class BoostIpcMemoryQueue {
 public:
  static constexpr size_t MAX_MSG_SIZE = MaxMsgSize;

  /** Create a new queue (main process). Removes any stale shm left by a
   *  previous process that crashed without cleanup. Falls back to
   *  open_or_create + drain if removal is not possible (e.g. race with
   *  another process, container permission issues). */
  BoostIpcMemoryQueue(const std::string& name, int capacity) : name_(name) {
    bi_ipc::message_queue::remove(name.c_str());
    try {
      queue_ = std::make_unique<bi_ipc::message_queue>(
          bi_ipc::create_only, name.c_str(), capacity, MAX_MSG_SIZE);
    } catch (const bi_ipc::interprocess_exception&) {
      TT_LOG_WARN(
          "[BoostIpcQueue] '{}' still exists after remove, "
          "falling back to open_or_create + drain",
          name);
      queue_ = std::make_unique<bi_ipc::message_queue>(
          bi_ipc::open_or_create, name.c_str(), capacity, MAX_MSG_SIZE);
      drain();
    }
  }

  /** Open an existing queue (worker process). */
  static std::unique_ptr<BoostIpcMemoryQueue> openExisting(
      const std::string& name) {
    try {
      return std::unique_ptr<BoostIpcMemoryQueue>(
          new BoostIpcMemoryQueue(name));
    } catch (const bi_ipc::interprocess_exception& e) {
      TT_LOG_ERROR("[BoostIpcQueue] Failed to open existing queue: {}", name);
      throw std::runtime_error("Failed to open existing queue: " + name + " " +
                               std::to_string(errno) + " " + e.what());
    }
  }

  ~BoostIpcMemoryQueue() {
    try {
      queue_.reset();
    } catch (const bi_ipc::interprocess_exception&) {
    }
  }

  BoostIpcMemoryQueue(const BoostIpcMemoryQueue&) = delete;
  BoostIpcMemoryQueue& operator=(const BoostIpcMemoryQueue&) = delete;

  // -- push (non-blocking, may block if queue is full) ----------------------

  void push(const MsgType& msg, unsigned int priority = 0) {
    if constexpr (Serializable<MsgType>) {
      auto& buf = sendBuffer();
      bi_ipc::obufferstream stream(buf.data(), buf.size());
      msg.serialize(stream);
      queue_->send(buf.data(), stream.tellp(), priority);
    } else {
      queue_->send(reinterpret_cast<const char*>(&msg), sizeof(MsgType),
                   priority);
    }
  }

  bool tryPush(const MsgType& msg, unsigned int priority = 0) {
    if constexpr (Serializable<MsgType>) {
      auto& buf = sendBuffer();
      bi_ipc::obufferstream stream(buf.data(), buf.size());
      msg.serialize(stream);
      return queue_->try_send(buf.data(), stream.tellp(), priority);
    } else {
      return queue_->try_send(reinterpret_cast<const char*>(&msg),
                              sizeof(MsgType), priority);
    }
  }

  // -- pop (non-blocking) ---------------------------------------------------

  bool tryPop(MsgType& out) {
    bi_ipc::message_queue::size_type recv_size = 0;
    unsigned int priority = 0;
    if constexpr (Serializable<MsgType>) {
      auto& buf = recvBuffer();
      if (!queue_->try_receive(buf.data(), buf.size(), recv_size, priority))
        return false;
      bi_ipc::ibufferstream stream(buf.data(), recv_size);
      out = MsgType::deserialize(stream);
    } else {
      if (!queue_->try_receive(reinterpret_cast<char*>(&out), sizeof(MsgType),
                               recv_size, priority))
        return false;
    }
    return true;
  }

  // -- receive (blocking) ---------------------------------------------------

  void receive(MsgType& out) {
    bi_ipc::message_queue::size_type recv_size = 0;
    unsigned int priority = 0;
    if constexpr (Serializable<MsgType>) {
      auto& buf = recvBuffer();
      queue_->receive(buf.data(), buf.size(), recv_size, priority);
      bi_ipc::ibufferstream stream(buf.data(), recv_size);
      out = MsgType::deserialize(stream);
    } else {
      queue_->receive(reinterpret_cast<char*>(&out), sizeof(MsgType), recv_size,
                      priority);
    }
  }

  // -- drain ----------------------------------------------------------------

  void tryPopAll(std::vector<MsgType>& out) {
    MsgType msg{};
    while (tryPop(msg)) {
      out.push_back(std::move(msg));
    }
  }

  // -- queries --------------------------------------------------------------

  bool empty() const { return queue_->get_num_msg() == 0; }

  // -- cleanup --------------------------------------------------------------

  void remove() { remove(name_); }

  static void remove(const std::string& name) {
    bi_ipc::message_queue::remove(name.c_str());
  }

 private:
  static std::vector<char>& sendBuffer() {
    thread_local std::vector<char> buf(MAX_MSG_SIZE);
    return buf;
  }

  static std::vector<char>& recvBuffer() {
    thread_local std::vector<char> buf(MAX_MSG_SIZE);
    return buf;
  }

  void drain() {
    auto& buf = recvBuffer();
    bi_ipc::message_queue::size_type recv_size = 0;
    unsigned int priority = 0;
    while (queue_->try_receive(buf.data(), buf.size(), recv_size, priority)) {
    }
  }

  explicit BoostIpcMemoryQueue(const std::string& name) : name_(name) {
    queue_ = std::make_unique<bi_ipc::message_queue>(bi_ipc::open_only,
                                                     name.c_str());
  }

  std::string name_;
  std::unique_ptr<bi_ipc::message_queue> queue_;
};

constexpr size_t MEMORY_REQUEST_MAX_MSG_SIZE = 256;
constexpr size_t MEMORY_RESULT_MAX_MSG_SIZE = 4096;
constexpr int MEMORY_QUEUE_CAPACITY = 64;

using MemoryRequestQueue =
    BoostIpcMemoryQueue<domain::ManageMemoryTask, MEMORY_REQUEST_MAX_MSG_SIZE>;
using MemoryResultQueue =
    BoostIpcMemoryQueue<domain::ManageMemoryResult, MEMORY_RESULT_MAX_MSG_SIZE>;

}  // namespace tt::ipc
