// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "ipc/boost_ipc_cancel_queue.hpp"

#include "utils/logger.hpp"

namespace tt::ipc {

namespace bip = boost::interprocess;

BoostIpcCancelQueue::BoostIpcCancelQueue(const std::string& name,
                                         size_t capacity)
    : name_(name),
      queue_(std::make_unique<bip::message_queue>(
          bip::create_only, name.c_str(), capacity,
          domain::TaskIDGenerator::K_SERIALIZED_SIZE)),
      recv_buffer_(queue_->get_max_msg_size()) {}

BoostIpcCancelQueue::BoostIpcCancelQueue(const std::string& name)
    : name_(name),
      queue_(
          std::make_unique<bip::message_queue>(bip::open_only, name.c_str())),
      recv_buffer_(queue_->get_max_msg_size()) {}

BoostIpcCancelQueue::~BoostIpcCancelQueue() {
  try {
    queue_.reset();
  } catch (const bip::interprocess_exception& e) {
    TT_LOG_WARN("[BoostIpcCancelQueue] Destructor: {} (ignored)", e.what());
  }
}

void BoostIpcCancelQueue::push(const domain::TaskID& taskId) {
  auto buf = domain::TaskIDGenerator::serialize(taskId);
  if (!queue_->try_send(buf.data(), buf.size(), 0)) {
    TT_LOG_WARN("[CancelQueue] Queue '{}' full, dropping cancel for task_id={}",
                name_, taskId);
  }
}

void BoostIpcCancelQueue::tryPopAll(std::vector<domain::TaskID>& out) {
  bip::message_queue::size_type recvdSize;
  unsigned int priority;
  while (queue_->try_receive(recv_buffer_.data(), recv_buffer_.size(),
                             recvdSize, priority)) {
    out.push_back(
        domain::TaskIDGenerator::deserialize(recv_buffer_.data(), recvdSize));
  }
}

void BoostIpcCancelQueue::remove() { removeByName(name_); }

void BoostIpcCancelQueue::removeByName(const std::string& name) {
  bip::message_queue::remove(name.c_str());
}

}  // namespace tt::ipc
