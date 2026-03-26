// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "ipc/boost_ipc_cancel_queue.hpp"

#include <boost/interprocess/errors.hpp>

#include "utils/logger.hpp"

namespace tt::ipc {

namespace bi_ipc = boost::interprocess;

using TaskID = tt::domain::TaskID;

static constexpr size_t MSG_SIZE = TaskID::K_SERIALIZED_SIZE;

BoostIpcCancelQueue::BoostIpcCancelQueue(const std::string& name,
                                         size_t capacity)
    : name_(name) {
  queue_ = std::make_unique<bi_ipc::message_queue>(
      bi_ipc::create_only, name.c_str(), capacity, MSG_SIZE);
}

BoostIpcCancelQueue::BoostIpcCancelQueue(const std::string& name)
    : name_(name) {
  queue_ =
      std::make_unique<bi_ipc::message_queue>(bi_ipc::open_only, name.c_str());
}

BoostIpcCancelQueue::~BoostIpcCancelQueue() {
  try {
    queue_.reset();
  } catch (const bi_ipc::interprocess_exception& e) {
    TT_LOG_WARN("[BoostIpcCancelQueue] Destructor: {} (ignored)", e.what());
  }
}

void BoostIpcCancelQueue::push(const TaskID& taskId) {
  auto buf = taskId.ipcSerialize();
  try {
    bool sent = queue_->try_send(buf.data(), buf.size(), /*priority=*/0);
    if (!sent) {
      TT_LOG_WARN("[BoostIpcCancelQueue] Queue full, dropping cancel for {}",
                  taskId.id);
    }
  } catch (const bi_ipc::interprocess_exception& e) {
    TT_LOG_WARN("[BoostIpcCancelQueue] push failed: {}", e.what());
  }
}

void BoostIpcCancelQueue::tryPopAll(std::vector<TaskID>& out) {
  char buf[MSG_SIZE];
  bi_ipc::message_queue::size_type recvSize = 0;
  unsigned int priority = 0;
  while (queue_->try_receive(buf, MSG_SIZE, recvSize, priority)) {
    out.push_back(TaskID::ipcDeserialize(buf, recvSize));
  }
}

void BoostIpcCancelQueue::remove() {
  bi_ipc::message_queue::remove(name_.c_str());
}

void BoostIpcCancelQueue::removeByName(const std::string& name) {
  bi_ipc::message_queue::remove(name.c_str());
}

}  // namespace tt::ipc
