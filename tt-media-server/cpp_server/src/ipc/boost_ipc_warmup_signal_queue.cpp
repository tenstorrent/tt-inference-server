// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "ipc/boost_ipc_warmup_signal_queue.hpp"

#include <boost/interprocess/errors.hpp>

#include "utils/logger.hpp"

namespace tt::ipc {

namespace bi_ipc = boost::interprocess;

BoostIpcWarmupSignalQueue::~BoostIpcWarmupSignalQueue() {
  try {
    queue_.reset();
  } catch (const bi_ipc::interprocess_exception& e) {
    TT_LOG_WARN("[BoostIpcWarmupSignalQueue] Destructor: {} (ignored)",
                e.what());
  }
}

BoostIpcWarmupSignalQueue::BoostIpcWarmupSignalQueue(const std::string& name)
    : name_(name) {
  queue_ =
      std::make_unique<bi_ipc::message_queue>(bi_ipc::open_only, name.c_str());
}

BoostIpcWarmupSignalQueue::BoostIpcWarmupSignalQueue(const std::string& name,
                                                     size_t capacity)
    : name_(name) {
  queue_ = std::make_unique<bi_ipc::message_queue>(
      bi_ipc::create_only, name.c_str(), capacity, WARMUP_SIGNAL_MSG_SIZE);
}

void BoostIpcWarmupSignalQueue::remove() {
  BoostIpcWarmupSignalQueue::remove(name_);
}

void BoostIpcWarmupSignalQueue::sendReady(int workerId) {
  int64_t payload = static_cast<int64_t>(workerId);
  queue_->send(&payload, sizeof(payload), /*priority=*/0);
}

int BoostIpcWarmupSignalQueue::receive() {
  bi_ipc::message_queue::size_type recvSize = 0;
  unsigned int priority = 0;
  int64_t payload = 0;
  queue_->receive(&payload, sizeof(payload), recvSize, priority);
  return static_cast<int>(payload);
}

void BoostIpcWarmupSignalQueue::remove(const std::string& name) {
  bi_ipc::message_queue::remove(name.c_str());
}

}  // namespace tt::ipc
