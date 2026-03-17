// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "ipc/boost_ipc_task_queue.hpp"

#include <boost/interprocess/errors.hpp>
#include <boost/interprocess/streams/bufferstream.hpp>

#include "utils/logger.hpp"

namespace tt::ipc {

namespace bi_ipc = boost::interprocess;

BoostIpcTaskQueue::~BoostIpcTaskQueue() {
  try {
    queue_.reset();
  } catch (const bi_ipc::interprocess_exception& e) {
    TT_LOG_WARN("[BoostIpcTaskQueue] Destructor: {} (ignored)", e.what());
  }
}

BoostIpcTaskQueue::BoostIpcTaskQueue(const std::string& name) {
  queue_ =
      std::make_unique<bi_ipc::message_queue>(bi_ipc::open_only, name.c_str());
  send_buffer_.resize(queue_->get_max_msg_size());
  recv_buffer_.resize(queue_->get_max_msg_size());
}

BoostIpcTaskQueue::BoostIpcTaskQueue(const std::string& name, int size) {
  queue_ = std::make_unique<bi_ipc::message_queue>(
      bi_ipc::create_only, name.c_str(), size, MAX_MSG_SIZE);
  send_buffer_.resize(queue_->get_max_msg_size());
  recv_buffer_.resize(queue_->get_max_msg_size());
}

void BoostIpcTaskQueue::push(const llm_engine::Sequence& seq) {
  std::lock_guard<std::mutex> lock(push_mutex_);
  bi_ipc::obufferstream stream(send_buffer_.data(), send_buffer_.size());
  seq.serialize(stream);
  auto bytesWritten = stream.tellp();
  queue_->send(send_buffer_.data(), bytesWritten, /*priority=*/0);
}

llm_engine::Sequence* BoostIpcTaskQueue::try_pop() {
  bi_ipc::message_queue::size_type recvSize = 0;
  unsigned int priority = 0;

  if (!queue_->try_receive(recv_buffer_.data(), recv_buffer_.size(), recvSize,
                           priority)) {
    return nullptr;
  }
  bi_ipc::ibufferstream recvStream(recv_buffer_.data(), recvSize);
  return llm_engine::Sequence::deserialize(recvStream);
}

llm_engine::Sequence* BoostIpcTaskQueue::receive() {
  bi_ipc::message_queue::size_type recvSize = 0;
  unsigned int priority = 0;
  queue_->receive(recv_buffer_.data(), recv_buffer_.size(), recvSize, priority);
  bi_ipc::ibufferstream recvStream(recv_buffer_.data(), recvSize);
  return llm_engine::Sequence::deserialize(recvStream);
}

bool BoostIpcTaskQueue::empty() const { return queue_->get_num_msg() == 0; }

void BoostIpcTaskQueue::remove(const std::string& name) {
  bi_ipc::message_queue::remove(name.c_str());
}

}  // namespace tt::ipc
