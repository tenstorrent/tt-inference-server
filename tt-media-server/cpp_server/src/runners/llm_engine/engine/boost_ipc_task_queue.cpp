// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "llm_engine/engine/boost_ipc_task_queue.hpp"
#include <boost/interprocess/streams/bufferstream.hpp>
#include <boost/interprocess/errors.hpp>
#include <iostream>

namespace llm_engine {

namespace ipc = boost::interprocess;

BoostIpcTaskQueue::~BoostIpcTaskQueue() {
  try {
    queue_.reset();
  } catch (const ipc::interprocess_exception& e) {
    std::cerr << "[BoostIpcTaskQueue] Destructor: " << e.what() << " (ignored)\n" << std::flush;
  }
}

BoostIpcTaskQueue::BoostIpcTaskQueue(const std::string& name) {
  queue_ = std::make_unique<ipc::message_queue>(ipc::open_only, name.c_str());
  send_buffer_.resize(queue_->get_max_msg_size());
  recv_buffer_.resize(queue_->get_max_msg_size());
}

BoostIpcTaskQueue::BoostIpcTaskQueue(const std::string& name, int size) {
  queue_ = std::make_unique<ipc::message_queue>(ipc::create_only, name.c_str(), size, MAX_MSG_SIZE);
  send_buffer_.resize(queue_->get_max_msg_size());
  recv_buffer_.resize(queue_->get_max_msg_size());
}

void BoostIpcTaskQueue::push(const Sequence& seq) {
  std::lock_guard<std::mutex> lock(push_mutex_);
  ipc::obufferstream stream(send_buffer_.data(), send_buffer_.size());
  seq.serialize(stream);
  auto bytes_written = stream.tellp();
  queue_->send(send_buffer_.data(), bytes_written, /*priority=*/0);
}

Sequence* BoostIpcTaskQueue::try_pop() {
  ipc::message_queue::size_type recv_size = 0;
  unsigned int priority = 0;

  if (!queue_->try_receive(recv_buffer_.data(), recv_buffer_.size(), recv_size, priority)) {
    return nullptr;
  }
  ipc::ibufferstream recv_stream(recv_buffer_.data(), recv_size);
  return Sequence::deserialize(recv_stream);
}

bool BoostIpcTaskQueue::empty() const {
  return queue_->get_num_msg() == 0;
}

void BoostIpcTaskQueue::remove(const std::string& name) {
  ipc::message_queue::remove(name.c_str());
}

}  // namespace llm_engine
