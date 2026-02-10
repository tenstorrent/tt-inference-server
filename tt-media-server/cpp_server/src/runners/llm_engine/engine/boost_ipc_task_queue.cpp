// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "llm_engine/engine/boost_ipc_task_queue.hpp"
#include <boost/interprocess/streams/bufferstream.hpp>

namespace llm_engine {

namespace ipc = boost::interprocess;

BoostIpcTaskQueue::BoostIpcTaskQueue(const std::string& name) {
  queue_ = std::make_unique<ipc::message_queue>(ipc::open_only, name.c_str());
  send_buffer_.resize(queue_->get_max_msg_size());
  recv_buffer_.resize(queue_->get_max_msg_size());
}

void BoostIpcTaskQueue::push(const Sequence& seq) {
  ipc::obufferstream stream(send_buffer_.data(), send_buffer_.size());
  seq.serialize(stream);
  auto bytes_written = stream.tellp();
  queue_->send(send_buffer_.data(), bytes_written, /*priority=*/0);
}

Sequence* BoostIpcTaskQueue::try_pop() {
  auto max_msg_size = queue_->get_max_msg_size();
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
