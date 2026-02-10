// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "llm_engine/engine/boost_ipc_task_queue.hpp"

#include "llm_engine/engine/sequence_message.hpp"

namespace llm_engine {

namespace ipc = boost::interprocess;

BoostIpcTaskQueue::BoostIpcTaskQueue(const std::string& name, int capacity) {
  queue_ = std::make_unique<ipc::message_queue>(
      ipc::open_or_create, name.c_str(),
      static_cast<ipc::message_queue::size_type>(capacity),
      sizeof(SequenceMessage));
}

void BoostIpcTaskQueue::push(const Sequence& seq) {
  SequenceMessage msg = to_sequence_message(seq);
  queue_->send(&msg, sizeof(msg), /*priority=*/0);
}

Sequence* BoostIpcTaskQueue::try_pop() {
  SequenceMessage msg;
  ipc::message_queue::size_type recv_size = 0;
  unsigned int priority = 0;

  if (!queue_->try_receive(&msg, sizeof(msg), recv_size, priority)) {
    return nullptr;
  }
  return from_sequence_message(msg);
}

bool BoostIpcTaskQueue::empty() const {
  return queue_->get_num_msg() == 0;
}

void BoostIpcTaskQueue::remove(const std::string& name) {
  ipc::message_queue::remove(name.c_str());
}

}  // namespace llm_engine
