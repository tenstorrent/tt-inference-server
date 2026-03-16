// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "ipc/cancel_queue.hpp"
#include "utils/logger.hpp"

#include <cstring>

namespace tt::ipc {

namespace bi_ipc = boost::interprocess;

CancelQueue::CancelQueue(const std::string& name, int max_messages) {
    queue_ = std::make_unique<bi_ipc::message_queue>(
        bi_ipc::create_only, name.c_str(),
        static_cast<size_t>(max_messages), sizeof(CancelMessage));
}

CancelQueue::CancelQueue(const std::string& name) {
    queue_ = std::make_unique<bi_ipc::message_queue>(bi_ipc::open_only, name.c_str());
}

CancelQueue::~CancelQueue() {
    try {
        queue_.reset();
    } catch (const bi_ipc::interprocess_exception& e) {
        TT_LOG_WARN("[CancelQueue] Destructor: {} (ignored)", e.what());
    }
}

bool CancelQueue::push(const std::string& task_id) {
    CancelMessage msg{};
    strncpy(msg.task_id, task_id.c_str(), sizeof(msg.task_id) - 1);
    return queue_->try_send(&msg, sizeof(msg), /*priority=*/0);
}

bool CancelQueue::try_pop(std::string& task_id) {
    CancelMessage msg{};
    bi_ipc::message_queue::size_type recv_size = 0;
    unsigned int priority = 0;
    if (!queue_->try_receive(&msg, sizeof(msg), recv_size, priority)) {
        return false;
    }
    task_id = std::string(msg.task_id, strnlen(msg.task_id, sizeof(msg.task_id)));
    return true;
}

void CancelQueue::remove(const std::string& name) {
    bi_ipc::message_queue::remove(name.c_str());
}

} // namespace tt::ipc
