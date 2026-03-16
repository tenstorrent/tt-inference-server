// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>

#include <boost/interprocess/ipc/message_queue.hpp>

namespace tt::ipc {

constexpr const char* CANCEL_QUEUE_NAME = "tt_cancel";

/**
 * Fixed-size cancel message sent from main process to worker.
 * task_id field size matches SharedToken::task_id.
 */
struct CancelMessage {
    char task_id[56];
};

/**
 * One-way IPC queue for cancel signals (M3).
 *
 * The main process (LLMService::cancel_request) pushes a task_id whenever a
 * request is cancelled.  The worker process (LLMRunner::step) drains the queue
 * at the top of every step and forwards the task_ids to Scheduler::cancel().
 *
 * Only LLMRunner / Scheduler is affected; SpPipelineRunner has its own
 * in-process cancel path and does not use this queue.
 */
class CancelQueue {
public:
    /** Create a new cancel queue (main-process side). */
    CancelQueue(const std::string& name, int max_messages);

    /** Open an existing cancel queue (worker-process side). */
    explicit CancelQueue(const std::string& name);

    ~CancelQueue();

    /** Push a cancel request. Non-blocking; returns false if queue is full. */
    bool push(const std::string& task_id);

    /** Try to pop the next pending cancellation. Returns false if queue is empty. */
    bool try_pop(std::string& task_id);

    /** Remove the named shared-memory queue (cleanup helper). */
    static void remove(const std::string& name);

private:
    std::unique_ptr<boost::interprocess::message_queue> queue_;
};

} // namespace tt::ipc
