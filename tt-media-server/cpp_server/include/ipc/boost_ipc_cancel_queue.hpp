// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <boost/interprocess/ipc/message_queue.hpp>
#include <memory>
#include <string>
#include <vector>

#include "domain/task_id.hpp"

namespace tt::ipc {

constexpr const char* CANCEL_QUEUE_NAME = "tt_cancels";
constexpr size_t CANCEL_QUEUE_CAPACITY = 1024;

/**
 * Lightweight IPC message queue for carrying request-cancel signals from the
 * main process to worker processes. Each message is exactly one serialized
 * TaskID (36 bytes).
 *
 * One queue per worker; the main process creates each queue and the
 * corresponding worker process opens it to drain cancel signals.
 * push() is non-blocking (drops the message and logs a warning when full).
 * tryPopAll() drains every available message without blocking.
 */
class BoostIpcCancelQueue {
 public:
  /** Create the queue (main-process side). */
  BoostIpcCancelQueue(const std::string& name, size_t capacity);

  /** Open an existing queue (worker-process side). */
  explicit BoostIpcCancelQueue(const std::string& name);

  ~BoostIpcCancelQueue();

  /**
   * Non-blocking push. Silently drops the message if the queue is full
   * (the request will finish on its own shortly anyway).
   * Thread-safe.
   */
  void push(const tt::domain::TaskID& taskId);

  /**
   * Drain all available cancel messages into out without blocking.
   * Intended to be called once per scheduler step from the worker thread.
   */
  void tryPopAll(std::vector<tt::domain::TaskID>& out);

  static void remove(const std::string& name);

 private:
  std::string name_;
  std::unique_ptr<boost::interprocess::message_queue> queue_;
};

}  // namespace tt::ipc
