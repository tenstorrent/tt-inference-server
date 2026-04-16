// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <iostream>

namespace tt::ipc {

struct SharedToken {
  uint32_t token_index = 0;
  uint32_t flags = 0;
  uint64_t token_id = 0;
  uint32_t task_id = 0;

  static constexpr uint32_t FLAG_FINAL = 1;
  static constexpr uint32_t FLAG_ERROR = 2;
  static constexpr uint32_t FLAG_DONE = 4;

  bool isFinal() const { return flags & FLAG_FINAL; }
  bool isError() const { return flags & FLAG_ERROR; }
  bool isDone() const { return flags & FLAG_DONE; }

  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&token_index), sizeof(token_index));
    os.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    os.write(reinterpret_cast<const char*>(&token_id), sizeof(token_id));
    os.write(reinterpret_cast<const char*>(&task_id), sizeof(task_id));
  }

  static SharedToken deserialize(std::istream& is) {
    SharedToken token{};
    is.read(reinterpret_cast<char*>(&token.token_index),
            sizeof(token.token_index));
    is.read(reinterpret_cast<char*>(&token.flags), sizeof(token.flags));
    is.read(reinterpret_cast<char*>(&token.token_id), sizeof(token.token_id));
    is.read(reinterpret_cast<char*>(&token.task_id), sizeof(token.task_id));
    return token;
  }
};

/**
 * Abstract interface for a token result queue (worker -> main process).
 *
 * - push        -- non-blocking enqueue, returns false if full.
 * - tryPop      -- non-blocking dequeue.
 * - blockingPop -- blocks until a token is available or shutdown.
 * - shutdown    -- signal consumers to stop.
 */
class IResultQueue {
 public:
  virtual ~IResultQueue() = default;

  virtual bool push(const SharedToken& token) = 0;
  virtual bool tryPop(SharedToken& out) = 0;
  virtual bool blockingPop(SharedToken& out) = 0;
  virtual bool empty() const = 0;
  virtual void shutdown() = 0;
  virtual bool isShutdown() const = 0;
  virtual void remove() {}
};

}  // namespace tt::ipc
