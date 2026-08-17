// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "domain/manage_memory.hpp"

namespace tt::ipc {

class IMemoryRequestQueue {
 public:
  virtual ~IMemoryRequestQueue() = default;

  virtual void push(const tt::domain::ManageMemoryTask& task) = 0;
  virtual bool tryPop(tt::domain::ManageMemoryTask& out) = 0;
};

class IMemoryResultQueue {
 public:
  virtual ~IMemoryResultQueue() = default;

  virtual void push(const tt::domain::ManageMemoryResult& result) = 0;
  virtual bool waitPop(tt::domain::ManageMemoryResult& out) = 0;
};

}  // namespace tt::ipc
