// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "ipc/in_memory/in_memory_task_queue.hpp"

namespace tt::runners::llm_engine {

using InMemoryTaskQueue = tt::ipc::in_memory::TaskQueue;

}  // namespace tt::runners::llm_engine
