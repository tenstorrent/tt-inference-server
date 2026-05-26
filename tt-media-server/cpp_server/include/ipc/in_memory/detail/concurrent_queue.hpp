// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "utils/concurrent_queue.hpp"

namespace tt::ipc::in_memory::detail {

template <typename T>
using ConcurrentQueue = tt::utils::BlockingQueue<T>;

}  // namespace tt::ipc::in_memory::detail
