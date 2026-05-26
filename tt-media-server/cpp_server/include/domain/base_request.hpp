// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace tt::domain {

struct BaseRequest {
  uint32_t task_id;

  explicit BaseRequest(uint32_t taskId) : task_id(taskId) {}
};

}  // namespace tt::domain
