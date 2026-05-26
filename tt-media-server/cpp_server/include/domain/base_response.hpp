// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace tt::domain {

struct BaseResponse {
  uint32_t task_id;

  explicit BaseResponse(uint32_t taskId) : task_id(taskId) {}
};

}  // namespace tt::domain
