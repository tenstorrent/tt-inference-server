#pragma once

#include <cstdint>

namespace tt::domain {

struct BaseResponse {
  uint32_t task_id;

  explicit BaseResponse(uint32_t taskId) : task_id(taskId) {}
};

}  // namespace tt::domain
