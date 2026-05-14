#pragma once

#include <cstdint>

namespace tt::domain {

struct BaseRequest {
  uint32_t task_id;

  explicit BaseRequest(uint32_t taskId) : task_id(taskId) {}
};

}  // namespace tt::domain
