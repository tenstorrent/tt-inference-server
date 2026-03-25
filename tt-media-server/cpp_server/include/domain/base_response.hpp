#pragma once

#include "domain/task_id.hpp"

namespace tt::domain {

struct BaseResponse {
  TaskID task_id;

  explicit BaseResponse(TaskID taskId) : task_id(std::move(taskId)) {}
};

}  // namespace tt::domain
