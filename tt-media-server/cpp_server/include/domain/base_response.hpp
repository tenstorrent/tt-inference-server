#pragma once

#include "domain/task_id.hpp"

namespace tt::domain {

struct BaseResponse {
  TaskID task_id;

  explicit BaseResponse(TaskID task_id) : task_id(std::move(task_id)) {}
};

}  // namespace tt::domain
