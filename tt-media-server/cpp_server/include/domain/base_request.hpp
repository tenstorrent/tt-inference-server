#pragma once

#include "domain/task_id.hpp"

namespace tt::domain {

struct BaseRequest {
  TaskID task_id;

  explicit BaseRequest(TaskID taskId) : task_id(std::move(taskId)) {}
};

}  // namespace tt::domain
