#pragma once

#include "domain/task_id.hpp"

namespace tt::domain {

struct BaseRequest {
  TaskID task_id;

  explicit BaseRequest(TaskID task_id) : task_id(std::move(task_id)) {}
};

}  // namespace tt::domain
