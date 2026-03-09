#pragma once

#include "domain/task_id.hpp"

namespace tt::domain {

struct BaseRequest {
    TaskID task_id;
};

} // namespace tt::domain
