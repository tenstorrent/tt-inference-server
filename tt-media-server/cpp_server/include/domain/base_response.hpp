#pragma once

#include "domain/task_id.hpp"

namespace tt::domain {

struct BaseResponse {
    TaskID task_id;
};

} // namespace tt::domain
