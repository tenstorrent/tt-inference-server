#pragma once

#include <string>

namespace tt::domain {

struct BaseRequest {
        // Internal task tracking
        std::string task_id;
    };

} // namespace tt::domain