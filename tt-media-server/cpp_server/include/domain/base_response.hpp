// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <string>

namespace tt::domain {

struct BaseResponse {
    std::string task_id;
};

} // namespace tt::domain
