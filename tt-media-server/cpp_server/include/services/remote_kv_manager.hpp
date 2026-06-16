// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <string>
#include <ctime>

namespace tt::services {

enum class MigrationStatus {
    UNKNOWN,
    IN_PROGRESS,
    SUCCESSFUL,
    FAILED,
};

struct Migration {
    std::string migration_id;
    std::time_t timestamp;
    MigrationStatus status;
};

} // namespace tt::services