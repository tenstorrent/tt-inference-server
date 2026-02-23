// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <string>

namespace tt::runners {

struct EmbeddingConfig {
    std::string device_id = "device_0";
    int visible_device = 0;
};

}  // namespace tt::runners
