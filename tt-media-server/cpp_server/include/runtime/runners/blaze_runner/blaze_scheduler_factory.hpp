// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>

#include "config/runner_config.hpp"
#include "runtime/runners/blaze_runner/scheduler_interface.hpp"

namespace tt::runners::blaze {

std::unique_ptr<IDecodeScheduler> makeDecodeScheduler(
    const tt::config::BlazeConfig& config);

std::unique_ptr<IPrefillScheduler> makePrefillScheduler(
    const tt::config::BlazeConfig& config);

}  // namespace tt::runners::blaze
