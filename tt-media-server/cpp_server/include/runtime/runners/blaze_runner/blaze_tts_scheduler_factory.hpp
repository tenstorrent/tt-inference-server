// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>

#include "config/runner_config.hpp"
#include "runtime/runners/blaze_runner/tts_scheduler_interface.hpp"

namespace tt::runners::blaze {

std::unique_ptr<tts_scheduler::ITtsScheduler> makeTtsScheduler(
    const tt::config::TtsConfig& config);

std::unique_ptr<tts_scheduler::ITtsScheduler> makeMockTtsScheduler(
    const tt::config::TtsConfig& config);

}  // namespace tt::runners::blaze
