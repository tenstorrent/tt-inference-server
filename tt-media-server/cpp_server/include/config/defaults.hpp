// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstddef>
#include <cstdint>

namespace tt::config::defaults {

/**
 * Default values when the corresponding environment variable is not set or
 * empty. Environment variables override these when present.
 */

constexpr const char* DEVICE_IDS = "(0)";
constexpr const char* MODEL_SERVICE = "llm";
constexpr unsigned MAX_BATCH_DELAY_TIME_MS = 5;
constexpr const char* TT_PYTHON_PATH = "..";
constexpr const char* LLM_MODE = "regular";  // "regular", "prefill", "decode"
constexpr const char* SOCKET_HOST = "127.0.0.1";
constexpr uint16_t SOCKET_PORT = 9000;
constexpr const char* KV_MIGRATION_HOST = "127.0.0.1";
constexpr uint16_t KV_MIGRATION_PORT = 9001;
constexpr size_t MAX_QUEUE_SIZE = 1000;
constexpr const char* SCHEDULING_POLICY =
    "prefill_first";  // "prefill_first" or "max_occupancy"
constexpr const char* LLM_DEVICE_BACKEND =
    "llama";  // "mock", "mock_pipeline", "pipeline", "llama"
constexpr const bool ENABLE_ACCUMULATED_STREAMING = false;
constexpr size_t MAX_ACCUMULATED_TOKENS = 5;
constexpr size_t MAX_IN_FLIGHT_COUNT = 32;
constexpr size_t MAX_SESSIONS_COUNT = 64;
constexpr unsigned SESSION_EVICTION_RATE = 90;
constexpr size_t SESSION_EVICTION_COUNT = 10;

}  // namespace tt::config::defaults
