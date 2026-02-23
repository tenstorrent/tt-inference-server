// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstddef>
#include <string>

namespace tt::config {

/**
 * Constants and enums used by settings, aligned with tt-media-server/config/constants.py.
 */

enum class ModelService {
    LLM,
    EMBEDDING,
};

/** String value for env MODEL_SERVICE (e.g. "llm", "embedding"). */
inline std::string to_string(ModelService s) {
    switch (s) {
        case ModelService::EMBEDDING:
            return "embedding";
        case ModelService::LLM:
        default:
            return "llm";
    }
}

/** Parse MODEL_SERVICE; empty or unknown -> LLM. */
inline ModelService model_service_from_string(const std::string& v) {
    if (v == "embedding") return ModelService::EMBEDDING;
    return ModelService::LLM;
}

enum class RunnerType {
    LLM_TEST,
};

/** String value for env MODEL_RUNNER (e.g. "llm_test"). */
inline std::string to_string(RunnerType r) {
    switch (r) {
        case RunnerType::LLM_TEST:
        default:
            return "llm_test";
    }
}

enum class SocketRole {
    NONE,
    SERVER,
    CLIENT,
};

/** String value for env SOCKET_ROLE (e.g. "server", "client"). */
inline std::string to_string(SocketRole r) {
    switch (r) {
        case SocketRole::SERVER:
            return "server";
        case SocketRole::CLIENT:
            return "client";
        case SocketRole::NONE:
        default:
            return "";
    }
}

/** Parse SOCKET_ROLE; empty or unknown -> NONE. */
inline SocketRole socket_role_from_string(const std::string& v) {
    if (v == "server") return SocketRole::SERVER;
    if (v == "client") return SocketRole::CLIENT;
    return SocketRole::NONE;
}

/** Parse MODEL_RUNNER; unknown -> LLM_TEST. */
inline RunnerType runner_type_from_string(const std::string& /*v*/) {
    return RunnerType::LLM_TEST;
}

/**
 * Default values when the corresponding environment variable is not set or empty.
 * Env overrides these when present.
 */
namespace defaults {
    constexpr const char* DEVICE_IDS = "(0)";
    constexpr const char* MODEL_SERVICE = "llm";
    constexpr size_t MAX_BATCH_SIZE = 1;
    constexpr unsigned MAX_BATCH_DELAY_TIME_MS = 5;
    constexpr const char* TT_PYTHON_PATH = "..";
    constexpr const char* SOCKET_ROLE = "";
    constexpr const char* SOCKET_HOST = "localhost";
    constexpr uint16_t SOCKET_PORT = 8000;
}

}  // namespace tt::config
