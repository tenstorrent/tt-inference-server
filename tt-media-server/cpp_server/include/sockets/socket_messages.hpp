// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

namespace tt::sockets {

/**
 * @brief Prefill request message - sent from decode server to prefill server
 */
struct PrefillRequestMessage {
    std::string task_id;
    std::string prompt;
    std::vector<int64_t> token_ids;
    int max_tokens = 0;

    template<class Archive>
    void serialize(Archive& ar) {
        ar(task_id, prompt, token_ids, max_tokens);
    }
};

/**
 * @brief Prefill result message - sent from prefill server back to decode server
 *
 * Contains the first token and updated sequence for decode server to continue generation.
 */
struct PrefillResultMessage {
    std::string task_id;
    std::string generated_text;
    bool finished = false;
    int tokens_generated = 0;
    double processing_time_ms = 0.0;
    std::vector<int64_t> token_ids;
    int remaining_tokens = 0;

    template<class Archive>
    void serialize(Archive& ar) {
        ar(task_id, generated_text, finished, tokens_generated, processing_time_ms,
           token_ids, remaining_tokens);
    }
};

/**
 * @brief Health check message
 */
struct HealthCheckMessage {
    std::string server_id;
    double cpu_usage = 0.0;
    double memory_usage = 0.0;
    int active_tasks = 0;

    template<class Archive>
    void serialize(Archive& ar) {
        ar(server_id, cpu_usage, memory_usage, active_tasks);
    }
};

/**
 * @brief Load balancing info message
 */
struct LoadBalanceMessage {
    std::string server_id;
    int queue_size = 0;
    double avg_processing_time = 0.0;
    bool accepting_tasks = false;

    template<class Archive>
    void serialize(Archive& ar) {
        ar(server_id, queue_size, avg_processing_time, accepting_tasks);
    }
};

} // namespace tt::sockets
