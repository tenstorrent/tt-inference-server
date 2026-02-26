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
 * @brief Task forwarding message - send tasks to remote server
 */
struct TaskForwardMessage {
    std::string task_id;
    std::string prompt;
    std::vector<int64_t> token_ids;
    int max_tokens;
    float temperature;
    std::vector<std::string> stop_sequences;

    template<class Archive>
    void serialize(Archive& ar) {
        ar(task_id, prompt, token_ids, max_tokens, temperature, stop_sequences);
    }
};

/**
 * @brief Task result message - receive results from remote server
 *
 * In prefill/decode split mode:
 * - Prefill server sends the first token along with the updated sequence
 * - Decode server continues generating remaining tokens using token_ids
 */
struct TaskResultMessage {
    std::string task_id;
    std::string generated_text;
    bool finished;
    int tokens_generated;
    double processing_time_ms;
    std::vector<int64_t> token_ids;
    int remaining_tokens;
    float temperature;
    std::vector<std::string> stop_sequences;

    template<class Archive>
    void serialize(Archive& ar) {
        ar(task_id, generated_text, finished, tokens_generated, processing_time_ms,
           token_ids, remaining_tokens, temperature, stop_sequences);
    }
};

/**
 * @brief Health check message
 */
struct HealthCheckMessage {
    std::string server_id;
    double cpu_usage;
    double memory_usage;
    int active_tasks;

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
    int queue_size;
    double avg_processing_time;
    bool accepting_tasks;

    template<class Archive>
    void serialize(Archive& ar) {
        ar(server_id, queue_size, avg_processing_time, accepting_tasks);
    }
};

} // namespace tt::sockets
