// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "sockets/socket_manager.hpp"
#include "sockets/socket_messages.hpp"
#include "config/settings.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <functional>
#include <vector>

namespace tt::sockets {

/**
 * @brief Service for managing inter-server communication
 *
 * Handles task forwarding, load balancing, and health checks
 * between multiple cpp_server instances. Initializes based on
 * configuration settings.
 */
class InterServerService {
public:
    /**
     * @brief Task result callback type (extended for prefill/decode split)
     */
    using TaskCallback = std::function<void(const TaskResultMessage& result)>;

    /**
     * @brief Task forward callback type (includes token_ids for pre-tokenized prompts)
     */
    using TaskForwardCallback = std::function<void(const TaskForwardMessage& message)>;

    /**
     * @brief Health info callback type
     */
    using HealthCallback = std::function<void(const std::string& server_id, double cpu, double memory, int active_tasks)>;

    InterServerService();
    ~InterServerService();

    /**
     * @brief Initialize based on configuration settings
     * @return true if socket communication is enabled and initialized
     */
    bool initializeFromConfig();

    /**
     * @brief Start the inter-server communication (if enabled)
     */
    void start();

    /**
     * @brief Stop the inter-server communication
     */
    void stop();

    /**
     * @brief Check if socket communication is enabled
     */
    bool isEnabled() const;

    /**
     * @brief Forward a task to the connected server
     * @param task_id Unique task identifier
     * @param prompt Task prompt (text)
     * @param token_ids Pre-tokenized prompt token IDs
     * @param max_tokens Maximum tokens to generate
     * @param temperature Sampling temperature
     * @param stop_sequences Stop sequences
     * @return true if sent successfully
     */
    bool forwardTask(const std::string& task_id,
                    const std::string& prompt,
                    const std::vector<int64_t>& token_ids,
                    int max_tokens = 100,
                    float temperature = 0.7f,
                    const std::vector<std::string>& stop_sequences = {});

    /**
     * @brief Send task result to connected server
     * @param task_id Task identifier
     * @param result Generated text
     * @param finished Whether task is complete
     * @param tokens_generated Number of tokens generated
     * @param processing_time_ms Processing time in milliseconds
     * @param token_ids Updated token sequence (prompt + generated tokens) for decode continuation
     * @param remaining_tokens Remaining tokens to generate
     * @param temperature Sampling temperature for continuation
     * @param stop_sequences Stop sequences for continuation
     * @return true if sent successfully
     */
    bool sendTaskResult(const std::string& task_id,
                       const std::string& result,
                       bool finished,
                       int tokens_generated,
                       double processing_time_ms,
                       const std::vector<int64_t>& token_ids = {},
                       int remaining_tokens = 0,
                       float temperature = 0.7f,
                       const std::vector<std::string>& stop_sequences = {});

    /**
     * @brief Send health check information
     * @param server_id This server's identifier
     * @param cpu_usage CPU usage percentage
     * @param memory_usage Memory usage percentage
     * @param active_tasks Number of active tasks
     * @return true if sent successfully
     */
    bool sendHealthCheck(const std::string& server_id,
                        double cpu_usage,
                        double memory_usage,
                        int active_tasks);

    /**
     * @brief Set callback for received task forwards
     * @param callback Function to call when task forward message is received
     */
    void setTaskForwardCallback(TaskForwardCallback callback);

    /**
     * @brief Set callback for received task results
     * @param callback Function to call when result is received
     */
    void setTaskResultCallback(TaskCallback callback);

    /**
     * @brief Set callback for received health checks
     * @param callback Function to call when health info is received
     */
    void setHealthCheckCallback(HealthCallback callback);

    /**
     * @brief Set callback for connection lost events
     * @param callback Function to call when connection is lost
     */
    void setConnectionLostCallback(std::function<void()> callback);

    /**
     * @brief Check if connected to peer server
     */
    bool isConnected() const;

    /**
     * @brief Get connection status
     */
    std::string getStatus() const;

private:
    void setupMessageHandlers();

    SocketManager& socket_manager_;
    TaskForwardCallback task_forward_callback_;
    TaskCallback task_result_callback_;
    HealthCallback health_check_callback_;
    bool enabled_ = false;
};

} // namespace tt::sockets
