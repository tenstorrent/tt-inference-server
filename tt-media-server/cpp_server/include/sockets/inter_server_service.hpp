// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "sockets/socket_manager.hpp"
#include "sockets/socket_messages.hpp"
#include "config/settings.hpp"
#include <memory>
#include <string>
#include <functional>

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
     * @brief Task completion callback type
     */
    using TaskCallback = std::function<void(const std::string& task_id, const std::string& result, bool finished)>;

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
     * @param prompt Task prompt
     * @param max_tokens Maximum tokens to generate
     * @param temperature Sampling temperature
     * @param stop_sequences Stop sequences
     * @return true if sent successfully
     */
    bool forwardTask(const std::string& task_id,
                    const std::string& prompt,
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
     * @return true if sent successfully
     */
    bool sendTaskResult(const std::string& task_id,
                       const std::string& result,
                       bool finished,
                       int tokens_generated,
                       double processing_time_ms);

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
     * @param callback Function to call when task is received
     */
    void setTaskForwardCallback(TaskCallback callback);

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
    TaskCallback task_forward_callback_;
    TaskCallback task_result_callback_;
    HealthCallback health_check_callback_;
    bool enabled_ = false;
};

} // namespace tt::sockets
