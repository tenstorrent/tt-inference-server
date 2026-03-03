// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "sockets/inter_server_service.hpp"
#include <iostream>

namespace tt::sockets {

InterServerService::InterServerService()
    : socket_manager_(SocketManager::getInstance()) {
    setupMessageHandlers();
}

InterServerService::~InterServerService() {
    stop();
}

bool InterServerService::initializeFromConfig() {
    auto mode = tt::config::llm_mode();

    if (mode == tt::config::LLMMode::REGULAR) {
        std::cout << "[InterServerService] Socket communication disabled (regular mode)" << std::endl;
        enabled_ = false;
        return false;
    }

    auto host = tt::config::socket_host();
    auto port = tt::config::socket_port();

    bool success = false;

    if (mode == tt::config::LLMMode::DECODE_ONLY) {
        std::cout << "[InterServerService] Initializing as server on port " << port << std::endl;
        success = socket_manager_.initializeAsServer(port);
    } else if (mode == tt::config::LLMMode::PREFILL_ONLY) {
        std::cout << "[InterServerService] Initializing as client to " << host << ":" << port << std::endl;
        success = socket_manager_.initializeAsClient(host, port);
    }

    if (success) {
        enabled_ = true;
        std::cout << "[InterServerService] Socket communication initialized successfully" << std::endl;
    } else {
        enabled_ = false;
        std::cerr << "[InterServerService] Failed to initialize socket communication" << std::endl;
    }

    return success;
}

void InterServerService::start() {
    if (!enabled_) {
        return;
    }

    socket_manager_.start();
    std::cout << "[InterServerService] Started socket communication" << std::endl;
}

void InterServerService::stop() {
    if (!enabled_) {
        return;
    }

    socket_manager_.stop();
    std::cout << "[InterServerService] Stopped socket communication" << std::endl;
}

bool InterServerService::isEnabled() const {
    return enabled_;
}

bool InterServerService::forwardTask(const std::string& task_id,
                                    const std::string& prompt,
                                    const std::vector<int64_t>& token_ids,
                                    int max_tokens) {
    if (!enabled_) {
        return false;
    }

    TaskForwardMessage message;
    message.task_id = task_id;
    message.prompt = prompt;
    message.token_ids = token_ids;
    message.max_tokens = max_tokens;

    return socket_manager_.sendObject("task_forward", message);
}

bool InterServerService::sendTaskResult(const TaskResultMessage& message) {
    if (!enabled_) {
        return false;
    }

    return socket_manager_.sendObject("task_result", message);
}

bool InterServerService::sendHealthCheck(const std::string& server_id,
                                        double cpu_usage,
                                        double memory_usage,
                                        int active_tasks) {
    if (!enabled_) {
        return false;
    }

    HealthCheckMessage message;
    message.server_id = server_id;
    message.cpu_usage = cpu_usage;
    message.memory_usage = memory_usage;
    message.active_tasks = active_tasks;

    return socket_manager_.sendObject("health_check", message);
}

void InterServerService::onPrefillRequested(TaskForwardCallback callback) {
    task_forward_callback_ = callback;
}

void InterServerService::onPrefillComplete(TaskCallback callback) {
    task_result_callback_ = callback;
}

void InterServerService::setHealthCheckCallback(HealthCallback callback) {
    health_check_callback_ = callback;
}

void InterServerService::setConnectionLostCallback(std::function<void()> callback) {
    socket_manager_.setConnectionLostCallback(std::move(callback));
}

bool InterServerService::isConnected() const {
    return enabled_ && socket_manager_.isConnected();
}

std::string InterServerService::getStatus() const {
    if (!enabled_) {
        return "disabled";
    }
    return socket_manager_.getStatus();
}

void InterServerService::setupMessageHandlers() {
    // Handle incoming task forwards
    socket_manager_.registerHandler<TaskForwardMessage>("task_forward",
        [this](const TaskForwardMessage& message) {
            std::cout << "[InterServerService] Received task forward: " << message.task_id
                      << " (tokens: " << message.token_ids.size() << ")" << std::endl;
            if (task_forward_callback_) {
                task_forward_callback_(message);
            }
        });

    // Handle incoming task results
    socket_manager_.registerHandler<TaskResultMessage>("task_result",
        [this](const TaskResultMessage& message) {
            std::cout << "[InterServerService] Received task result: " << message.task_id
                      << " - text: '" << message.generated_text.substr(0, 50)
                      << "', remaining: " << message.remaining_tokens
                      << ", token_ids: " << message.token_ids.size() << std::endl;
            if (task_result_callback_) {
                task_result_callback_(message);
            }
        });

    // Handle incoming health checks
    socket_manager_.registerHandler<HealthCheckMessage>("health_check",
        [this](const HealthCheckMessage& message) {
            std::cout << "[InterServerService] Received health check from: " << message.server_id
                      << " (CPU: " << message.cpu_usage << "%, Memory: " << message.memory_usage
                      << "%, Tasks: " << message.active_tasks << ")" << std::endl;
            if (health_check_callback_) {
                health_check_callback_(message.server_id, message.cpu_usage,
                                     message.memory_usage, message.active_tasks);
            }
        });
}

} // namespace tt::sockets
