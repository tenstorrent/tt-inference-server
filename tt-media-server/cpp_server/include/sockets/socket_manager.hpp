// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <sstream>

namespace tt::sockets {

/**
 * @brief Serializable message wrapper for socket communication
 */
template<typename T>
struct SocketMessage {
    std::string message_type;
    T payload;

    template<class Archive>
    void serialize(Archive& ar) {
        ar(message_type, payload);
    }
};

/**
 * @brief Singleton socket manager for inter-server communication
 *
 * Supports both server (listening) and client (connecting) modes.
 * Can serialize and send objects using Cereal library.
 */
class SocketManager {
public:
    enum class Mode {
        SERVER,  // Listen for incoming connections
        CLIENT   // Connect to remote server
    };

    /**
     * @brief Get singleton instance
     */
    static SocketManager& getInstance();

    /**
     * @brief Initialize as server (listening mode)
     * @param port Port to listen on
     * @return true if successful
     */
    bool initializeAsServer(uint16_t port);

    /**
     * @brief Initialize as client (connecting mode)
     * @param host Remote host to connect to
     * @param port Remote port to connect to
     * @return true if successful
     */
    bool initializeAsClient(const std::string& host, uint16_t port);

    /**
     * @brief Send serializable object to connected peer
     * @param message_type Type identifier for the message
     * @param obj Object to send
     * @return true if successful
     */
    template<typename T>
    bool sendObject(const std::string& message_type, const T& obj);

    /**
     * @brief Register handler for incoming messages of specific type
     * @param message_type Type identifier to handle
     * @param handler Function to call when message is received
     */
    template<typename T>
    void registerHandler(const std::string& message_type,
                        std::function<void(const T&)> handler);

    /**
     * @brief Start the socket manager (begins listening/connecting)
     */
    void start();

    /**
     * @brief Stop the socket manager
     */
    void stop();

    /**
     * @brief Check if connected to peer
     */
    bool isConnected() const;

    /**
     * @brief Get connection status string
     */
    std::string getStatus() const;

    /**
     * @brief Set callback for connection lost events
     * @param callback Function to call when connection is lost
     */
    void setConnectionLostCallback(std::function<void()> callback);

    // Disable copy/move for singleton
    SocketManager(const SocketManager&) = delete;
    SocketManager& operator=(const SocketManager&) = delete;

private:
    SocketManager() = default;
    ~SocketManager();

    void serverLoop();
    void clientLoop();
    void messageLoop();
    void handleIncomingMessage(const std::vector<uint8_t>& data);
    bool sendRawData(const std::vector<uint8_t>& data);
    std::vector<uint8_t> receiveRawData();

    Mode mode_;
    std::string host_;
    uint16_t port_;

    int server_socket_ = -1;
    int client_socket_ = -1;
    int peer_socket_ = -1;  // Active connection socket

    std::atomic<bool> running_{false};
    std::atomic<bool> connected_{false};

    std::thread server_thread_;
    std::thread message_thread_;

    mutable std::mutex handlers_mutex_;
    std::map<std::string, std::function<void(const std::vector<uint8_t>&)>> handlers_;

    mutable std::mutex send_mutex_;

    std::function<void()> connection_lost_callback_;
};

// Template implementations
template<typename T>
bool SocketManager::sendObject(const std::string& message_type, const T& obj) {
    if (!connected_) {
        return false;
    }

    try {
        SocketMessage<T> message;
        message.message_type = message_type;
        message.payload = obj;

        std::ostringstream oss;
        {
            cereal::BinaryOutputArchive archive(oss);
            archive(message);
        }

        std::string serialized = oss.str();
        std::vector<uint8_t> data(serialized.begin(), serialized.end());

        return sendRawData(data);
    } catch (const std::exception& e) {
        std::cerr << "[SocketManager] Serialization error: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
void SocketManager::registerHandler(const std::string& message_type,
                                   std::function<void(const T&)> handler) {
    std::lock_guard<std::mutex> lock(handlers_mutex_);

    handlers_[message_type] = [handler](const std::vector<uint8_t>& data) {
        try {
            std::string serialized(data.begin(), data.end());
            std::istringstream iss(serialized);

            cereal::BinaryInputArchive archive(iss);
            SocketMessage<T> message;
            archive(message);

            handler(message.payload);
        } catch (const std::exception& e) {
            std::cerr << "[SocketManager] Deserialization error: " << e.what() << std::endl;
        }
    };
}

} // namespace tt::sockets
