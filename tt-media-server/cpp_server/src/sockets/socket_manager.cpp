// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "sockets/socket_manager.hpp"
#include <iostream>
#include <cstring>
#include <fcntl.h>
#include <errno.h>

namespace tt::sockets {

SocketManager& SocketManager::getInstance() {
    static SocketManager instance;
    return instance;
}

SocketManager::~SocketManager() {
    stop();
}

bool SocketManager::initializeAsServer(uint16_t port) {
    mode_ = Mode::SERVER;
    port_ = port;

    server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket_ < 0) {
        std::cerr << "[SocketManager] Failed to create server socket: " << strerror(errno) << std::endl;
        return false;
    }

    // Set socket options
    int opt = 1;
    if (setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "[SocketManager] Failed to set SO_REUSEADDR: " << strerror(errno) << std::endl;
        close(server_socket_);
        return false;
    }

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(server_socket_, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "[SocketManager] Failed to bind to port " << port_ << ": " << strerror(errno) << std::endl;
        close(server_socket_);
        return false;
    }

    if (listen(server_socket_, 1) < 0) {
        std::cerr << "[SocketManager] Failed to listen: " << strerror(errno) << std::endl;
        close(server_socket_);
        return false;
    }

    std::cout << "[SocketManager] Server initialized on port " << port_ << std::endl;
    return true;
}

bool SocketManager::initializeAsClient(const std::string& host, uint16_t port) {
    mode_ = Mode::CLIENT;
    host_ = host;
    port_ = port;

    std::cout << "[SocketManager] Client initialized to connect to " << host_ << ":" << port_ << std::endl;
    return true;
}

void SocketManager::start() {
    if (running_) {
        return;
    }

    running_ = true;

    if (mode_ == Mode::SERVER) {
        server_thread_ = std::thread(&SocketManager::serverLoop, this);
    } else {
        server_thread_ = std::thread(&SocketManager::clientLoop, this);
    }

    message_thread_ = std::thread(&SocketManager::messageLoop, this);
}

void SocketManager::stop() {
    if (!running_) {
        return;
    }

    running_ = false;
    connected_ = false;

    if (peer_socket_ >= 0) {
        close(peer_socket_);
        peer_socket_ = -1;
    }

    if (client_socket_ >= 0) {
        close(client_socket_);
        client_socket_ = -1;
    }

    if (server_socket_ >= 0) {
        close(server_socket_);
        server_socket_ = -1;
    }

    if (server_thread_.joinable()) {
        server_thread_.join();
    }

    if (message_thread_.joinable()) {
        message_thread_.join();
    }

    std::cout << "[SocketManager] Stopped" << std::endl;
}

void SocketManager::serverLoop() {
    while (running_) {
        std::cout << "[SocketManager] Waiting for client connection..." << std::endl;

        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        int new_socket = accept(server_socket_, (struct sockaddr*)&client_addr, &client_len);
        if (new_socket < 0) {
            if (running_) {
                std::cerr << "[SocketManager] Accept failed: " << strerror(errno) << std::endl;
            }
            break;
        }

        // Set non-blocking mode
        int flags = fcntl(new_socket, F_GETFL, 0);
        fcntl(new_socket, F_SETFL, flags | O_NONBLOCK);

        peer_socket_ = new_socket;
        connected_ = true;

        std::cout << "[SocketManager] Client connected from "
                  << inet_ntoa(client_addr.sin_addr) << ":" << ntohs(client_addr.sin_port) << std::endl;

        // Wait until disconnected
        while (running_ && connected_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (peer_socket_ >= 0) {
            close(peer_socket_);
            peer_socket_ = -1;
        }
        connected_ = false;

        std::cout << "[SocketManager] Client disconnected" << std::endl;
    }
}

void SocketManager::clientLoop() {
    while (running_) {
        client_socket_ = socket(AF_INET, SOCK_STREAM, 0);
        if (client_socket_ < 0) {
            std::cerr << "[SocketManager] Failed to create client socket: " << strerror(errno) << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }

        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port_);

        if (inet_pton(AF_INET, host_.c_str(), &server_addr.sin_addr) <= 0) {
            std::cerr << "[SocketManager] Invalid address: " << host_ << std::endl;
            close(client_socket_);
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }

        std::cout << "[SocketManager] Attempting to connect to " << host_ << ":" << port_ << std::endl;

        if (connect(client_socket_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "[SocketManager] Connection failed: " << strerror(errno) << std::endl;
            close(client_socket_);
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }

        // Set non-blocking mode
        int flags = fcntl(client_socket_, F_GETFL, 0);
        fcntl(client_socket_, F_SETFL, flags | O_NONBLOCK);

        peer_socket_ = client_socket_;
        connected_ = true;

        std::cout << "[SocketManager] Connected to server" << std::endl;

        // Wait until disconnected
        while (running_ && connected_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (client_socket_ >= 0) {
            close(client_socket_);
            client_socket_ = -1;
        }
        peer_socket_ = -1;
        connected_ = false;

        std::cout << "[SocketManager] Disconnected from server" << std::endl;

        if (running_) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }
}

void SocketManager::messageLoop() {
    while (running_) {
        if (!connected_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        try {
            auto data = receiveRawData();
            if (!data.empty()) {
                handleIncomingMessage(data);
            }
        } catch (const std::exception& e) {
            std::cerr << "[SocketManager] Message loop error: " << e.what() << std::endl;
            connected_ = false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

bool SocketManager::sendRawData(const std::vector<uint8_t>& data) {
    if (!connected_ || peer_socket_ < 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(send_mutex_);

    // Send data size first
    uint32_t size = static_cast<uint32_t>(data.size());
    uint32_t net_size = htonl(size);

    ssize_t sent = send(peer_socket_, &net_size, sizeof(net_size), MSG_NOSIGNAL);
    if (sent != sizeof(net_size)) {
        connected_ = false;
        return false;
    }

    // Send actual data
    size_t total_sent = 0;
    while (total_sent < data.size()) {
        sent = send(peer_socket_, data.data() + total_sent, data.size() - total_sent, MSG_NOSIGNAL);
        if (sent <= 0) {
            connected_ = false;
            return false;
        }
        total_sent += sent;
    }

    return true;
}

std::vector<uint8_t> SocketManager::receiveRawData() {
    if (!connected_ || peer_socket_ < 0) {
        return {};
    }

    // Read data size first
    uint32_t net_size;
    ssize_t received = recv(peer_socket_, &net_size, sizeof(net_size), MSG_DONTWAIT);
    if (received <= 0) {
        if (received == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
            connected_ = false;
        }
        return {};
    }

    if (received != sizeof(net_size)) {
        connected_ = false;
        return {};
    }

    uint32_t size = ntohl(net_size);
    if (size == 0 || size > 1024*1024) {  // Max 1MB per message
        connected_ = false;
        return {};
    }

    // Read actual data
    std::vector<uint8_t> data(size);
    size_t total_received = 0;

    while (total_received < size) {
        received = recv(peer_socket_, data.data() + total_received, size - total_received, 0);
        if (received <= 0) {
            connected_ = false;
            return {};
        }
        total_received += received;
    }

    return data;
}

void SocketManager::handleIncomingMessage(const std::vector<uint8_t>& data) {
    try {
        // First, extract the message type
        std::string serialized(data.begin(), data.end());
        std::istringstream iss(serialized);

        cereal::BinaryInputArchive archive(iss);
        std::string message_type;
        archive(message_type);  // Use operator() instead of loadBinaryValue

        // Find handler for this message type
        std::lock_guard<std::mutex> lock(handlers_mutex_);
        auto it = handlers_.find(message_type);
        if (it != handlers_.end()) {
            it->second(data);
        } else {
            std::cout << "[SocketManager] No handler for message type: " << message_type << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[SocketManager] Message handling error: " << e.what() << std::endl;
    }
}

bool SocketManager::isConnected() const {
    return connected_;
}

std::string SocketManager::getStatus() const {
    if (!running_) {
        return "stopped";
    }

    if (connected_) {
        return mode_ == Mode::SERVER ? "server:connected" : "client:connected";
    }

    return mode_ == Mode::SERVER ? "server:waiting" : "client:connecting";
}

} // namespace tt::sockets
