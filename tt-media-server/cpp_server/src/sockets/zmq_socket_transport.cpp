// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "sockets/zmq_socket_transport.hpp"

#include <chrono>
#include <thread>
#include <zmq.hpp>

#include "utils/logger.hpp"

namespace tt::sockets {

ZmqSocketTransport::ZmqSocketTransport()
    : context_(std::make_unique<zmq::context_t>(1)) {}

ZmqSocketTransport::~ZmqSocketTransport() { stop(); }

bool ZmqSocketTransport::initializeAsServer(uint16_t port) {
  mode_ = Mode::SERVER;
  endpoint_ = "tcp://*:" + std::to_string(port);

  try {
    socket_ =
        std::make_unique<zmq::socket_t>(*context_, zmq::socket_type::router);
    socket_->set(zmq::sockopt::linger, 0);
    socket_->set(zmq::sockopt::rcvtimeo, 100);  // 100ms poll timeout
    socket_->bind(endpoint_);
    TT_LOG_INFO("[ZmqSocketTransport] Server bound to {}", endpoint_);
    return true;
  } catch (const zmq::error_t& e) {
    TT_LOG_ERROR("[ZmqSocketTransport] Failed to bind to {}: {}", endpoint_,
                 e.what());
    socket_.reset();
    return false;
  }
}

bool ZmqSocketTransport::initializeAsClient(const std::string& host,
                                            uint16_t port) {
  mode_ = Mode::CLIENT;
  endpoint_ = "tcp://" + host + ":" + std::to_string(port);

  try {
    socket_ =
        std::make_unique<zmq::socket_t>(*context_, zmq::socket_type::dealer);
    socket_->set(zmq::sockopt::linger, 0);
    socket_->set(zmq::sockopt::rcvtimeo, 100);
    socket_->set(zmq::sockopt::reconnect_ivl, 1000);      // 1s reconnect
    socket_->set(zmq::sockopt::reconnect_ivl_max, 5000);  // max 5s backoff
    socket_->connect(endpoint_);
    TT_LOG_INFO("[ZmqSocketTransport] Client connecting to {}", endpoint_);
    return true;
  } catch (const zmq::error_t& e) {
    TT_LOG_ERROR("[ZmqSocketTransport] Failed to connect to {}: {}", endpoint_,
                 e.what());
    socket_.reset();
    return false;
  }
}

void ZmqSocketTransport::start() {
  if (running_) return;
  running_ = true;

  if (mode_ == Mode::CLIENT) {
    // DEALER: send an empty "hello" frame so the ROUTER learns our identity.
    std::lock_guard<std::mutex> lock(sendMutex_);
    try {
      zmq::message_t hello(0);
      socket_->send(hello, zmq::send_flags::dontwait);
    } catch (...) {
    }
    connected_ = true;  // Client considers itself connected once socket is up.
  } else {
    // SERVER (ROUTER): we are NOT connected until we receive the first message
    // from a peer, which tells us the peer identity.
    connected_ = false;
  }

  TT_LOG_INFO("[ZmqSocketTransport] Started ({})",
              mode_ == Mode::SERVER ? "server" : "client");
}

void ZmqSocketTransport::stop() {
  if (!running_) return;
  running_ = false;
  connected_ = false;

  if (socket_) {
    socket_->close();
    socket_.reset();
  }
  if (context_) {
    context_->close();
  }

  TT_LOG_INFO("[ZmqSocketTransport] Stopped");
}

bool ZmqSocketTransport::isConnected() const { return connected_; }

std::string ZmqSocketTransport::getStatus() const {
  if (!running_) return "stopped";
  if (connected_) {
    return mode_ == Mode::SERVER ? "server:connected" : "client:connected";
  }
  return mode_ == Mode::SERVER ? "server:waiting" : "client:connecting";
}

bool ZmqSocketTransport::sendRawData(const std::vector<uint8_t>& data) {
  if (!running_ || !socket_) return false;

  std::lock_guard<std::mutex> lock(sendMutex_);
  try {
    if (mode_ == Mode::SERVER) {
      // ROUTER must prefix every outgoing message with the peer's identity.
      std::lock_guard<std::mutex> idLock(peerIdMutex_);
      if (peerId_.empty()) {
        TT_LOG_ERROR(
            "[ZmqSocketTransport] Cannot send — no peer identity known yet");
        return false;
      }
      zmq::message_t idFrame(peerId_.data(), peerId_.size());
      socket_->send(idFrame, zmq::send_flags::sndmore);
    }
    zmq::message_t msg(data.data(), data.size());
    auto result = socket_->send(msg, zmq::send_flags::dontwait);
    return result.has_value();
  } catch (const zmq::error_t& e) {
    TT_LOG_ERROR("[ZmqSocketTransport] Send failed: {}", e.what());
    return false;
  }
}

std::vector<uint8_t> ZmqSocketTransport::receiveRawData() {
  if (!running_ || !socket_) return {};

  try {
    if (mode_ == Mode::SERVER) {
      // ROUTER: first frame is peer identity — store it for future sends.
      zmq::message_t identity;
      auto idResult = socket_->recv(identity, zmq::recv_flags::dontwait);
      if (!idResult.has_value()) return {};

      // Store peer identity (update on every receive to stay current).
      {
        std::lock_guard<std::mutex> idLock(peerIdMutex_);
        peerId_.assign(
            static_cast<uint8_t*>(identity.data()),
            static_cast<uint8_t*>(identity.data()) + identity.size());
        if (!connected_) {
          connected_ = true;
          TT_LOG_INFO("[ZmqSocketTransport] Peer connected (identity size={})",
                      peerId_.size());
        }
      }

      if (!identity.more())
        return {};  // Peer sent only identity (hello frame).
    }

    zmq::message_t msg;
    auto result = socket_->recv(msg, zmq::recv_flags::dontwait);
    if (!result.has_value()) return {};

    // Skip empty "hello" frames from client.
    if (msg.size() == 0) return {};

    auto* ptr = static_cast<uint8_t*>(msg.data());
    return {ptr, ptr + msg.size()};
  } catch (const zmq::error_t& e) {
    if (e.num() != EAGAIN) {
      TT_LOG_ERROR("[ZmqSocketTransport] Recv failed: {}", e.what());
    }
    return {};
  }
}

void ZmqSocketTransport::setConnectionLostCallback(
    std::function<void()> callback) {
  connectionLostCallback_ = std::move(callback);
}

}  // namespace tt::sockets
