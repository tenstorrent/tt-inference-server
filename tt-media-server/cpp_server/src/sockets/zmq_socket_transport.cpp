// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "sockets/zmq_socket_transport.hpp"

#include <cstdint>
#include <cstring>
#include <utility>
#include <zmq.hpp>

#include "utils/logger.hpp"

namespace tt::sockets {

namespace {
std::string makeMonitorEndpoint(const void* self) {
  return "inproc://zmq-monitor-" +
         std::to_string(reinterpret_cast<uintptr_t>(self));
}
}  // namespace

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
    setupMonitor();
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
    socket_->set(zmq::sockopt::reconnect_ivl,
                 static_cast<int>(reconnectInitialDelayMs_));
    socket_->set(zmq::sockopt::reconnect_ivl_max,
                 static_cast<int>(reconnectMaxDelayMs_));
    setupMonitor();
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

void ZmqSocketTransport::setupMonitor() {
  const std::string monitorAddr = makeMonitorEndpoint(this);
  int events =
      ZMQ_EVENT_CONNECTED | ZMQ_EVENT_ACCEPTED | ZMQ_EVENT_DISCONNECTED;
  if (zmq_socket_monitor(static_cast<void*>(*socket_), monitorAddr.c_str(),
                         events) != 0) {
    TT_LOG_WARN(
        "[ZmqSocketTransport] zmq_socket_monitor failed; connection events "
        "will be missed");
    return;
  }

  // libzmq drops events fired before a PAIR has attached to the inproc
  // endpoint, so block here until the monitor thread has connected its PAIR.
  monitorActive_ = true;
  std::promise<void> ready;
  auto fut = ready.get_future();
  monitorThread_ =
      std::thread(&ZmqSocketTransport::monitorLoop, this, std::move(ready));
  fut.wait();
}

void ZmqSocketTransport::start() {
  if (running_) return;
  running_ = true;
  // Do NOT reset `connected_` — the monitor thread (running since
  // initializeAsClient/Server) may already have observed and recorded a
  // CONNECTED/ACCEPTED event that fired between connect()/bind() and start().

  TT_LOG_INFO("[ZmqSocketTransport] Started ({})",
              mode_ == Mode::SERVER ? "server" : "client");
}

void ZmqSocketTransport::stop() {
  if (!running_ && !monitorActive_) return;
  running_ = false;
  monitorActive_ = false;
  connected_ = false;

  if (monitorThread_.joinable()) {
    monitorThread_.join();
  }

  std::lock_guard<std::mutex> lock(socketMutex_);
  if (socket_) {
    socket_->close();
    socket_.reset();
  }
  if (context_) {
    context_->close();
  }

  TT_LOG_INFO("[ZmqSocketTransport] Stopped");
}

void ZmqSocketTransport::monitorLoop(std::promise<void> ready) {
  zmq::socket_t monitorSocket(*context_, zmq::socket_type::pair);
  try {
    monitorSocket.set(zmq::sockopt::linger, 0);
    monitorSocket.set(zmq::sockopt::rcvtimeo, 200);
    monitorSocket.connect(makeMonitorEndpoint(this));
  } catch (const zmq::error_t& e) {
    ready.set_value();
    TT_LOG_ERROR("[ZmqSocketTransport] Monitor connect failed: {}", e.what());
    return;
  }
  ready.set_value();

  while (monitorActive_) {
    try {
      zmq::message_t eventMsg;
      auto eventResult = monitorSocket.recv(eventMsg, zmq::recv_flags::none);
      if (!eventResult.has_value()) continue;
      if (eventMsg.size() < sizeof(uint16_t)) continue;

      uint16_t eventId = 0;
      std::memcpy(&eventId, eventMsg.data(), sizeof(eventId));

      // Endpoint frame follows — drain it even if we don't use it.
      if (eventMsg.more()) {
        zmq::message_t addrMsg;
        (void)monitorSocket.recv(addrMsg, zmq::recv_flags::none);
      }

      if (eventId == ZMQ_EVENT_CONNECTED || eventId == ZMQ_EVENT_ACCEPTED) {
        bool wasConnected = connected_.exchange(true);
        if (!wasConnected) {
          TT_LOG_DEBUG("[ZmqSocketTransport] Peer connected ({})",
                       mode_ == Mode::SERVER ? "server" : "client");
        }
      } else if (eventId == ZMQ_EVENT_DISCONNECTED) {
        bool wasConnected = connected_.exchange(false);
        if (wasConnected) {
          TT_LOG_DEBUG("[ZmqSocketTransport] Peer disconnected ({})",
                       mode_ == Mode::SERVER ? "server" : "client");
          if (connectionLostCallback_) {
            connectionLostCallback_();
          }
        }
      }
    } catch (const zmq::error_t& e) {
      if (e.num() == ETERM || e.num() == EINTR) break;
      // Ignore EAGAIN from rcvtimeo; loop back and re-check running_.
    }
  }
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
  std::lock_guard<std::mutex> lock(socketMutex_);
  if (!running_ || !socket_) return false;

  try {
    return mode_ == Mode::SERVER ? sendAsRouter(data) : sendAsDealer(data);
  } catch (const zmq::error_t& e) {
    TT_LOG_ERROR("[ZmqSocketTransport] Send failed: {}", e.what());
    return false;
  }
}

bool ZmqSocketTransport::sendAsRouter(const std::vector<uint8_t>& data) {
  // ROUTER must prefix every outgoing message with the peer's identity.
  std::lock_guard<std::mutex> idLock(peerIdMutex_);
  if (peerId_.empty()) {
    TT_LOG_ERROR(
        "[ZmqSocketTransport] Cannot send — no peer identity known yet");
    return false;
  }
  zmq::message_t idFrame(peerId_.data(), peerId_.size());
  socket_->send(idFrame, zmq::send_flags::sndmore);

  zmq::message_t msg(data.data(), data.size());
  auto result = socket_->send(msg, zmq::send_flags::dontwait);
  return result.has_value();
}

bool ZmqSocketTransport::sendAsDealer(const std::vector<uint8_t>& data) {
  zmq::message_t msg(data.data(), data.size());
  auto result = socket_->send(msg, zmq::send_flags::dontwait);
  return result.has_value();
}

std::vector<uint8_t> ZmqSocketTransport::receiveRawData() {
  std::lock_guard<std::mutex> lock(socketMutex_);
  if (!running_ || !socket_) return {};

  try {
    return mode_ == Mode::SERVER ? receiveAsRouter() : receiveAsDealer();
  } catch (const zmq::error_t& e) {
    if (e.num() != EAGAIN) {
      TT_LOG_ERROR("[ZmqSocketTransport] Recv failed: {}", e.what());
    }
    return {};
  }
}

std::vector<uint8_t> ZmqSocketTransport::receiveAsRouter() {
  // ROUTER: first frame is peer identity — store it for future sends.
  // Connection state itself is tracked by the monitor thread.
  zmq::message_t identity;
  auto idResult = socket_->recv(identity, zmq::recv_flags::dontwait);
  if (!idResult.has_value()) return {};

  {
    std::lock_guard<std::mutex> idLock(peerIdMutex_);
    peerId_.assign(
        static_cast<uint8_t*>(identity.data()),
        static_cast<uint8_t*>(identity.data()) + identity.size());
  }

  if (!identity.more()) return {};

  zmq::message_t msg;
  auto result = socket_->recv(msg, zmq::recv_flags::dontwait);
  if (!result.has_value() || msg.size() == 0) return {};

  auto* ptr = static_cast<uint8_t*>(msg.data());
  return {ptr, ptr + msg.size()};
}

std::vector<uint8_t> ZmqSocketTransport::receiveAsDealer() {
  zmq::message_t msg;
  auto result = socket_->recv(msg, zmq::recv_flags::dontwait);
  if (!result.has_value() || msg.size() == 0) return {};

  auto* ptr = static_cast<uint8_t*>(msg.data());
  return {ptr, ptr + msg.size()};
}

void ZmqSocketTransport::setConnectionLostCallback(
    std::function<void()> callback) {
  connectionLostCallback_ = std::move(callback);
}

void ZmqSocketTransport::setReconnectBackoff(uint32_t initialDelayMs,
                                             uint32_t maxDelayMs) {
  reconnectInitialDelayMs_ = initialDelayMs;
  reconnectMaxDelayMs_ = maxDelayMs;
}

}  // namespace tt::sockets
