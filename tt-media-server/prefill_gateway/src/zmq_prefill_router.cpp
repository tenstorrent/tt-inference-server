// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/zmq_prefill_router.hpp"

#include <zmq.hpp>

#include "sockets/zmq_socket_options.hpp"
#include "utils/logger.hpp"

namespace tt::gateway {
namespace {

constexpr int ZMQ_RECEIVE_TIMEOUT_MS = 50;
constexpr auto IO_IDLE_WAIT = std::chrono::milliseconds(1);

std::vector<uint8_t> toBytes(const zmq::message_t& message) {
  auto* ptr = static_cast<const uint8_t*>(message.data());
  return {ptr, ptr + message.size()};
}

}  // namespace

class ZmqPrefillRouter::Impl {
 public:
  Impl() : context(tt::sockets::zmq_options::CONTEXT_IO_THREADS) {}

  zmq::context_t context;
  std::unique_ptr<zmq::socket_t> socket;
};

ZmqPrefillRouter::ZmqPrefillRouter() : impl_(std::make_unique<Impl>()) {}

ZmqPrefillRouter::~ZmqPrefillRouter() { stop(); }

bool ZmqPrefillRouter::start(const std::string& bindHost, uint16_t port) {
  if (running_) {
    return false;
  }

  endpoint_ = "tcp://" + bindHost + ":" + std::to_string(port);
  running_ = true;
  if (!startIoThread()) {
    running_ = false;
    return false;
  }
  return true;
}

void ZmqPrefillRouter::stop() {
  if (!running_) {
    return;
  }

  running_ = false;
  sendQueue.notifyStopped();

  if (io_thread_.joinable()) {
    io_thread_.request_stop();
    io_thread_.join();
  }

  TT_LOG_INFO("[ZmqPrefillRouter] Stopped");
}

std::string ZmqPrefillRouter::peerKey(const PeerIdentity& peerId) {
  return {reinterpret_cast<const char*>(peerId.data()), peerId.size()};
}

void ZmqPrefillRouter::rememberRegistration(const std::string& serverId,
                                            const PeerIdentity& peerId) {
  const std::string key = peerKey(peerId);
  std::lock_guard<std::mutex> lock(peer_mutex_);

  auto oldPeer = server_to_peer_.find(serverId);
  if (oldPeer != server_to_peer_.end()) {
    peer_to_server_.erase(peerKey(oldPeer->second));
  }

  server_to_peer_[serverId] = peerId;
  peer_to_server_[key] = serverId;
  last_seen_by_server_[serverId] = std::chrono::steady_clock::now();
}

std::optional<std::string> ZmqPrefillRouter::serverIdForPeer(
    const PeerIdentity& peerId) const {
  const std::string key = peerKey(peerId);
  std::lock_guard<std::mutex> lock(peer_mutex_);
  auto it = peer_to_server_.find(key);
  if (it == peer_to_server_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::vector<std::string> ZmqPrefillRouter::takeStaleServers(
    std::chrono::milliseconds timeout) {
  const auto now = std::chrono::steady_clock::now();
  std::vector<std::string> staleServers;

  std::lock_guard<std::mutex> lock(peer_mutex_);
  std::erase_if(last_seen_by_server_, [&](const auto& lastSeen) {
    if (now - lastSeen.second <= timeout) {
      return false;
    }

    staleServers.push_back(lastSeen.first);
    auto peerIt = server_to_peer_.find(lastSeen.first);
    if (peerIt != server_to_peer_.end()) {
      peer_to_server_.erase(peerKey(peerIt->second));
      server_to_peer_.erase(peerIt);
    }
    return true;
  });

  return staleServers;
}

std::optional<ZmqPrefillRouter::PeerIdentity> ZmqPrefillRouter::peerIdForServer(
    const std::string& serverId) const {
  std::lock_guard<std::mutex> lock(peer_mutex_);
  auto it = server_to_peer_.find(serverId);
  if (it == server_to_peer_.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool ZmqPrefillRouter::startIoThread() {
  std::promise<bool> initialized;
  auto fut = initialized.get_future();
  io_thread_ = std::jthread([this, initialized = std::move(initialized)](
                                std::stop_token stopToken) mutable {
    ioLoop(stopToken, std::move(initialized));
  });
  bool initializedOk = fut.get();
  if (!initializedOk && io_thread_.joinable()) {
    io_thread_.request_stop();
    io_thread_.join();
  }
  return initializedOk;
}

void ZmqPrefillRouter::ioLoop(std::stop_token stopToken,
                              std::promise<bool> initialized) {
  if (!initializeSocket()) {
    initialized.set_value(false);
    return;
  }

  initialized.set_value(true);

  while (running_ && !stopToken.stop_requested()) {
    const bool sent = processPendingSends();
    const bool received = receiveAvailableMessages();
    if (!sent && !received) {
      waitForIoWork();
    }
  }

  failPendingSends();
  if (impl_->socket) {
    impl_->socket->close();
    impl_->socket.reset();
  }
  impl_->context.close();
}

bool ZmqPrefillRouter::initializeSocket() {
  try {
    impl_->socket = std::make_unique<zmq::socket_t>(impl_->context,
                                                    zmq::socket_type::router);
    tt::sockets::zmq_options::applyRouterOptions(*impl_->socket,
                                                 ZMQ_RECEIVE_TIMEOUT_MS);
    impl_->socket->bind(endpoint_);
    TT_LOG_INFO("[ZmqPrefillRouter] Bound prefill ROUTER to {}", endpoint_);
    return true;
  } catch (const zmq::error_t& e) {
    TT_LOG_ERROR("[ZmqPrefillRouter] Failed to bind {}: {}", endpoint_,
                 e.what());
    impl_->socket.reset();
    return false;
  }
}

bool ZmqPrefillRouter::processPendingSends() {
  bool processed = false;

  while (true) {
    std::shared_ptr<SendRequest> request;
    if (!sendQueue.tryPop(request)) {
      return processed;
    }

    bool ok = false;
    try {
      zmq::message_t idFrame(request->peerKey.data(), request->peerKey.size());
      auto idResult = impl_->socket->send(idFrame, zmq::send_flags::sndmore);

      zmq::message_t msg(request->data.data(), request->data.size());
      auto msgResult = impl_->socket->send(msg, zmq::send_flags::dontwait);
      ok = idResult.has_value() && msgResult.has_value();
    } catch (const zmq::error_t& e) {
      TT_LOG_ERROR("[ZmqPrefillRouter] Send failed: {}", e.what());
    }
    request->result.set_value(ok);
    processed = true;
  }
}

bool ZmqPrefillRouter::receiveAvailableMessages() {
  bool received = false;
  try {
    while (true) {
      zmq::message_t identity;
      auto idResult = impl_->socket->recv(identity, zmq::recv_flags::dontwait);
      if (!idResult.has_value()) {
        return received;
      }
      if (!identity.more()) {
        continue;
      }

      zmq::message_t msg;
      auto msgResult = impl_->socket->recv(msg, zmq::recv_flags::dontwait);
      if (!msgResult.has_value() || msg.size() == 0) {
        continue;
      }

      handleIncomingMessage(toBytes(identity), toBytes(msg));
      received = true;
    }
  } catch (const zmq::error_t& e) {
    if (e.num() != EAGAIN) {
      TT_LOG_ERROR("[ZmqPrefillRouter] Receive failed: {}", e.what());
    }
  }
  return received;
}

void ZmqPrefillRouter::waitForIoWork() {
  sendQueue.waitForWork(IO_IDLE_WAIT, [this] { return !running_.load(); });
}

void ZmqPrefillRouter::failPendingSends() {
  while (true) {
    std::shared_ptr<SendRequest> request;
    if (!sendQueue.tryPop(request)) {
      return;
    }
    request->result.set_value(false);
  }
}

void ZmqPrefillRouter::handleIncomingMessage(const PeerIdentity& peerId,
                                             const std::vector<uint8_t>& data) {
  try {
    {
      const std::string key = peerKey(peerId);
      std::lock_guard<std::mutex> lock(peer_mutex_);
      auto serverIt = peer_to_server_.find(key);
      if (serverIt != peer_to_server_.end()) {
        last_seen_by_server_[serverIt->second] =
            std::chrono::steady_clock::now();
      }
    }

    std::string messageType = tt::sockets::wire::readMessageType(data);
    RawHandler handler;
    {
      std::lock_guard<std::mutex> lock(handlers_mutex_);
      auto it = handlers_.find(messageType);
      if (it == handlers_.end()) {
        TT_LOG_DEBUG("[ZmqPrefillRouter] No handler for message type: {}",
                     messageType);
        return;
      }
      handler = it->second;
    }

    handler(peerId, data);
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[ZmqPrefillRouter] Message handling error: {}", e.what());
  }
}

}  // namespace tt::gateway
