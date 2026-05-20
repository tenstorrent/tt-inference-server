// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/zmq_prefill_router.hpp"

#include <zmq.hpp>

#include "utils/logger.hpp"

namespace tt::gateway {
namespace {

constexpr int ZMQ_CONTEXT_IO_THREADS = 1;
constexpr int ZMQ_RECEIVE_TIMEOUT_MS = 50;
constexpr auto IO_IDLE_WAIT = std::chrono::milliseconds(1);

std::vector<uint8_t> toBytes(const zmq::message_t& message) {
  auto* ptr = static_cast<const uint8_t*>(message.data());
  return {ptr, ptr + message.size()};
}

}  // namespace

class ZmqPrefillRouter::Impl {
 public:
  Impl() : context(ZMQ_CONTEXT_IO_THREADS) {}

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
  send_cv_.notify_all();

  if (io_thread_.joinable()) {
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
  for (auto it = last_seen_by_server_.begin();
       it != last_seen_by_server_.end();) {
    if (now - it->second <= timeout) {
      ++it;
      continue;
    }

    staleServers.push_back(it->first);
    auto peerIt = server_to_peer_.find(it->first);
    if (peerIt != server_to_peer_.end()) {
      peer_to_server_.erase(peerKey(peerIt->second));
      server_to_peer_.erase(peerIt);
    }
    it = last_seen_by_server_.erase(it);
  }

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
  io_thread_ =
      std::thread(&ZmqPrefillRouter::ioLoop, this, std::move(initialized));
  bool initializedOk = fut.get();
  if (!initializedOk && io_thread_.joinable()) {
    io_thread_.join();
  }
  return initializedOk;
}

void ZmqPrefillRouter::ioLoop(std::promise<bool> initialized) {
  if (!initializeSocket()) {
    initialized.set_value(false);
    return;
  }

  initialized.set_value(true);

  while (running_) {
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
    impl_->socket->set(zmq::sockopt::linger, 0);
    impl_->socket->set(zmq::sockopt::rcvtimeo, ZMQ_RECEIVE_TIMEOUT_MS);
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
    {
      std::lock_guard<std::mutex> lock(send_mutex_);
      if (pending_sends_.empty()) {
        return processed;
      }
      request = std::move(pending_sends_.front());
      pending_sends_.pop_front();
    }

    bool ok = false;
    try {
      zmq::message_t idFrame(request->peerKey.data(), request->peerKey.size());
      impl_->socket->send(idFrame, zmq::send_flags::sndmore);

      zmq::message_t msg(request->data.data(), request->data.size());
      ok = impl_->socket->send(msg, zmq::send_flags::dontwait).has_value();
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
  std::unique_lock<std::mutex> lock(send_mutex_);
  send_cv_.wait_for(lock, IO_IDLE_WAIT,
                    [this] { return !pending_sends_.empty() || !running_; });
}

void ZmqPrefillRouter::failPendingSends() {
  while (true) {
    std::shared_ptr<SendRequest> request;
    {
      std::lock_guard<std::mutex> lock(send_mutex_);
      if (pending_sends_.empty()) {
        return;
      }
      request = std::move(pending_sends_.front());
      pending_sends_.pop_front();
    }
    request->result.set_value(false);
  }
}

void ZmqPrefillRouter::handleIncomingMessage(const PeerIdentity& peerId,
                                             const std::vector<uint8_t>& data) {
  try {
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
