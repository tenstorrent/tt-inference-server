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

ZmqPrefillRouter::ZmqPrefillRouter() : impl(std::make_unique<Impl>()) {}

ZmqPrefillRouter::~ZmqPrefillRouter() { stop(); }

bool ZmqPrefillRouter::start(const std::string& bindHost, uint16_t port) {
  if (running) {
    return false;
  }

  endpoint = "tcp://" + bindHost + ":" + std::to_string(port);
  running = true;
  if (!startIoThread()) {
    running = false;
    return false;
  }
  return true;
}

void ZmqPrefillRouter::stop() {
  if (!running) {
    return;
  }

  running = false;
  sendQueue.notifyStopped();

  if (ioThread.joinable()) {
    ioThread.request_stop();
    ioThread.join();
  }

  TT_LOG_INFO("[ZmqPrefillRouter] Stopped");
}

std::string ZmqPrefillRouter::peerKey(const PeerIdentity& peerId) {
  return {reinterpret_cast<const char*>(peerId.data()), peerId.size()};
}

void ZmqPrefillRouter::rememberRegistration(const std::string& serverId,
                                            const PeerIdentity& peerId) {
  const std::string key = peerKey(peerId);
  std::lock_guard<std::mutex> lock(peerMutex);

  auto oldPeer = serverToPeer.find(serverId);
  if (oldPeer != serverToPeer.end()) {
    peerToServer.erase(peerKey(oldPeer->second));
  }

  serverToPeer[serverId] = peerId;
  peerToServer[key] = serverId;
  lastSeenByServer[serverId] = std::chrono::steady_clock::now();
}

std::optional<std::string> ZmqPrefillRouter::serverIdForPeer(
    const PeerIdentity& peerId) const {
  const std::string key = peerKey(peerId);
  std::lock_guard<std::mutex> lock(peerMutex);
  auto it = peerToServer.find(key);
  if (it == peerToServer.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::vector<std::string> ZmqPrefillRouter::takeStaleServers(
    std::chrono::milliseconds timeout) {
  const auto now = std::chrono::steady_clock::now();
  std::vector<std::string> staleServers;

  std::lock_guard<std::mutex> lock(peerMutex);
  std::erase_if(lastSeenByServer, [&](const auto& lastSeen) {
    if (now - lastSeen.second <= timeout) {
      return false;
    }

    staleServers.push_back(lastSeen.first);
    auto peerIt = serverToPeer.find(lastSeen.first);
    if (peerIt != serverToPeer.end()) {
      peerToServer.erase(peerKey(peerIt->second));
      serverToPeer.erase(peerIt);
    }
    return true;
  });

  return staleServers;
}

std::optional<ZmqPrefillRouter::PeerIdentity> ZmqPrefillRouter::peerIdForServer(
    const std::string& serverId) const {
  std::lock_guard<std::mutex> lock(peerMutex);
  auto it = serverToPeer.find(serverId);
  if (it == serverToPeer.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool ZmqPrefillRouter::startIoThread() {
  std::promise<bool> initialized;
  auto fut = initialized.get_future();
  ioThread = std::jthread([this, initialized = std::move(initialized)](
                              std::stop_token stopToken) mutable {
    ioLoop(stopToken, std::move(initialized));
  });
  bool initializedOk = fut.get();
  if (!initializedOk && ioThread.joinable()) {
    ioThread.request_stop();
    ioThread.join();
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

  while (running && !stopToken.stop_requested()) {
    const bool sent = processPendingSends();
    const bool received = receiveAvailableMessages();
    if (!sent && !received) {
      waitForIoWork();
    }
  }

  failPendingSends();
  if (impl->socket) {
    impl->socket->close();
    impl->socket.reset();
  }
  impl->context.close();
}

bool ZmqPrefillRouter::initializeSocket() {
  try {
    impl->socket = std::make_unique<zmq::socket_t>(impl->context,
                                                   zmq::socket_type::router);
    tt::sockets::zmq_options::applyRouterOptions(*impl->socket,
                                                 ZMQ_RECEIVE_TIMEOUT_MS);
    impl->socket->bind(endpoint);
    TT_LOG_INFO("[ZmqPrefillRouter] Bound prefill ROUTER to {}", endpoint);
    return true;
  } catch (const zmq::error_t& e) {
    TT_LOG_ERROR("[ZmqPrefillRouter] Failed to bind {}: {}", endpoint,
                 e.what());
    impl->socket.reset();
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
      auto idResult = impl->socket->send(idFrame, zmq::send_flags::sndmore);

      zmq::message_t msg(request->data.data(), request->data.size());
      auto msgResult = impl->socket->send(msg, zmq::send_flags::dontwait);
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
      auto idResult = impl->socket->recv(identity, zmq::recv_flags::dontwait);
      if (!idResult.has_value()) {
        return received;
      }
      if (!identity.more()) {
        continue;
      }

      zmq::message_t msg;
      auto msgResult = impl->socket->recv(msg, zmq::recv_flags::dontwait);
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
  sendQueue.waitForWork(IO_IDLE_WAIT, [this] { return !running.load(); });
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
      std::lock_guard<std::mutex> lock(peerMutex);
      auto serverIt = peerToServer.find(key);
      if (serverIt != peerToServer.end()) {
        lastSeenByServer[serverIt->second] = std::chrono::steady_clock::now();
      }
    }

    std::string messageType = tt::sockets::wire::readMessageType(data);
    RawHandler handler;
    {
      std::lock_guard<std::mutex> lock(handlersMutex);
      auto it = handlers.find(messageType);
      if (it == handlers.end()) {
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
