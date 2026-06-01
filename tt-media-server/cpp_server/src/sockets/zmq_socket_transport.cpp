// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "sockets/zmq_socket_transport.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <zmq.hpp>

#include "sockets/zmq_socket_options.hpp"
#include "utils/logger.hpp"

namespace tt::sockets {

namespace {
constexpr int ZMQ_MONITOR_TIMEOUT_MS = 200;
constexpr auto IO_IDLE_WAIT = std::chrono::milliseconds(1);

std::string makeMonitorEndpoint(const void* self) {
  return "inproc://zmq-monitor-" +
         std::to_string(reinterpret_cast<uintptr_t>(self));
}
}  // namespace

ZmqSocketTransport::ZmqSocketTransport()
    : SocketTransportState(std::chrono::seconds(1), std::chrono::seconds(5)),
      context_(
          std::make_unique<zmq::context_t>(zmq_options::CONTEXT_IO_THREADS)) {}

ZmqSocketTransport::~ZmqSocketTransport() { stop(); }

bool ZmqSocketTransport::initializeAsServer(uint16_t port) {
  mode_ = Mode::SERVER;
  endpoint_ = "tcp://*:" + std::to_string(port);
  return startIoThread();
}

bool ZmqSocketTransport::initializeAsClient(const std::string& host,
                                            uint16_t port) {
  mode_ = Mode::CLIENT;
  endpoint_ = "tcp://" + host + ":" + std::to_string(port);
  return startIoThread();
}

bool ZmqSocketTransport::startIoThread() {
  if (ioActive_) return false;
  ioActive_ = true;

  std::promise<bool> initialized;
  auto fut = initialized.get_future();
  ioThread_ = std::jthread([this, initialized = std::move(initialized)](
                               std::stop_token stopToken) mutable {
    ioLoop(stopToken, std::move(initialized));
  });

  bool initializedOk = fut.get();
  if (!initializedOk && ioThread_.joinable()) {
    ioThread_.request_stop();
    ioThread_.join();
  }
  return initializedOk;
}

void ZmqSocketTransport::ioLoop(std::stop_token stopToken,
                                std::promise<bool> initialized) {
  if (!initializeSocket()) {
    initialized.set_value(false);
    return;
  }

  initialized.set_value(true);

  while (ioActive_ && !stopToken.stop_requested()) {
    const bool sent = processPendingSends();
    const bool received = receiveAvailableMessages();

    if (!sent && !received) {
      waitForIoWork();
    }
  }

  failPendingSends();
  if (socket_) {
    socket_->close();
    socket_.reset();
  }
}

bool ZmqSocketTransport::initializeSocket() {
  try {
    socket_ = std::make_unique<zmq::socket_t>(
        *context_, mode_ == Mode::SERVER ? zmq::socket_type::router
                                         : zmq::socket_type::dealer);
    zmq_options::applyCommonOptions(*socket_);
    if (mode_ == Mode::CLIENT) {
      socket_->set(zmq::sockopt::reconnect_ivl,
                   static_cast<int>(reconnectInitialDelay_.count()));
      socket_->set(zmq::sockopt::reconnect_ivl_max,
                   static_cast<int>(reconnectMaxDelay_.count()));
    }
    setupMonitor();
    if (mode_ == Mode::SERVER) {
      socket_->bind(endpoint_);
      TT_LOG_INFO("[ZmqSocketTransport] Server bound to {}", endpoint_);
    } else {
      socket_->connect(endpoint_);
      TT_LOG_INFO("[ZmqSocketTransport] Client connecting to {}", endpoint_);
    }
    return true;
  } catch (const zmq::error_t& e) {
    TT_LOG_ERROR("[ZmqSocketTransport] Failed to initialize {}: {}", endpoint_,
                 e.what());
    socket_.reset();
    ioActive_ = false;
    monitorActive_ = false;
    if (monitorThread_.joinable()) {
      monitorThread_.request_stop();
      monitorThread_.join();
    }
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
  monitorThread_ = std::jthread(
      [this, ready = std::move(ready)](std::stop_token stopToken) mutable {
        monitorLoop(stopToken, std::move(ready));
      });
  fut.wait();
}

void ZmqSocketTransport::start() {
  if (running_) return;
  running_ = true;
  // Do NOT reset `connected_` — the monitor thread (running since
  // initializeAsClient/Server) may already have observed and recorded a
  // CONNECTED/ACCEPTED event that fired between connect()/bind() and start().

  TT_LOG_INFO("[ZmqSocketTransport] Started ({})", modeName());
}

void ZmqSocketTransport::stop() {
  if (!running_ && !ioActive_ && !monitorActive_) return;
  running_ = false;
  ioActive_ = false;
  monitorActive_ = false;
  connected_ = false;
  sendQueue.notifyStopped();

  if (ioThread_.joinable()) {
    ioThread_.request_stop();
    ioThread_.join();
  }

  if (monitorThread_.joinable()) {
    monitorThread_.request_stop();
    monitorThread_.join();
  }

  if (context_) {
    context_->close();
  }

  TT_LOG_INFO("[ZmqSocketTransport] Stopped");
}

void ZmqSocketTransport::monitorLoop(std::stop_token stopToken,
                                     std::promise<void> ready) {
  zmq::socket_t monitorSocket(*context_, zmq::socket_type::pair);
  try {
    monitorSocket.set(zmq::sockopt::linger, 0);
    monitorSocket.set(zmq::sockopt::rcvtimeo, ZMQ_MONITOR_TIMEOUT_MS);
    monitorSocket.connect(makeMonitorEndpoint(this));
  } catch (const zmq::error_t& e) {
    ready.set_value();
    TT_LOG_ERROR("[ZmqSocketTransport] Monitor connect failed: {}", e.what());
    return;
  }
  ready.set_value();

  while (monitorActive_ && !stopToken.stop_requested()) {
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
          TT_LOG_DEBUG("[ZmqSocketTransport] Peer connected ({})", modeName());
          notifyConnectionEstablished();
        }
      } else if (eventId == ZMQ_EVENT_DISCONNECTED) {
        bool wasConnected = connected_.exchange(false);
        if (wasConnected) {
          TT_LOG_DEBUG("[ZmqSocketTransport] Peer disconnected ({})",
                       modeName());
          notifyConnectionLost();
        }
      }
    } catch (const zmq::error_t& e) {
      if (e.num() == ETERM || e.num() == EINTR) break;
      // Ignore EAGAIN from rcvtimeo; loop back and re-check running_.
    }
  }
}

bool ZmqSocketTransport::isConnected() const { return isConnectedState(); }

std::string ZmqSocketTransport::getStatus() const { return getStatusString(); }

bool ZmqSocketTransport::sendRawData(std::span<const uint8_t> data) {
  if (!running_ || !ioActive_) return false;

  auto request = std::make_shared<SendRequest>();
  request->data.assign(data.begin(), data.end());
  auto result = request->result.get_future();

  if (!sendQueue.pushIf(std::move(request),
                        [this] { return running_.load() && ioActive_.load(); })) {
    return false;
  }

  try {
    return result.get();
  } catch (const std::future_error& e) {
    TT_LOG_ERROR("[ZmqSocketTransport] Send result failed: {}", e.what());
    return false;
  }
}

bool ZmqSocketTransport::processPendingSends() {
  bool processed = false;

  while (true) {
    std::shared_ptr<SendRequest> request;
    if (!sendQueue.tryPop(request)) {
      return processed;
    }

    bool ok = false;
    try {
      ok = running_ && (mode_ == Mode::SERVER ? sendAsRouter(request->data)
                                              : sendAsDealer(request->data));
    } catch (const zmq::error_t& e) {
      TT_LOG_ERROR("[ZmqSocketTransport] Send failed: {}", e.what());
    }
    request->result.set_value(ok);
    processed = true;
  }
}

bool ZmqSocketTransport::receiveAvailableMessages() {
  bool received = false;
  try {
    while (true) {
      auto data = mode_ == Mode::SERVER ? receiveAsRouter() : receiveAsDealer();
      if (data.empty()) break;
      enqueueReceivedMessage(std::move(data));
      received = true;
    }
  } catch (const zmq::error_t& e) {
    if (e.num() != EAGAIN) {
      TT_LOG_ERROR("[ZmqSocketTransport] Recv failed: {}", e.what());
    }
  }
  return received;
}

void ZmqSocketTransport::waitForIoWork() {
  sendQueue.waitForWork(IO_IDLE_WAIT, [this] { return !ioActive_.load(); });
}

void ZmqSocketTransport::failPendingSends() {
  while (true) {
    std::shared_ptr<SendRequest> request;
    if (!sendQueue.tryPop(request)) {
      return;
    }
    request->result.set_value(false);
  }
}

bool ZmqSocketTransport::sendAsRouter(const std::vector<uint8_t>& data) {
  // ROUTER must prefix every outgoing message with the peer's identity.
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
  std::lock_guard<std::mutex> lock(receiveMutex_);
  if (receivedMessages_.empty()) {
    return {};
  }

  auto data = std::move(receivedMessages_.front());
  receivedMessages_.pop_front();
  return data;
}

void ZmqSocketTransport::enqueueReceivedMessage(std::vector<uint8_t> data) {
  std::lock_guard<std::mutex> lock(receiveMutex_);
  receivedMessages_.push_back(std::move(data));
}

std::vector<uint8_t> ZmqSocketTransport::receiveAsRouter() {
  // ROUTER: first frame is peer identity — store it for future sends.
  // Connection state itself is tracked by the monitor thread.
  zmq::message_t identity;
  auto idResult = socket_->recv(identity, zmq::recv_flags::dontwait);
  if (!idResult.has_value()) return {};

  peerId_.assign(static_cast<uint8_t*>(identity.data()),
                 static_cast<uint8_t*>(identity.data()) + identity.size());

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
  setConnectionLostCallbackCommon(std::move(callback));
}

void ZmqSocketTransport::setConnectionEstablishedCallback(
    std::function<void()> callback) {
  setConnectionEstablishedCallbackCommon(std::move(callback));
}

void ZmqSocketTransport::setReconnectBackoff(
    std::chrono::milliseconds initialDelay,
    std::chrono::milliseconds maxDelay) {
  setReconnectBackoffCommon(initialDelay, maxDelay);
}

}  // namespace tt::sockets
