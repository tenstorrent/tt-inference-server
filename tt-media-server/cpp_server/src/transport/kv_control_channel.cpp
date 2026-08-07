// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_control_channel.hpp"

#include <thread>
#include <utility>

#include "utils/logger.hpp"

namespace tt::transport {

namespace {

// Arms the transport's wall-clock IO budget for one send/recv burst so a
// mid-payload stall cannot pin the socket mutex past the call's timeout.
struct IoBudgetGuard {
  sockets::ISocketTransport* transport = nullptr;
  explicit IoBudgetGuard(sockets::ISocketTransport* t,
                         std::chrono::milliseconds budget)
      : transport(t) {
    if (transport != nullptr) {
      transport->beginIoBudget(budget);
    }
  }
  ~IoBudgetGuard() {
    if (transport != nullptr) {
      transport->clearIoBudget();
    }
  }
  IoBudgetGuard(const IoBudgetGuard&) = delete;
  IoBudgetGuard& operator=(const IoBudgetGuard&) = delete;
};

}  // namespace

KvControlChannel::Transaction::Transaction(KvControlChannel& channel)
    : lock_(channel.txnMutex_) {}

KvControlChannel::Transaction::Transaction(KvControlChannel& channel,
                                           std::try_to_lock_t)
    : lock_(channel.txnMutex_, std::try_to_lock) {}

KvControlChannel::KvControlChannel(
    std::shared_ptr<sockets::ISocketTransport> transport,
    std::chrono::milliseconds receiveTimeout,
    std::chrono::milliseconds pollInterval)
    : transport_(std::move(transport)),
      receive_timeout_(receiveTimeout),
      poll_interval_(pollInterval) {}

bool KvControlChannel::isConnected() const {
  return transport_ && transport_->isConnected();
}

bool KvControlChannel::send(const KvControlMessage& message) {
  return send(message, receive_timeout_);
}

bool KvControlChannel::send(const KvControlMessage& message,
                            std::chrono::milliseconds ioTimeout) {
  if (!transport_) {
    TT_LOG_ERROR("[KvControlChannel] send with no transport");
    return false;
  }
  const std::vector<uint8_t> bytes = message.serialize();
  if (bytes.empty()) {
    TT_LOG_ERROR(
        "[KvControlChannel] send: serialization failed (oversized field)");
    return false;
  }
  std::lock_guard<std::recursive_mutex> lock(txnMutex_);
  // TABLE_EXCHANGE send can block on the decode accept backlog; migrate Begin
  // stays on the short channel default. Without a deadline that pins
  // socketMutex forever.
  IoBudgetGuard budget(transport_.get(), ioTimeout);
  return transport_->sendRawData(bytes);
}

KvControlChannel::ReceiveOutcome KvControlChannel::receiveMessage(
    KvControlMessage& out) {
  return receiveMessage(out, receive_timeout_);
}

KvControlChannel::ReceiveOutcome KvControlChannel::receiveMessage(
    KvControlMessage& out, std::chrono::milliseconds ioTimeout) {
  if (!transport_) {
    TT_LOG_ERROR("[KvControlChannel] receive with no transport");
    return ReceiveOutcome::Closed;
  }
  std::lock_guard<std::recursive_mutex> lock(txnMutex_);
  return receiveMessageLocked(out, ioTimeout);
}

KvControlChannel::ReceiveOutcome KvControlChannel::receiveMessageLocked(
    KvControlMessage& out, std::chrono::milliseconds ioTimeout) {
  // One budget for the whole wait (idle NO_DATA polls + in-flight payload).
  // TcpSocketTransport enforces it inside tryReceiveMessage so mid-payload
  // stalls cannot make the deadline check below unreachable.
  IoBudgetGuard budget(transport_.get(), ioTimeout);
  const auto deadline = std::chrono::steady_clock::now() + ioTimeout;
  for (;;) {
    // Check BEFORE probing: a poll_interval_ sleep can cross the deadline
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      return ReceiveOutcome::TimedOut;
    }

    sockets::ReceiveResult result = transport_->tryReceiveMessage();
    switch (result.status) {
      case sockets::ReceiveStatus::DATA: {
        auto message = KvControlMessage::deserialize(result.data);
        if (!message) {
          TT_LOG_ERROR(
              "[KvControlChannel] received malformed message ({} bytes)",
              result.data.size());
          return ReceiveOutcome::Closed;  // corrupt stream: unusable
        }
        out = *message;
        return ReceiveOutcome::Message;
      }
      case sockets::ReceiveStatus::CLOSED:
        return ReceiveOutcome::Closed;  // connection closed
      case sockets::ReceiveStatus::NO_DATA:
        // The peer hasn't replied yet (e.g. still preparing the mirror). Wait
        // and retry rather than aborting a healthy migration on a normal
        // few-millisecond response delay. This is the whole reason the
        // transport reports NO_DATA distinctly instead of an ambiguous empty
        // buffer: receive() never has to guess via isConnected().
        break;
    }

    const auto remaining = deadline - std::chrono::steady_clock::now();
    if (remaining <= std::chrono::milliseconds::zero()) {
      return ReceiveOutcome::TimedOut;
    }
    std::this_thread::sleep_for(remaining < poll_interval_ ? remaining
                                                           : poll_interval_);
  }
}

std::optional<KvControlMessage> KvControlChannel::receive() {
  return receive(receive_timeout_);
}

std::optional<KvControlMessage> KvControlChannel::receive(
    std::chrono::milliseconds ioTimeout) {
  KvControlMessage msg;
  switch (receiveMessage(msg, ioTimeout)) {
    case ReceiveOutcome::Message:
      return msg;
    case ReceiveOutcome::TimedOut:
      // A bounded wait (the sender awaiting MirrorReady/Ack) that expires is an
      // error for that caller; log it here, where the timeout is unexpected.
      TT_LOG_ERROR("[KvControlChannel] receive timed out after {} ms",
                   ioTimeout.count());
      return std::nullopt;
    case ReceiveOutcome::Closed:
      return std::nullopt;
  }
  return std::nullopt;
}

}  // namespace tt::transport
