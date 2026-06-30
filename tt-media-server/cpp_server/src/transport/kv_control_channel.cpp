// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_control_channel.hpp"

#include <thread>
#include <utility>

#include "utils/logger.hpp"

namespace tt::transport {

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
  return transport_->sendRawData(bytes);
}

std::optional<KvControlMessage> KvControlChannel::receive() {
  if (!transport_) {
    TT_LOG_ERROR("[KvControlChannel] receive with no transport");
    return std::nullopt;
  }

  const auto deadline = std::chrono::steady_clock::now() + receive_timeout_;
  for (;;) {
    sockets::ReceiveResult result = transport_->tryReceiveMessage();
    switch (result.status) {
      case sockets::ReceiveStatus::DATA: {
        auto message = KvControlMessage::deserialize(result.data);
        if (!message) {
          TT_LOG_ERROR(
              "[KvControlChannel] received malformed message ({} bytes)",
              result.data.size());
        }
        return message;
      }
      case sockets::ReceiveStatus::CLOSED:
        return std::nullopt;  // connection closed
      case sockets::ReceiveStatus::NO_DATA:
        // The peer hasn't replied yet (e.g. still preparing the mirror). Wait
        // and retry rather than aborting a healthy migration on a normal
        // few-millisecond response delay. This is the whole reason the
        // transport reports NO_DATA distinctly instead of an ambiguous empty
        // buffer: receive() never has to guess via isConnected().
        break;
    }

    if (std::chrono::steady_clock::now() >= deadline) {
      TT_LOG_ERROR("[KvControlChannel] receive timed out after {} ms",
                   receive_timeout_.count());
      return std::nullopt;
    }
    std::this_thread::sleep_for(poll_interval_);
  }
}

}  // namespace tt::transport
