// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_migration_orchestrator.hpp"

#include <utility>

#include "transport/kv_bounce_buffer.hpp"
#include "transport/kv_control_message.hpp"
#include "transport/kv_table_provisioning.hpp"
#include "utils/logger.hpp"

namespace tt::transport {

namespace {

// BeginMigration carries only the destination coordinates (the receiver holds
// no table; the sender ships the dst slice and the receiver relays it into
// BounceReady).
KvControlMessage beginMessage(uint64_t uuid, const KvSlice& dst) {
  KvControlMessage m;
  m.type = KvControlType::BEGIN_MIGRATION;
  m.uuid = uuid;
  m.slot = dst.slot;
  m.layer_begin = dst.layer_begin;
  m.layer_end = dst.layer_end;
  m.position_begin = dst.position_begin;
  m.position_end = dst.position_end;
  return m;
}

}  // namespace

// --- Sender ----------------------------------------------------------------

KvMigrationSender::KvMigrationSender(KvControlChannel& channel,
                                     MooncakeKvSender& sender)
    : channel_(channel), sender_(sender) {}

std::optional<std::vector<uint8_t>> KvMigrationSender::exchangeTables(
    const std::vector<uint8_t>& localTableBlob) {
  return exchangeTableBlob(channel_, TableExchangeRole::Sender, localTableBlob);
}

bool KvMigrationSender::migrate(uint64_t uuid,
                                const MigrationRequest& request) {
  KvControlChannel::Transaction txn(channel_);

  if (!channel_.send(beginMessage(uuid, request.dstSlice()))) {
    TT_LOG_ERROR("[KvMigrationSender] uuid={}: send BeginMigration failed",
                 uuid);
    return false;
  }

  KvControlMessage ready;
  switch (channel_.receiveMessage(ready)) {
    case KvControlChannel::ReceiveOutcome::TimedOut:
      TT_LOG_ERROR("[KvMigrationSender] uuid={}: timed out for BounceReady",
                   uuid);
      return false;
    case KvControlChannel::ReceiveOutcome::Closed:
      TT_LOG_ERROR("[KvMigrationSender] uuid={}: closed awaiting BounceReady",
                   uuid);
      return false;
    case KvControlChannel::ReceiveOutcome::Message:
      break;
  }
  if (ready.type != KvControlType::BOUNCE_READY || ready.uuid != uuid) {
    TT_LOG_ERROR(
        "[KvMigrationSender] uuid={}: expected BounceReady, got type={} "
        "uuid={}",
        uuid, static_cast<int>(ready.type), ready.uuid);
    return false;
  }
  if (!ready.ok || ready.segment_name.empty()) {
    TT_LOG_ERROR(
        "[KvMigrationSender] uuid={}: receiver could not ready its bounce "
        "buffer "
        "(ok={}, segment empty={})",
        uuid, ready.ok, ready.segment_name.empty());
    return false;
  }
  const BounceGeometry geometry{ready.bounce_section_count,
                                ready.bounce_section_size};

  // The sink sends one window and blocks for its ack — the credit handshake.
  const WindowSink sink =
      [&](uint64_t id,
          const std::vector<BounceSectionDescriptor>& window) -> bool {
    KvControlMessage wr;
    wr.type = KvControlType::WINDOW_READY;
    wr.uuid = id;
    wr.window = window;
    if (!channel_.send(wr)) {
      TT_LOG_ERROR("[KvMigrationSender] uuid={}: send WindowReady failed", id);
      return false;
    }
    KvControlMessage ack;
    if (channel_.receiveMessage(ack) !=
        KvControlChannel::ReceiveOutcome::Message) {
      TT_LOG_ERROR("[KvMigrationSender] uuid={}: no WindowAck", id);
      return false;
    }
    if (ack.type != KvControlType::WINDOW_ACK || ack.uuid != id || !ack.ok) {
      TT_LOG_ERROR(
          "[KvMigrationSender] uuid={}: bad WindowAck (type={}, uuid={}, "
          "ok={})",
          id, static_cast<int>(ack.type), ack.uuid, ack.ok);
      return false;
    }
    return true;
  };

  if (!sender_.transferSlot(uuid, request, ready.segment_name, geometry,
                            sink)) {
    TT_LOG_ERROR("[KvMigrationSender] uuid={}: transferSlot failed", uuid);
    return false;
  }

  KvControlMessage done;
  done.type = KvControlType::DONE_MARKER;
  done.uuid = uuid;
  if (!channel_.send(done)) {
    TT_LOG_ERROR("[KvMigrationSender] uuid={}: send DoneMarker failed", uuid);
    return false;
  }

  KvControlMessage ack;
  switch (channel_.receiveMessage(ack)) {
    case KvControlChannel::ReceiveOutcome::TimedOut:
      TT_LOG_ERROR("[KvMigrationSender] uuid={}: timed out for Ack", uuid);
      return false;
    case KvControlChannel::ReceiveOutcome::Closed:
      TT_LOG_ERROR("[KvMigrationSender] uuid={}: closed awaiting Ack", uuid);
      return false;
    case KvControlChannel::ReceiveOutcome::Message:
      break;
  }
  if (ack.type != KvControlType::ACK || ack.uuid != uuid || !ack.ok) {
    TT_LOG_ERROR(
        "[KvMigrationSender] uuid={}: bad final Ack (type={}, uuid={}, "
        "ok={})",
        uuid, static_cast<int>(ack.type), ack.uuid, ack.ok);
    return false;
  }
  TT_LOG_INFO("[KvMigrationSender] uuid={} complete", uuid);
  return true;
}

// --- Receiver --------------------------------------------------------------

KvMigrationReceiver::KvMigrationReceiver(
    KvControlChannel& channel, MooncakeKvReceiver& receiver,
    std::shared_ptr<const std::vector<uint8_t>> localTableBlob)
    : KvMigrationReceiver(channel, &receiver, std::move(localTableBlob)) {}

KvMigrationReceiver::KvMigrationReceiver(
    KvControlChannel& channel, MooncakeKvReceiver* receiver,
    std::shared_ptr<const std::vector<uint8_t>> localTableBlob)
    : channel_(channel),
      receiver_(receiver),
      local_table_blob_(std::move(localTableBlob)) {}

std::optional<std::vector<uint8_t>> KvMigrationReceiver::exchangeTables(
    const std::vector<uint8_t>& localTableBlob) {
  return exchangeTableBlob(channel_, TableExchangeRole::Receiver,
                           localTableBlob);
}

bool KvMigrationReceiver::handleTableExchange(const KvControlMessage& msg) {
  if (msg.table_blob.empty()) {
    TT_LOG_ERROR("[KvMigrationReceiver] TABLE_EXCHANGE missing peer blob");
    return false;
  }
  if (!local_table_blob_ || local_table_blob_->empty()) {
    TT_LOG_ERROR(
        "[KvMigrationReceiver] TABLE_EXCHANGE with no local table blob");
    return false;
  }
  KvControlMessage out;
  out.type = KvControlType::TABLE_EXCHANGE;
  out.role = static_cast<uint8_t>(TableExchangeRole::Receiver);
  out.table_blob = *local_table_blob_;
  if (!channel_.send(out, kDefaultTableExchangeTimeout)) {
    TT_LOG_ERROR("[KvMigrationReceiver] TABLE_EXCHANGE reply failed");
    return false;
  }
  return true;
}

bool KvMigrationReceiver::serveOne() {
  const auto msg = channel_.receive();
  if (!msg) return false;
  return handle(*msg);
}

bool KvMigrationReceiver::handle(const KvControlMessage& msg) {
  switch (msg.type) {
    case KvControlType::TABLE_EXCHANGE:
      return handleTableExchange(msg);
    case KvControlType::BEGIN_MIGRATION: {
      if (receiver_ == nullptr) {
        TT_LOG_WARN(
            "[KvMigrationReceiver] dry-run BeginMigration(uuid={}) received; "
            "no "
            "device or bounce buffer is available — rejecting",
            msg.uuid);
      }
      // The channel is strictly serial (the sender holds one Transaction across
      // a whole migration), so a fresh Begin means any surviving entry is an
      // abandoned migration whose Done never arrived. Clear so `ok_` stays
      // bounded to the in-flight uuid instead of leaking one entry per drop.
      ok_.clear();
      ok_[msg.uuid] = true;  // fresh migration — no window failed yet
      const bool registered = receiver_ != nullptr && receiver_->registered();
      KvControlMessage ready;
      ready.type = KvControlType::BOUNCE_READY;
      ready.uuid = msg.uuid;
      ready.ok = registered;
      ready.segment_name = registered ? receiver_->segmentName() : "";
      if (receiver_ != nullptr) {
        const BounceGeometry g = receiver_->geometry();
        ready.bounce_section_count = g.section_count;
        ready.bounce_section_size = g.section_size;
      }
      if (!channel_.send(ready)) {
        TT_LOG_ERROR("[KvMigrationReceiver] send BounceReady(uuid={}) failed",
                     msg.uuid);
        return false;
      }
      break;
    }
    case KvControlType::WINDOW_READY: {
      const bool drained =
          receiver_ != nullptr && receiver_->drainWindow(msg.window);
      if (!drained) ok_[msg.uuid] = false;
      KvControlMessage ack;
      ack.type = KvControlType::WINDOW_ACK;
      ack.uuid = msg.uuid;
      ack.ok = drained;
      ack.credits = static_cast<uint32_t>(msg.window.size());
      if (!channel_.send(ack)) {
        TT_LOG_ERROR("[KvMigrationReceiver] send WindowAck(uuid={}) failed",
                     msg.uuid);
        return false;
      }
      break;
    }
    case KvControlType::DONE_MARKER: {
      const auto it = ok_.find(msg.uuid);
      const bool allOk = it == ok_.end() ? false : it->second;
      ok_.erase(msg.uuid);
      // The Ack carries the drain status so the sender does not treat a failed
      // (partial/corrupt) drain as a successful migration.
      KvControlMessage ack;
      ack.type = KvControlType::ACK;
      ack.uuid = msg.uuid;
      ack.ok = allOk;
      if (!channel_.send(ack)) {
        TT_LOG_ERROR("[KvMigrationReceiver] send Ack(uuid={}) failed",
                     msg.uuid);
        return false;
      }
      break;
    }
    default:
      TT_LOG_WARN("[KvMigrationReceiver] unexpected message type {}",
                  static_cast<int>(msg.type));
      break;
  }
  return true;
}

void KvMigrationReceiver::run() {
  for (;;) {
    KvControlMessage msg;
    switch (channel_.receiveMessage(msg)) {
      case KvControlChannel::ReceiveOutcome::Closed:
        return;
      case KvControlChannel::ReceiveOutcome::TimedOut:
        continue;
      case KvControlChannel::ReceiveOutcome::Message:
        if (!handle(msg)) return;
        break;
    }
  }
}

}  // namespace tt::transport
