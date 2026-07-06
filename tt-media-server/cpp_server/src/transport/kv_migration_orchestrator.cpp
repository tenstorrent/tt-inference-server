// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_migration_orchestrator.hpp"

#include "transport/kv_control_message.hpp"
#include "utils/logger.hpp"

namespace tt::transport {

namespace {

// BeginMigration carries only the destination coordinates: the receiver builds
// its mirror/drain against its own (decode) table, and the sender holds the
// source coordinates locally. So the wire format is unchanged by the asymmetric
// request — it ships the dst slice.
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

KvSlice sliceOf(const KvControlMessage& m) {
  return KvSlice{m.slot, m.layer_begin, m.layer_end, m.position_begin,
                 m.position_end};
}

}  // namespace

// --- Sender ----------------------------------------------------------------

KvMigrationSender::KvMigrationSender(KvControlChannel& channel,
                                     MooncakeKvSender& sender)
    : channel_(channel), sender_(sender) {}

std::optional<std::vector<uint8_t>> KvMigrationSender::exchangeTables(
    const std::vector<uint8_t>& localTableBlob) {
  KvControlMessage out;
  out.type = KvControlType::TABLE_EXCHANGE;
  out.role = 0;  // prefill/sender
  out.table_blob = localTableBlob;
  if (!channel_.send(out)) return std::nullopt;

  const auto peer = channel_.receive();
  if (!peer || peer->type != KvControlType::TABLE_EXCHANGE) {
    TT_LOG_ERROR("[KvMigrationSender] exchangeTables: bad/absent peer table");
    return std::nullopt;
  }
  return peer->table_blob;
}

bool KvMigrationSender::migrate(uint64_t uuid,
                                const MigrationRequest& request) {
  if (!channel_.send(beginMessage(uuid, request.dstSlice()))) {
    TT_LOG_ERROR(
        "[KvMigrationSender] migrate(uuid={}): send BeginMigration failed",
        uuid);
    return false;
  }

  const auto ready = channel_.receive();
  if (!ready || ready->type != KvControlType::MIRROR_READY ||
      ready->uuid != uuid) {
    TT_LOG_ERROR(
        "[KvMigrationSender] migrate(uuid={}): expected MirrorReady, got "
        "something else",
        uuid);
    return false;
  }
  if (!ready->ok || ready->segment_name.empty()) {
    TT_LOG_ERROR(
        "[KvMigrationSender] migrate(uuid={}): receiver failed to prepare "
        "mirror (ok={}, segment empty={})",
        uuid, ready->ok, ready->segment_name.empty());
    return false;
  }

  if (!sender_.transferSlot(request, ready->segment_name)) {
    TT_LOG_ERROR("[KvMigrationSender] migrate(uuid={}): transferSlot failed",
                 uuid);
    return false;
  }

  KvControlMessage done;
  done.type = KvControlType::DONE_MARKER;
  done.uuid = uuid;
  if (!channel_.send(done)) {
    TT_LOG_ERROR("[KvMigrationSender] migrate(uuid={}): send DoneMarker failed",
                 uuid);
    return false;
  }

  const auto ack = channel_.receive();
  if (!ack || ack->type != KvControlType::ACK || ack->uuid != uuid) {
    TT_LOG_ERROR("[KvMigrationSender] migrate(uuid={}): expected Ack", uuid);
    return false;
  }
  if (!ack->ok) {
    TT_LOG_ERROR(
        "[KvMigrationSender] migrate(uuid={}): receiver reported drain failure",
        uuid);
    return false;
  }
  TT_LOG_INFO("[KvMigrationSender] migrate(uuid={}) complete", uuid);
  return true;
}

// --- Receiver --------------------------------------------------------------

KvMigrationReceiver::KvMigrationReceiver(KvControlChannel& channel,
                                         MooncakeKvReceiver& receiver)
    : channel_(channel), receiver_(receiver) {}

std::optional<std::vector<uint8_t>> KvMigrationReceiver::exchangeTables(
    const std::vector<uint8_t>& localTableBlob) {
  const auto peer = channel_.receive();
  if (!peer || peer->type != KvControlType::TABLE_EXCHANGE) {
    TT_LOG_ERROR("[KvMigrationReceiver] exchangeTables: bad/absent peer table");
    return std::nullopt;
  }
  KvControlMessage out;
  out.type = KvControlType::TABLE_EXCHANGE;
  out.role = 1;  // decode/receiver
  out.table_blob = localTableBlob;
  if (!channel_.send(out)) return std::nullopt;
  return peer->table_blob;
}

bool KvMigrationReceiver::serveOne() {
  const auto msg = channel_.receive();
  if (!msg) return false;  // channel closed

  switch (msg->type) {
    case KvControlType::BEGIN_MIGRATION: {
      const auto segment = receiver_.prepareMirror(sliceOf(*msg), msg->uuid);
      KvControlMessage ready;
      ready.type = KvControlType::MIRROR_READY;
      ready.uuid = msg->uuid;
      ready.ok = segment.has_value();
      ready.segment_name = segment.value_or("");
      if (!channel_.send(ready)) {
        TT_LOG_ERROR("[KvMigrationReceiver] send MirrorReady(uuid={}) failed",
                     msg->uuid);
        return false;  // peer will block on receive; stop serving
      }
      break;
    }
    case KvControlType::DONE_MARKER: {
      const bool ok = receiver_.drain(msg->uuid);
      if (!ok) {
        TT_LOG_ERROR("[KvMigrationReceiver] drain(uuid={}) failed", msg->uuid);
      }
      // The Ack carries the drain status so the sender does not treat a failed
      // (partial/corrupt) drain as a successful migration.
      KvControlMessage ack;
      ack.type = KvControlType::ACK;
      ack.uuid = msg->uuid;
      ack.ok = ok;
      if (!channel_.send(ack)) {
        TT_LOG_ERROR("[KvMigrationReceiver] send Ack(uuid={}) failed",
                     msg->uuid);
        return false;  // peer will block on receive; stop serving
      }
      break;
    }
    default:
      TT_LOG_WARN("[KvMigrationReceiver] unexpected message type {}",
                  static_cast<int>(msg->type));
      break;
  }
  return true;
}

void KvMigrationReceiver::run() {
  while (serveOne()) {
  }
}

}  // namespace tt::transport
