// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "transport/transfer_types.hpp"

namespace tt::transport {

class ITransferEngine;

/// Default / fleet-wide max table body (512 MiB). Keep in sync with deploy.
inline constexpr std::size_t K_DEFAULT_MAX_TABLE_BYTES = 512ull << 20;

/// Tunables for Transfer-Engine KV-table exchange (#4295 / #4279 PoC).
/// maxTableBytes must match across the fleet — it fixes the per-peer slot
/// stride. Flag-last completion assumes TCP write ordering (not RDMA-safe
/// without a fence). Hard constraint today: TCP only.
struct PeerTableExchangeConfig {
  int timeoutSec = 30;
  int pollIntervalMs = 1;
  /// Max table body per slot; slot = header + this + 1-byte flag.
  std::size_t maxTableBytes = K_DEFAULT_MAX_TABLE_BYTES;
};

/**
 * @brief Symmetric init-time KV-table exchange over the Transfer Engine.
 *
 * One registered recv region (buffers[0]) holds N per-peer slots:
 *   slot(i) = [TableHeader | body maxTableBytes | done-flag]
 * Peer i (sorted local peer names) WRITEs only into slot(i) — no fan-in races
 * under NP×MD mesh. Push is concurrent both ways (slots isolate writers).
 *
 * Remote slot index = index of localSegmentName in the peer's sorted peer
 * list (looked up from metadata, or passed in). Call BEFORE registering the
 * KV mirror so exchange can unregister and the mirror becomes buffers[0].
 */
class PeerTableExchange {
 public:
  struct TableHeader {
    std::uint64_t tableBytes = 0;
    std::uint64_t checksum = 0;  // FNV-1a
  };

  /// Per-peer targeting: where they write into us / where we write into them.
  struct PeerSlot {
    SegmentHandle handle = K_INVALID_SEGMENT;
    std::size_t localSlotIndex = 0;   ///< Their writes land in our slot[i].
    std::size_t remoteSlotIndex = 0;  ///< Our writes land in their slot[j].
  };

  static constexpr std::uint8_t K_DONE_FLAG = 0xAB;
  static constexpr std::size_t headerBytes() { return sizeof(TableHeader); }

  explicit PeerTableExchange(PeerTableExchangeConfig config = {});

  static std::uint64_t fnv1a(const std::uint8_t* data, std::size_t n);

  std::size_t slotBytes() const {
    return headerBytes() + config_.maxTableBytes + 1;
  }
  std::size_t requiredRecvBytes(std::size_t peerCount) const {
    return peerCount == 0 ? 0 : peerCount * slotBytes();
  }
  std::size_t slotOffset(std::size_t slotIndex) const {
    return slotIndex * slotBytes();
  }
  std::size_t flagOffsetInSlot() const {
    return headerBytes() + config_.maxTableBytes;
  }

  /// Exchange @p localBlob with every peer. @p recvBase must cover
  /// requiredRecvBytes(peers.size()) and already be registered as buffers[0].
  /// @p peers is keyed by peer name; slot indices must be consistent with the
  /// peer's own sorted peer list (remote) and ours (local).
  std::optional<std::map<std::string, std::vector<std::uint8_t>>> exchange(
      ITransferEngine& engine, const std::map<std::string, PeerSlot>& peers,
      const std::string& localSegmentName,
      const std::vector<std::uint8_t>& localBlob, std::uint8_t* recvBase,
      const std::atomic<bool>* cancelToken = nullptr) const;

 private:
  bool pushToPeer(ITransferEngine& engine, SegmentHandle peer,
                  std::size_t remoteSlotIndex,
                  const std::vector<std::uint8_t>& table,
                  const TableHeader& header, const std::uint8_t& flag) const;
  /// Polls @p flag with acquire loads (NIC/TCP writer; not a C++ atomic store).
  bool waitForFlag(std::uint8_t* flag,
                   const std::atomic<bool>* cancelToken) const;
  std::optional<std::vector<std::uint8_t>> readPeerTable(
      const std::uint8_t* slotBase) const;

  PeerTableExchangeConfig config_;
};

}  // namespace tt::transport
