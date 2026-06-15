// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace tt::transport {

/**
 * @brief NOC address of a region in TT device DRAM.
 *
 * Encoded as `channel << 32 | local_addr`, matching the migration layer's
 * `noc_addr` convention. The UMD access wrapper (UmdDeviceAccess) reads/writes
 * device DRAM channels by this address.
 */
using NocAddr = uint64_t;

/// Packs a DRAM channel and channel-local byte offset into a NocAddr.
constexpr NocAddr makeNocAddr(uint32_t channel, uint32_t local_addr) {
  return (static_cast<NocAddr>(channel) << 32) | local_addr;
}

/// Extracts the DRAM channel from a NocAddr.
constexpr uint32_t nocChannel(NocAddr addr) {
  return static_cast<uint32_t>(addr >> 32);
}

/// Extracts the channel-local byte offset from a NocAddr.
constexpr uint32_t nocLocalAddr(NocAddr addr) {
  return static_cast<uint32_t>(addr & 0xFFFFFFFFULL);
}

/**
 * @brief Opaque handle to a remote, registered memory region (a "segment").
 *
 * Mirrors Mooncake's `SegmentHandle`/`SegmentID`. Obtained from
 * ITransferEngine::openSegment and used as the target of a transfer.
 */
using SegmentHandle = int64_t;

/// Default for an invalid / not-yet-opened segment.
inline constexpr SegmentHandle kInvalidSegment =
    std::numeric_limits<SegmentHandle>::min();

/**
 * @brief Direction of a transfer relative to the local registered buffer.
 *
 * Mirrors Mooncake's `TransferRequest::OpCode`.
 */
enum class TransferOp : uint8_t {
  Read,   ///< Pull bytes from the remote segment into the local buffer.
  Write,  ///< Push bytes from the local buffer into the remote segment.
};

/**
 * @brief A single transfer between a local registered buffer and a remote
 *        segment.
 *
 * `local_addr` must point inside a region previously passed to
 * ITransferEngine::registerLocalMemory.
 */
struct TransferRequest {
  TransferOp op = TransferOp::Write;
  void* local_addr = nullptr;
  SegmentHandle target = kInvalidSegment;
  /// Byte offset *into* the remote segment; the engine resolves it against the
  /// segment's registered base address.
  uint64_t target_offset = 0;
  std::size_t length = 0;
};

/// Lifecycle state of a submitted transfer. Mirrors Mooncake's status enum.
enum class TransferState : uint8_t {
  Pending,
  Completed,
  Failed,
};

/// Result of a submitted transfer.
struct TransferStatus {
  TransferState state = TransferState::Pending;
  std::size_t transferred_bytes = 0;
};

/**
 * @brief Storage mechanism a transfer addresses (issue #3890 core assumption:
 *        "Transfer Engine defines both Storage mechanism (host DRAM, device
 *        DRAM..) [and] Transport mechanism (TCP, RDMA..)").
 *
 * Selects which IStorageBackend stages bytes to/from a registered host buffer.
 */
enum class StorageMedium : uint8_t {
  HostDram,    ///< Plain host memory.
  DeviceDram,  ///< TT device DRAM, accessed via the UMD (the custom backend).
};

/// Underlying wire protocol the transport mechanism uses to move bytes.
enum class TransportProtocol : uint8_t {
  Tcp,   ///< Stock TCP transport. Default for the PoC.
  Rdma,  ///< RDMA transport. Requires libibverbs + an active NIC.
};

/**
 * @brief Configuration for bringing up a transfer engine.
 *
 * Captures both mechanisms the Transfer Engine defines (#3890): `protocol`
 * selects the *transport* mechanism; the *storage* mechanism is supplied
 * separately as an IStorageBackend. `metadata_uri` selects how peers exchange
 * segment metadata; "P2PHANDSHAKE" means a direct peer handshake with no
 * external etcd/redis/http store.
 */
struct EngineConfig {
  std::string metadata_uri = "P2PHANDSHAKE";
  std::string local_server_name;
  TransportProtocol protocol = TransportProtocol::Tcp;
};

}  // namespace tt::transport
