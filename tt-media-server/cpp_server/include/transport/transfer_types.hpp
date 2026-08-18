// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace tt::transport {

/**
 * @brief NOC address of a region in TT device DRAM.
 *
 * Encoded as `channel << 32 | local_addr`, matching the migration layer's
 * `noc_addr` convention. DriscDeviceIo reads/writes device DRAM channels by
 * this address.
 */
using NocAddr = uint64_t;

/// Packs a DRAM channel and channel-local byte offset into a NocAddr.
constexpr NocAddr makeNocAddr(uint32_t channel, uint32_t localAddr) {
  return (static_cast<NocAddr>(channel) << 32) | localAddr;
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
inline constexpr SegmentHandle K_INVALID_SEGMENT =
    std::numeric_limits<SegmentHandle>::min();

/**
 * @brief Direction of a transfer relative to the local registered buffer.
 *
 * Mirrors Mooncake's `TransferRequest::OpCode`.
 */
enum class TransferOp : uint8_t {
  READ,   ///< Pull bytes from the remote segment into the local buffer.
  WRITE,  ///< Push bytes from the local buffer into the remote segment.
};

/**
 * @brief A single transfer between a local registered buffer and a remote
 *        segment.
 *
 * `local_addr` must point inside a region previously passed to
 * ITransferEngine::registerLocalMemory.
 */
struct TransferRequest {
  TransferOp op = TransferOp::WRITE;
  void* local_addr = nullptr;
  SegmentHandle target = K_INVALID_SEGMENT;
  /// Byte offset *into* the remote segment; the engine resolves it against the
  /// segment's registered base address.
  uint64_t target_offset = 0;
  std::size_t length = 0;
};

/// Lifecycle state of a submitted transfer. Mirrors Mooncake's status enum.
enum class TransferState : uint8_t {
  PENDING,
  COMPLETED,
  FAILED,
};

/// Result of a submitted transfer.
struct TransferStatus {
  TransferState state = TransferState::PENDING;
  std::size_t transferred_bytes = 0;
};

/**
 * @brief Opaque handle to a batch submitted asynchronously via
 *        ITransferEngine::submitBatch, awaited with ITransferEngine::waitBatch.
 *
 * `value` is engine-private (the Mooncake BatchID) and meaningless to callers.
 * A default-constructed handle is invalid — it means submitBatch failed before
 * dispatch, and waitBatch on it reports FAILED.
 */
struct TransferHandle {
  uint64_t value = 0;  ///< engine-private; do not interpret.
  bool valid = false;  ///< false if the batch was never dispatched.
};

/**
 * @brief Storage mechanism a transfer addresses (issue #3890 core assumption:
 *        "Transfer Engine defines both Storage mechanism (host DRAM, device
 *        DRAM..) [and] Transport mechanism (TCP, RDMA..)").
 *
 * Selects which IStorageBackend stages bytes to/from a registered host buffer.
 */
enum class StorageMedium : uint8_t {
  HOST_DRAM,    ///< Plain host memory.
  DEVICE_DRAM,  ///< TT device DRAM, accessed via the UMD (the custom backend).
};

/// Underlying wire protocol the transport mechanism uses to move bytes.
enum class TransportProtocol : uint8_t {
  TCP,   ///< Stock TCP transport. Default for the PoC.
  RDMA,  ///< RDMA transport. Requires libibverbs + an active NIC.
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
  TransportProtocol protocol = TransportProtocol::TCP;
  /// Optional RDMA NIC allowlist (device names like "mlx5_0"). Empty (the
  /// default) => Mooncake auto-discovers whatever NIC(s) are present, which is
  /// the single-NIC path. Non-empty restricts RDMA topology discovery to
  /// exactly these devices — set it on a multi-NIC host to pin which NIC(s) the
  /// data plane uses (name one to force a single NIC, or several for multi-NIC
  /// fan-out). Only consulted under the RDMA protocol; ignored for TCP.
  std::vector<std::string> rdma_nic_filter;
};

}  // namespace tt::transport
