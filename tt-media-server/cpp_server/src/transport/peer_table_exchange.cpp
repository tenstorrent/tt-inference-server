// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/peer_table_exchange.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <utility>

#include "transport/i_transfer_engine.hpp"
#include "utils/logger.hpp"

namespace tt::transport {

namespace {

bool isCancelled(const std::atomic<bool>* cancelToken) {
  return cancelToken != nullptr &&
         cancelToken->load(std::memory_order_relaxed);
}

}  // namespace

PeerTableExchange::PeerTableExchange(PeerTableExchangeConfig config)
    : config_(std::move(config)) {
  if (config_.pollIntervalMs < 1) config_.pollIntervalMs = 1;
  if (config_.timeoutSec < 0) config_.timeoutSec = 0;
}

std::uint64_t PeerTableExchange::fnv1a(const std::uint8_t* data,
                                       std::size_t n) {
  std::uint64_t h = 1469598103934665603ULL;
  for (std::size_t i = 0; i < n; ++i) {
    h ^= data[i];
    h *= 1099511628211ULL;
  }
  return h;
}

bool PeerTableExchange::pushToPeer(ITransferEngine& engine, SegmentHandle peer,
                                   std::size_t remoteSlotIndex,
                                   const std::vector<std::uint8_t>& table,
                                   const TableHeader& header,
                                   const std::uint8_t& flag) const {
  const std::uint64_t base = slotOffset(remoteSlotIndex);
  auto writeAt = [&](const std::uint8_t* local, std::size_t len,
                     std::uint64_t peerOffset) {
    TransferRequest req;
    req.op = TransferOp::WRITE;
    req.local_addr = const_cast<std::uint8_t*>(local);
    req.target = peer;
    req.target_offset = peerOffset;
    req.length = len;
    return engine.submitAndWait(req).state == TransferState::COMPLETED;
  };
  // Body → header → flag. TCP sequential WRITEs: flag last ⇒ body+header
  // landed. Not RDMA-safe without a fence.
  return writeAt(table.data(), table.size(), base + headerBytes()) &&
         writeAt(reinterpret_cast<const std::uint8_t*>(&header), headerBytes(),
                 base) &&
         writeAt(&flag, 1, base + flagOffsetInSlot());
}

bool PeerTableExchange::waitForFlag(
    std::uint8_t* flag, const std::atomic<bool>* cancelToken) const {
  // Acquire load: pairs with the writer's completion (TCP submitAndWait returns
  // only after the remote store is visible) or a release store in unit-test
  // fakes. volatile alone is not a C++ synchronization primitive.
  std::atomic_ref<std::uint8_t> flagAtomic(*flag);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(config_.timeoutSec);
  while (flagAtomic.load(std::memory_order_acquire) != K_DONE_FLAG) {
    if (isCancelled(cancelToken)) return false;
    if (std::chrono::steady_clock::now() > deadline) {
      TT_LOG_ERROR("[PeerTableExchange] timed out waiting for peer flag");
      return false;
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(config_.pollIntervalMs));
  }
  return true;
}

std::optional<std::vector<std::uint8_t>> PeerTableExchange::readPeerTable(
    const std::uint8_t* slotBase) const {
  TableHeader header{};
  std::memcpy(&header, slotBase, headerBytes());
  if (header.tableBytes == 0 || header.tableBytes > config_.maxTableBytes) {
    TT_LOG_ERROR("[PeerTableExchange] bad peer header size {}",
                 header.tableBytes);
    return std::nullopt;
  }
  const auto* body = slotBase + headerBytes();
  if (fnv1a(body, header.tableBytes) != header.checksum) {
    TT_LOG_ERROR("[PeerTableExchange] peer table checksum mismatch");
    return std::nullopt;
  }
  return std::vector<std::uint8_t>(body, body + header.tableBytes);
}

std::optional<std::map<std::string, std::vector<std::uint8_t>>>
PeerTableExchange::exchange(ITransferEngine& engine,
                            const std::map<std::string, PeerSlot>& peers,
                            const std::string& localSegmentName,
                            const std::vector<std::uint8_t>& localBlob,
                            std::uint8_t* recvBase,
                            const std::atomic<bool>* cancelToken) const {
  if (peers.empty()) {
    return std::map<std::string, std::vector<std::uint8_t>>{};
  }
  if (config_.maxTableBytes == 0 || recvBase == nullptr) {
    TT_LOG_ERROR("[PeerTableExchange] invalid config/recvBase");
    return std::nullopt;
  }
  if (localBlob.size() > config_.maxTableBytes) {
    TT_LOG_ERROR("[PeerTableExchange] local table {} B > max {}",
                 localBlob.size(), config_.maxTableBytes);
    return std::nullopt;
  }

  // Validate slot indices fit the registered region.
  for (const auto& [name, slot] : peers) {
    if (slot.handle == K_INVALID_SEGMENT ||
        slot.localSlotIndex >= peers.size()) {
      TT_LOG_ERROR("[PeerTableExchange] bad slot for '{}'", name);
      return std::nullopt;
    }
  }

  TableHeader header{localBlob.size(),
                     fnv1a(localBlob.data(), localBlob.size())};
  std::uint8_t flag = K_DONE_FLAG;
  auto* tablePtr = const_cast<std::uint8_t*>(localBlob.data());
  if (!engine.registerLocalMemory(tablePtr, localBlob.size()) ||
      !engine.registerLocalMemory(&header, headerBytes()) ||
      !engine.registerLocalMemory(&flag, sizeof(flag))) {
    TT_LOG_ERROR("[PeerTableExchange] register outbound framing failed");
    return std::nullopt;
  }
  auto unregisterOutbound = [&] {
    engine.unregisterLocalMemory(&flag);
    engine.unregisterLocalMemory(&header);
    engine.unregisterLocalMemory(tablePtr);
  };

  // Push to every peer first (into their slot for our name). Per-peer slots
  // mean concurrent writers into OUR region are isolated — no half-duplex.
  for (const auto& [peerName, slot] : peers) {
    if (isCancelled(cancelToken)) {
      unregisterOutbound();
      return std::nullopt;
    }
    TT_LOG_INFO("[PeerTableExchange] '{}' push {} B -> '{}' (remote slot {})",
                localSegmentName, localBlob.size(), peerName,
                slot.remoteSlotIndex);
    if (!pushToPeer(engine, slot.handle, slot.remoteSlotIndex, localBlob,
                    header, flag)) {
      TT_LOG_ERROR("[PeerTableExchange] push to '{}' failed", peerName);
      unregisterOutbound();
      return std::nullopt;
    }
  }

  // Then collect every peer's blob from our per-peer slots.
  std::map<std::string, std::vector<std::uint8_t>> received;
  for (const auto& [peerName, slot] : peers) {
    if (isCancelled(cancelToken)) {
      unregisterOutbound();
      return std::nullopt;
    }
    auto* slotBase = recvBase + slotOffset(slot.localSlotIndex);
    auto* flagSlot = slotBase + flagOffsetInSlot();
    TT_LOG_INFO("[PeerTableExchange] '{}' wait table from '{}' (local slot {})",
                localSegmentName, peerName, slot.localSlotIndex);
    if (!waitForFlag(flagSlot, cancelToken)) {
      unregisterOutbound();
      return std::nullopt;
    }
    auto blob = readPeerTable(slotBase);
    if (!blob) {
      unregisterOutbound();
      return std::nullopt;
    }
    TT_LOG_INFO("[PeerTableExchange] got {} B from '{}'", blob->size(),
                peerName);
    received.emplace(peerName, std::move(*blob));
  }

  unregisterOutbound();
  TT_LOG_INFO("[PeerTableExchange] EXCHANGED with {} peers", received.size());
  return received;
}

}  // namespace tt::transport
