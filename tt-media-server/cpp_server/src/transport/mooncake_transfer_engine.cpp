// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/mooncake_transfer_engine.hpp"

#include <algorithm>
#include <utility>

#include "utils/logger.hpp"

#ifdef TT_TRANSPORT_WITH_MOONCAKE
#include <chrono>
#include <cstdint>
#include <cstdlib>  // setenv
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "common.h"                    // parseHostNameWithPort
#include "transfer_engine.h"           // mooncake::TransferEngine
#include "transfer_metadata_plugin.h"  // mooncake::MetadataStoragePlugin
#include "transport/transport.h"
#endif

namespace tt::transport {

#ifdef TT_TRANSPORT_WITH_MOONCAKE

namespace {

// Map our transport selector to Mooncake's installTransport proto string.
const char* protoString(TransportProtocol protocol) {
  return protocol == TransportProtocol::RDMA ? "rdma" : "tcp";
}

// Resolve a transfer's target_offset (relative to the remote segment) against
// the segment's registered base address, mirroring Mooncake's TCP usage.
// Returns std::nullopt if the segment has no registered buffer.
std::optional<uint64_t> resolveRemoteAddr(mooncake::TransferEngine& engine,
                                          mooncake::SegmentID segmentId,
                                          uint64_t targetOffset) {
  auto metadata = engine.getMetadata();
  auto segmentDesc =
      metadata ? metadata->getSegmentDescByID(segmentId) : nullptr;
  if (!segmentDesc || segmentDesc->buffers.empty()) return std::nullopt;
  return segmentDesc->buffers[0].addr + targetOffset;
}

// Poll interval while blocking on a submitted (batch) transfer. Small enough to
// not add meaningful latency, large enough that we don't spin a core at 100%
// and starve Mooncake's own transport progress threads for the transfer's
// whole duration.
constexpr auto K_POLL_INTERVAL = std::chrono::microseconds(50);

mooncake::TransferRequest::OpCode toMooncakeOp(TransferOp op) {
  return op == TransferOp::READ ? mooncake::TransferRequest::READ
                                : mooncake::TransferRequest::WRITE;
}

}  // namespace

// Holds the wrapped Mooncake engine and the transport it installed. The
// Transport* is owned by the engine, so we only keep it to detect a failed
// install.
struct MooncakeTransferEngine::Impl {
  std::unique_ptr<mooncake::TransferEngine> engine;
  mooncake::Transport* transport = nullptr;
  // Direct handle to the metadata service's key-value store (the same backing
  // store openSegment/rpc_meta use), so we can publish/resolve arbitrary peer
  // facts Mooncake's own registries don't carry (e.g. KV control endpoints).
  // Null under P2PHANDSHAKE, which has no backing store.
  std::shared_ptr<mooncake::MetadataStoragePlugin> metaStore;
  bool initialized = false;
};

MooncakeTransferEngine::MooncakeTransferEngine(
    std::shared_ptr<IStorageBackend> storage)
    : storage_(std::move(storage)), impl_(std::make_unique<Impl>()) {}

MooncakeTransferEngine::~MooncakeTransferEngine() = default;

MooncakeTransferEngine::MooncakeTransferEngine(
    MooncakeTransferEngine&&) noexcept = default;

MooncakeTransferEngine& MooncakeTransferEngine::operator=(
    MooncakeTransferEngine&&) noexcept = default;

StorageMedium MooncakeTransferEngine::storageMedium() const {
  return storage_ ? storage_->medium() : StorageMedium::HOST_DRAM;
}

std::shared_ptr<IStorageBackend> MooncakeTransferEngine::storage() const {
  return storage_;
}

std::string MooncakeTransferEngine::localServerName() const {
  if (!impl_->initialized) return {};
  return impl_->engine->getLocalIpAndPort();
}

bool MooncakeTransferEngine::init(const EngineConfig& config) {
  if (impl_->initialized) {
    TT_LOG_WARN("[MooncakeTransferEngine] init called twice; ignoring");
    return true;
  }

  // auto_discover=false: topology discovery is for RDMA NIC selection and not
  // needed for the TCP PoC.
  impl_->engine =
      std::make_unique<mooncake::TransferEngine>(/*auto_discover=*/false);

  const auto hostPort =
      mooncake::parseHostNameWithPort(config.local_server_name);
  const int rc =
      impl_->engine->init(config.metadata_uri, config.local_server_name,
                          hostPort.first.c_str(), hostPort.second);
  if (rc != 0) {
    TT_LOG_ERROR(
        "[MooncakeTransferEngine] TransferEngine::init(metadata_uri={}, "
        "local_server_name={}) failed (rc={})",
        config.metadata_uri, config.local_server_name, rc);
    impl_->engine.reset();
    return false;
  }

  const char* proto = protoString(config.protocol);

  // Mooncake's TCP transport defaults its connection pool OFF, which makes it
  // open AND tear down a fresh TCP connection for every slice (see
  // TcpTransport::getConnection: !enable_connection_pool_ => resolve+connect
  // per transfer). A KV migration submits tens of thousands of small writes, so
  // that per-slice connect/close dominates submitTransfer (measured ~3x
  // slower). Enable pooling so sockets are reused. overwrite=0: an operator who
  // set the var explicitly still wins. The flag is read when the transport is
  // installed below, so this must precede installTransport().
  if (config.protocol == TransportProtocol::TCP) {
    ::setenv("MC_TCP_ENABLE_CONNECTION_POOL", "1", /*overwrite=*/0);
  }

  impl_->transport = impl_->engine->installTransport(proto, nullptr);
  if (impl_->transport == nullptr) {
    TT_LOG_ERROR("[MooncakeTransferEngine] installTransport({}) failed", proto);
    impl_->engine.reset();
    return false;
  }

  // Open a direct client onto the metadata store for publish/lookupMetadata.
  // P2PHANDSHAKE has no backing store, so leave it null (those calls no-op and
  // callers fall back to a static convention). A failure here is non-fatal:
  // the transport is up, only cross-worker discovery of extra facts is lost.
  if (config.metadata_uri != "P2PHANDSHAKE") {
    impl_->metaStore =
        mooncake::MetadataStoragePlugin::Create(config.metadata_uri);
    if (!impl_->metaStore) {
      TT_LOG_WARN(
          "[MooncakeTransferEngine] no metadata store for '{}'; "
          "publishMetadata/lookupMetadata disabled",
          config.metadata_uri);
    }
  }

  impl_->initialized = true;
  TT_LOG_INFO(
      "[MooncakeTransferEngine] init OK (metadata_uri={}, "
      "local_server_name={}, "
      "protocol={})",
      config.metadata_uri, config.local_server_name, proto);
  return true;
}

bool MooncakeTransferEngine::registerLocalMemory(void* addr,
                                                 std::size_t length) {
  if (!impl_->initialized) {
    TT_LOG_ERROR("[MooncakeTransferEngine] registerLocalMemory before init");
    return false;
  }
  const int rc = impl_->engine->registerLocalMemory(addr, length);
  if (rc != 0) {
    TT_LOG_ERROR(
        "[MooncakeTransferEngine] registerLocalMemory(length={}) failed "
        "(rc={})",
        length, rc);
    return false;
  }
  registeredLocalBuffers_.push_back(addr);
  return true;
}

bool MooncakeTransferEngine::unregisterLocalMemory(void* addr) {
  if (!impl_->initialized) {
    TT_LOG_ERROR("[MooncakeTransferEngine] unregisterLocalMemory before init");
    return false;
  }
  const int rc = impl_->engine->unregisterLocalMemory(addr);
  if (rc != 0) {
    TT_LOG_ERROR(
        "[MooncakeTransferEngine] unregisterLocalMemory failed (rc={})", rc);
    return false;
  }
  const auto it = std::find(registeredLocalBuffers_.begin(),
                            registeredLocalBuffers_.end(), addr);
  if (it != registeredLocalBuffers_.end()) {
    registeredLocalBuffers_.erase(it);
  }
  return true;
}

void* MooncakeTransferEngine::firstRegisteredLocalBuffer() const {
  return registeredLocalBuffers_.empty() ? nullptr
                                         : registeredLocalBuffers_.front();
}

std::size_t MooncakeTransferEngine::registeredLocalBufferCount() const {
  return registeredLocalBuffers_.size();
}

SegmentHandle MooncakeTransferEngine::openSegment(
    const std::string& segmentName) {
  if (!impl_->initialized) {
    TT_LOG_ERROR("[MooncakeTransferEngine] openSegment before init");
    return K_INVALID_SEGMENT;
  }
  const mooncake::SegmentID id = impl_->engine->openSegment(segmentName);
  // Mooncake returns a negative SegmentID (cast to a huge unsigned) on failure.
  if (static_cast<int64_t>(id) < 0) {
    TT_LOG_ERROR("[MooncakeTransferEngine] openSegment({}) failed",
                 segmentName);
    return K_INVALID_SEGMENT;
  }
  return static_cast<SegmentHandle>(id);
}

SegmentHandle MooncakeTransferEngine::refreshSegment(
    const std::string& segmentName) {
  if (!impl_->initialized) {
    TT_LOG_ERROR("[MooncakeTransferEngine] refreshSegment before init");
    return K_INVALID_SEGMENT;
  }
  const mooncake::SegmentID id = impl_->engine->openSegment(segmentName);
  if (static_cast<int64_t>(id) < 0) {
    TT_LOG_ERROR(
        "[MooncakeTransferEngine] refreshSegment({}): openSegment "
        "failed",
        segmentName);
    return K_INVALID_SEGMENT;
  }
  // force_update re-fetches the descriptor from the metadata service and
  // overwrites the cached (pre-restart) entry, so the next submit reads the
  // peer's current address. The non-forced read in submitAndWait then sees it.
  auto metadata = impl_->engine->getMetadata();
  auto segmentDesc =
      metadata ? metadata->getSegmentDescByID(id, /*force_update=*/true)
               : nullptr;
  if (!segmentDesc || segmentDesc->buffers.empty()) {
    TT_LOG_ERROR(
        "[MooncakeTransferEngine] refreshSegment({}): no registered buffer "
        "after force-update",
        segmentName);
    return K_INVALID_SEGMENT;
  }
  return static_cast<SegmentHandle>(id);
}

std::string MooncakeTransferEngine::resolveServerName(
    const std::string& segmentName) {
  if (!impl_->initialized) return {};
  auto metadata = impl_->engine->getMetadata();
  if (!metadata) return {};
  // The peer's routable address lives in the metadata service's rpc_meta
  // registry, keyed by its server name (the peer's local_server_name, which it
  // published via MC_TCP_BIND_ADDRESS). With a metadata server this turns a
  // LOGICAL tag (e.g. "decode-0") into the peer's real IP; under P2PHANDSHAKE
  // Mooncake parses host:port straight from the name. We return the host only:
  // rpc_port is Mooncake's data-plane port, NOT the KV control port the caller
  // pairs this with. NB: SegmentDesc.name is just the looked-up name, so it
  // could not resolve a logical tag — rpc_meta is the correct source.
  mooncake::TransferMetadata::RpcMetaDesc rpc;
  if (metadata->getRpcMetaEntry(segmentName, rpc) != 0) return {};
  return rpc.ip_or_host_name;
}

bool MooncakeTransferEngine::publishMetadata(const std::string& key,
                                             const std::string& value) {
  if (!impl_->metaStore) return false;
  // Store the raw string as a JSON string so lookupMetadata reads it back
  // verbatim; the HTTP metadata server keeps the body as-is under `key`.
  return impl_->metaStore->set(key, Json::Value(value));
}

std::optional<std::string> MooncakeTransferEngine::lookupMetadata(
    const std::string& key) {
  if (!impl_->metaStore) return std::nullopt;
  Json::Value value;
  // get() returns false on a 404 (key absent) or any transport error, so an
  // unpublished peer fact surfaces as std::nullopt rather than a hard failure.
  if (!impl_->metaStore->get(key, value) || value.isNull()) return std::nullopt;
  return value.asString();
}

TransferStatus MooncakeTransferEngine::submitAndWait(
    const TransferRequest& request) {
  TransferStatus result{TransferState::FAILED, 0};

  if (!impl_->initialized) {
    TT_LOG_ERROR("[MooncakeTransferEngine] submitAndWait before init");
    return result;
  }
  if (request.target == K_INVALID_SEGMENT) {
    TT_LOG_ERROR("[MooncakeTransferEngine] submitAndWait with invalid segment");
    return result;
  }

  const auto segmentId = static_cast<mooncake::SegmentID>(request.target);

  const auto remoteAddr =
      resolveRemoteAddr(*impl_->engine, segmentId, request.target_offset);
  if (!remoteAddr) {
    TT_LOG_ERROR(
        "[MooncakeTransferEngine] submitAndWait: no registered buffer for "
        "segment {}",
        request.target);
    return result;
  }

  const mooncake::BatchID batchId = impl_->engine->allocateBatchID(1);

  mooncake::TransferRequest entry;
  entry.opcode = toMooncakeOp(request.op);
  entry.source = request.local_addr;
  entry.target_id = segmentId;
  entry.target_offset = *remoteAddr;
  entry.length = request.length;

  mooncake::Status s = impl_->engine->submitTransfer(batchId, {entry});
  if (!s.ok()) {
    TT_LOG_ERROR("[MooncakeTransferEngine] submitTransfer failed: {}",
                 std::string(s.message()));
    impl_->engine->freeBatchID(batchId);
    return result;
  }

  // Block until the single task completes or fails. Sleep between polls so we
  // don't spin a core against Mooncake's own transport progress threads for the
  // transfer's whole duration (same rationale as waitBatch).
  for (;;) {
    mooncake::TransferStatus status;
    mooncake::Status poll =
        impl_->engine->getTransferStatus(batchId, 0, status);
    if (!poll.ok()) {
      TT_LOG_ERROR("[MooncakeTransferEngine] getTransferStatus failed: {}",
                   std::string(poll.message()));
      break;
    }
    if (status.s == mooncake::TransferStatusEnum::COMPLETED) {
      result.state = TransferState::COMPLETED;
      result.transferred_bytes = status.transferred_bytes;
      break;
    }
    if (status.s == mooncake::TransferStatusEnum::FAILED ||
        status.s == mooncake::TransferStatusEnum::TIMEOUT ||
        status.s == mooncake::TransferStatusEnum::CANCELED ||
        status.s == mooncake::TransferStatusEnum::INVALID) {
      TT_LOG_ERROR("[MooncakeTransferEngine] transfer failed (status={})",
                   static_cast<int>(status.s));
      result.transferred_bytes = status.transferred_bytes;
      break;
    }
    std::this_thread::sleep_for(K_POLL_INTERVAL);
  }

  impl_->engine->freeBatchID(batchId);
  return result;
}

TransferHandle MooncakeTransferEngine::submitBatch(
    const std::vector<TransferRequest>& requests) {
  TransferHandle handle;  // invalid until dispatched

  if (requests.empty()) return handle;  // nothing to dispatch
  if (!impl_->initialized) {
    TT_LOG_ERROR("[MooncakeTransferEngine] submitBatch before init");
    return handle;
  }

  // Translate every request into a Mooncake entry, resolving each target's
  // remote base up front. A single bad request fails the whole batch: the
  // caller submits chunks it needs to land together, so a partial batch would
  // leave the migration in a half-written state.
  const auto tResolve0 = std::chrono::steady_clock::now();
  std::vector<mooncake::TransferRequest> entries;
  entries.reserve(requests.size());
  for (const TransferRequest& request : requests) {
    if (request.target == K_INVALID_SEGMENT) {
      TT_LOG_ERROR(
          "[MooncakeTransferEngine] submitBatch: invalid segment in "
          "batch");
      return handle;
    }
    const auto segmentId = static_cast<mooncake::SegmentID>(request.target);
    const auto remoteAddr =
        resolveRemoteAddr(*impl_->engine, segmentId, request.target_offset);
    if (!remoteAddr) {
      TT_LOG_ERROR(
          "[MooncakeTransferEngine] submitBatch: no registered buffer for "
          "segment {}",
          request.target);
      return handle;
    }
    mooncake::TransferRequest entry;
    entry.opcode = toMooncakeOp(request.op);
    entry.source = request.local_addr;
    entry.target_id = segmentId;
    entry.target_offset = *remoteAddr;
    entry.length = request.length;
    entries.push_back(entry);
  }

  const auto tResolve1 = std::chrono::steady_clock::now();

  // One batch sized to the whole request list, so all entries run concurrently
  // across Mooncake's worker threads instead of one blocking round-trip each.
  const mooncake::BatchID batchId =
      impl_->engine->allocateBatchID(entries.size());
  mooncake::Status s = impl_->engine->submitTransfer(batchId, entries);
  if (!s.ok()) {
    TT_LOG_ERROR(
        "[MooncakeTransferEngine] submitTransfer(batch of {}) failed: "
        "{}",
        entries.size(), std::string(s.message()));
    impl_->engine->freeBatchID(batchId);
    return handle;
  }
  const auto tSubmit1 = std::chrono::steady_clock::now();

  // Diagnostic: split submitBatch into our per-request resolve loop vs
  // Mooncake's allocateBatchID+submitTransfer, to see which half dominates.
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };
  TT_LOG_INFO(
      "[MooncakeTransferEngine] submitBatch: {} entries | resolveLoop={}ms | "
      "allocateBatchID+submitTransfer={}ms",
      entries.size(), ms(tResolve0, tResolve1), ms(tResolve1, tSubmit1));

  // Return without blocking: the caller stages its next window while these run.
  handle.value = static_cast<uint64_t>(batchId);
  handle.valid = true;
  return handle;
}

TransferStatus MooncakeTransferEngine::waitBatch(TransferHandle handle) {
  TransferStatus result{TransferState::FAILED, 0};
  if (!handle.valid) {
    TT_LOG_ERROR("[MooncakeTransferEngine] waitBatch on invalid handle");
    return result;
  }
  if (!impl_->initialized) {
    TT_LOG_ERROR("[MooncakeTransferEngine] waitBatch before init");
    return result;
  }

  const auto batchId = static_cast<mooncake::BatchID>(handle.value);

  // Block on the aggregate batch status: COMPLETED only once every task
  // finished, FAILED if any task failed. Sleep between polls so we don't spin a
  // core against the transport threads for the batch's whole duration.
  for (;;) {
    mooncake::TransferStatus status;
    mooncake::Status poll =
        impl_->engine->getBatchTransferStatus(batchId, status);
    if (!poll.ok()) {
      TT_LOG_ERROR("[MooncakeTransferEngine] getBatchTransferStatus failed: {}",
                   std::string(poll.message()));
      break;
    }
    if (status.s == mooncake::TransferStatusEnum::COMPLETED) {
      result.state = TransferState::COMPLETED;
      result.transferred_bytes = status.transferred_bytes;
      break;
    }
    if (status.s == mooncake::TransferStatusEnum::FAILED ||
        status.s == mooncake::TransferStatusEnum::TIMEOUT ||
        status.s == mooncake::TransferStatusEnum::CANCELED ||
        status.s == mooncake::TransferStatusEnum::INVALID) {
      TT_LOG_ERROR("[MooncakeTransferEngine] batch transfer failed (status={})",
                   static_cast<int>(status.s));
      result.transferred_bytes = status.transferred_bytes;
      break;
    }
    std::this_thread::sleep_for(K_POLL_INTERVAL);
  }

  impl_->engine->freeBatchID(batchId);
  return result;
}

#else  // !TT_TRANSPORT_WITH_MOONCAKE

// Fallback when Mooncake is not in the build (default / CI): keep the engine a
// no-op that reports failure, so transport_lib builds in every configuration.
struct MooncakeTransferEngine::Impl {};

MooncakeTransferEngine::MooncakeTransferEngine(
    std::shared_ptr<IStorageBackend> storage)
    : storage_(std::move(storage)), impl_(std::make_unique<Impl>()) {}

MooncakeTransferEngine::~MooncakeTransferEngine() = default;

MooncakeTransferEngine::MooncakeTransferEngine(
    MooncakeTransferEngine&&) noexcept = default;

MooncakeTransferEngine& MooncakeTransferEngine::operator=(
    MooncakeTransferEngine&&) noexcept = default;

StorageMedium MooncakeTransferEngine::storageMedium() const {
  return storage_ ? storage_->medium() : StorageMedium::HOST_DRAM;
}

std::shared_ptr<IStorageBackend> MooncakeTransferEngine::storage() const {
  return storage_;
}

std::string MooncakeTransferEngine::localServerName() const { return {}; }

bool MooncakeTransferEngine::init(const EngineConfig& config) {
  TT_LOG_WARN(
      "[MooncakeTransferEngine] init(metadata_uri={}, local_server_name={}, "
      "protocol={}) unavailable (built without Mooncake)",
      config.metadata_uri, config.local_server_name,
      config.protocol == TransportProtocol::RDMA ? "rdma" : "tcp");
  return false;
}

bool MooncakeTransferEngine::registerLocalMemory(void* /*addr*/,
                                                 std::size_t length) {
  TT_LOG_WARN(
      "[MooncakeTransferEngine] registerLocalMemory(length={}) unavailable "
      "(built without Mooncake)",
      length);
  return false;
}

bool MooncakeTransferEngine::unregisterLocalMemory(void* /*addr*/) {
  TT_LOG_WARN(
      "[MooncakeTransferEngine] unregisterLocalMemory unavailable (built "
      "without Mooncake)");
  return false;
}

void* MooncakeTransferEngine::firstRegisteredLocalBuffer() const {
  return registeredLocalBuffers_.empty() ? nullptr
                                         : registeredLocalBuffers_.front();
}

std::size_t MooncakeTransferEngine::registeredLocalBufferCount() const {
  return registeredLocalBuffers_.size();
}

SegmentHandle MooncakeTransferEngine::openSegment(
    const std::string& segmentName) {
  TT_LOG_WARN(
      "[MooncakeTransferEngine] openSegment({}) unavailable (built without "
      "Mooncake)",
      segmentName);
  return K_INVALID_SEGMENT;
}

SegmentHandle MooncakeTransferEngine::refreshSegment(
    const std::string& segmentName) {
  TT_LOG_WARN(
      "[MooncakeTransferEngine] refreshSegment({}) unavailable (built without "
      "Mooncake)",
      segmentName);
  return K_INVALID_SEGMENT;
}

std::string MooncakeTransferEngine::resolveServerName(
    const std::string& segmentName) {
  TT_LOG_WARN(
      "[MooncakeTransferEngine] resolveServerName({}) unavailable (built "
      "without Mooncake)",
      segmentName);
  return {};
}

bool MooncakeTransferEngine::publishMetadata(const std::string& /*key*/,
                                             const std::string& /*value*/) {
  TT_LOG_WARN(
      "[MooncakeTransferEngine] publishMetadata unavailable (built without "
      "Mooncake)");
  return false;
}

std::optional<std::string> MooncakeTransferEngine::lookupMetadata(
    const std::string& /*key*/) {
  TT_LOG_WARN(
      "[MooncakeTransferEngine] lookupMetadata unavailable (built without "
      "Mooncake)");
  return std::nullopt;
}

TransferStatus MooncakeTransferEngine::submitAndWait(
    const TransferRequest& request) {
  TT_LOG_WARN(
      "[MooncakeTransferEngine] submitAndWait(op={}, length={}, target={}, "
      "target_offset={}) unavailable (built without Mooncake)",
      request.op == TransferOp::READ ? "read" : "write", request.length,
      request.target, request.target_offset);
  return TransferStatus{TransferState::FAILED, 0};
}

TransferHandle MooncakeTransferEngine::submitBatch(
    const std::vector<TransferRequest>& requests) {
  TT_LOG_WARN(
      "[MooncakeTransferEngine] submitBatch({} requests) unavailable (built "
      "without Mooncake)",
      requests.size());
  return TransferHandle{};  // invalid
}

TransferStatus MooncakeTransferEngine::waitBatch(TransferHandle /*handle*/) {
  TT_LOG_WARN(
      "[MooncakeTransferEngine] waitBatch unavailable (built without "
      "Mooncake)");
  return TransferStatus{TransferState::FAILED, 0};
}

#endif  // TT_TRANSPORT_WITH_MOONCAKE

}  // namespace tt::transport
