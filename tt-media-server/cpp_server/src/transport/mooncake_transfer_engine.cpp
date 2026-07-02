// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/mooncake_transfer_engine.hpp"

#include <utility>

#include "utils/logger.hpp"

#ifdef TT_TRANSPORT_WITH_MOONCAKE
#include <cstdint>
#include <string>
#include <vector>

#include "common.h"           // parseHostNameWithPort
#include "transfer_engine.h"  // mooncake::TransferEngine
#include "transport/transport.h"
#endif

namespace tt::transport {

#ifdef TT_TRANSPORT_WITH_MOONCAKE

namespace {

// Map our transport selector to Mooncake's installTransport proto string.
const char* protoString(TransportProtocol protocol) {
  return protocol == TransportProtocol::RDMA ? "rdma" : "tcp";
}

}  // namespace

// Holds the wrapped Mooncake engine and the transport it installed. The
// Transport* is owned by the engine, so we only keep it to detect a failed
// install.
struct MooncakeTransferEngine::Impl {
  std::unique_ptr<mooncake::TransferEngine> engine;
  mooncake::Transport* transport = nullptr;
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
  impl_->transport = impl_->engine->installTransport(proto, nullptr);
  if (impl_->transport == nullptr) {
    TT_LOG_ERROR("[MooncakeTransferEngine] installTransport({}) failed", proto);
    impl_->engine.reset();
    return false;
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
  return true;
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

  // Resolve target_offset (relative to the remote segment) against the
  // segment's registered base address, mirroring Mooncake's TCP usage.
  auto metadata = impl_->engine->getMetadata();
  auto segmentDesc =
      metadata ? metadata->getSegmentDescByID(segmentId) : nullptr;
  if (!segmentDesc || segmentDesc->buffers.empty()) {
    TT_LOG_ERROR(
        "[MooncakeTransferEngine] submitAndWait: no registered buffer for "
        "segment {}",
        request.target);
    return result;
  }
  const uint64_t remoteAddr =
      segmentDesc->buffers[0].addr + request.target_offset;

  const mooncake::BatchID batchId = impl_->engine->allocateBatchID(1);

  mooncake::TransferRequest entry;
  entry.opcode = request.op == TransferOp::READ
                     ? mooncake::TransferRequest::READ
                     : mooncake::TransferRequest::WRITE;
  entry.source = request.local_addr;
  entry.target_id = segmentId;
  entry.target_offset = remoteAddr;
  entry.length = request.length;

  mooncake::Status s = impl_->engine->submitTransfer(batchId, {entry});
  if (!s.ok()) {
    TT_LOG_ERROR("[MooncakeTransferEngine] submitTransfer failed: {}",
                 std::string(s.message()));
    impl_->engine->freeBatchID(batchId);
    return result;
  }

  // Block until the single task completes or fails.
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

SegmentHandle MooncakeTransferEngine::openSegment(
    const std::string& segmentName) {
  TT_LOG_WARN(
      "[MooncakeTransferEngine] openSegment({}) unavailable (built without "
      "Mooncake)",
      segmentName);
  return K_INVALID_SEGMENT;
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

#endif  // TT_TRANSPORT_WITH_MOONCAKE

}  // namespace tt::transport
