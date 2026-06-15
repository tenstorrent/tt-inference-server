// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "transport/i_transfer_engine.hpp"
#include "transport/transfer_types.hpp"

namespace tt::transport {

/**
 * @brief A loopback (in-process) transfer engine for testing.
 *
 * Simulates RDMA transfers using plain memory copies. Each "segment" is a
 * pre-registered buffer that can be read from via submitAndWait(). This lets
 * the full MigrationWorker Kafka → parse → transfer flow be exercised without
 * Mooncake or real hardware.
 *
 * Usage:
 *   auto engine = std::make_shared<LoopbackTransferEngine>();
 *   engine->init(EngineConfig{.local_server_name = "127.0.0.1:17777"});
 *   // Populate a fake "remote" segment for the worker to pull from:
 *   engine->publishSegment("127.0.0.1:17777", data.data(), data.size());
 */
class LoopbackTransferEngine : public ITransferEngine {
 public:
  LoopbackTransferEngine() = default;
  ~LoopbackTransferEngine() override = default;

  StorageMedium storageMedium() const override { return StorageMedium::HostDram; }
  std::shared_ptr<IStorageBackend> storage() const override { return nullptr; }

  bool init(const EngineConfig& /*config*/) override { return true; }

  bool registerLocalMemory(void* /*addr*/, std::size_t /*length*/) override {
    return true;
  }

  bool unregisterLocalMemory(void* /*addr*/) override { return true; }

  SegmentHandle openSegment(const std::string& segmentName) override {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = segments_.find(segmentName);
    if (it == segments_.end()) {
      return kInvalidSegment;
    }
    return it->second.handle;
  }

  TransferStatus submitAndWait(const TransferRequest& request) override {
    std::lock_guard<std::mutex> lock(mu_);
    // Find the segment by handle.
    for (const auto& [name, seg] : segments_) {
      if (seg.handle == request.target) {
        // Bounds check.
        if (request.target_offset + request.length > seg.size) {
          return {TransferState::Failed, 0};
        }
        // Simulate the RDMA read: copy from "remote" segment into local buffer.
        const auto* src =
            static_cast<const uint8_t*>(seg.data) + request.target_offset;
        std::memcpy(request.local_addr, src, request.length);
        return {TransferState::Completed, request.length};
      }
    }
    return {TransferState::Failed, 0};
  }

  // --- Test helper: publish a fake segment that workers can pull from ---

  /**
   * @brief Register a buffer as a "remote" segment accessible by name.
   *
   * The buffer must outlive the engine (or until removeSegment is called).
   */
  void publishSegment(const std::string& name, const void* data,
                      std::size_t size) {
    std::lock_guard<std::mutex> lock(mu_);
    segments_[name] = Segment{nextHandle_++, data, size};
  }

  void removeSegment(const std::string& name) {
    std::lock_guard<std::mutex> lock(mu_);
    segments_.erase(name);
  }

 private:
  struct Segment {
    SegmentHandle handle = kInvalidSegment;
    const void* data = nullptr;
    std::size_t size = 0;
  };

  std::mutex mu_;
  std::unordered_map<std::string, Segment> segments_;
  SegmentHandle nextHandle_ = 1;
};

}  // namespace tt::transport
