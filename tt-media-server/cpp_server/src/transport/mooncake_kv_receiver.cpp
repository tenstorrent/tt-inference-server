// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/mooncake_kv_receiver.hpp"

#include <utility>

#include "utils/logger.hpp"

namespace tt::transport {

MooncakeKvReceiver::MooncakeKvReceiver(std::shared_ptr<ITransferEngine> engine,
                                       IDeviceIo& device,
                                       std::string advertisedSegmentName,
                                       BounceGeometry geometry,
                                       DeviceMapFn deviceMap)
    : engine_(std::move(engine)),
      device_(device),
      advertised_segment_name_(std::move(advertisedSegmentName)),
      buffer_(geometry),
      device_map_(std::move(deviceMap)) {
  if (!engine_) {
    TT_LOG_ERROR(
        "[MooncakeKvReceiver] no engine; bounce buffer not registered");
    return;
  }
  if (!geometry.valid() || buffer_.totalBytes() == 0) {
    TT_LOG_ERROR(
        "[MooncakeKvReceiver] invalid bounce geometry (sections={}, "
        "section_size={}); not registered",
        geometry.section_count, geometry.section_size);
    return;
  }
  // totalBytes() is the geometry-derived capacity, non-zero even when the
  // allocation failed; guard the actual pointer before registering it (an
  // env-tunable geometry can request more than aligned_alloc can serve).
  if (buffer_.base() == nullptr) {
    TT_LOG_ERROR(
        "[MooncakeKvReceiver] bounce buffer allocation failed ({} bytes); not "
        "registered",
        buffer_.totalBytes());
    return;
  }
  if (!engine_->registerLocalMemory(buffer_.base(), buffer_.totalBytes())) {
    TT_LOG_ERROR(
        "[MooncakeKvReceiver] registerLocalMemory({} bytes) failed; bounce "
        "buffer "
        "not registered",
        buffer_.totalBytes());
    return;
  }
  // Remote WRITEs resolve against buffers[0] (see ITransferEngine::
  // firstRegisteredLocalBuffer), so the bounce buffer must be that slot.
  if (engine_->registeredLocalBufferCount() > 0 &&
      engine_->firstRegisteredLocalBuffer() != buffer_.base()) {
    TT_LOG_ERROR(
        "[MooncakeKvReceiver] bounce buffer is not buffers[0] after register "
        "(count={}); aborting",
        engine_->registeredLocalBufferCount());
    engine_->unregisterLocalMemory(buffer_.base());
    return;
  }
  registered_ = true;
  // Double-pin: NOC-map the SAME bounce buffer for device DMA so the drain
  // DMAs straight from the bounce buffer to device DRAM (no bounce). Null in
  // host/MMIO mode. The bounce buffer outlives the DriscDeviceIo links (this
  // receiver owns it), so the map — released with those links — never dangles.
  if (device_map_) {
    device_map_(buffer_.base(), buffer_.totalBytes());
  }
  TT_LOG_INFO(
      "[MooncakeKvReceiver] registered bounce buffer: {} sections x {} bytes = "
      "{} B, "
      "segment={}{}",
      geometry.section_count, geometry.section_size, buffer_.totalBytes(),
      advertised_segment_name_, device_map_ ? " (NOC-mapped for DRISC)" : "");
}

MooncakeKvReceiver::~MooncakeKvReceiver() {
  if (registered_ && engine_) {
    engine_->unregisterLocalMemory(buffer_.base());
  }
}

bool MooncakeKvReceiver::drainWindow(
    const std::vector<BounceSectionDescriptor>& window) {
  if (!registered_) {
    TT_LOG_ERROR(
        "[MooncakeKvReceiver] drainWindow: bounce buffer not registered");
    return false;
  }
  bool ok = true;
  for (const BounceSectionDescriptor& d : window) {
    const uint8_t* src = buffer_.sectionPtr(d.section_offset, d.size);
    if (src == nullptr) {
      TT_LOG_ERROR(
          "[MooncakeKvReceiver] descriptor section [{:#x}, +{}) escapes the "
          "bounce buffer ({} B); rejecting",
          d.section_offset, d.size, buffer_.totalBytes());
      ok = false;
      continue;
    }
    for (const DrainTarget& t : d.targets) {
      if (!device_.write(t.device, t.noc_addr, src, d.size)) {
        TT_LOG_ERROR(
            "[MooncakeKvReceiver] device write failed device={:#x} "
            "noc={:#x} size={}",
            t.device, t.noc_addr, d.size);
        ok = false;
      }
    }
  }
  return ok;
}

}  // namespace tt::transport
