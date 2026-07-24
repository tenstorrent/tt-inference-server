// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/drisc_device_io.hpp"

#include "transport/device_enumeration.hpp"
#include "utils/logger.hpp"

#ifdef USE_METAL_CPP_LIB
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Coexistence UMD device (raw Cluster, no start_device) + the DRISC socket
// backend, both from tt-llm-engine/disaggregation (on transport_lib's private
// include path under USE_METAL_CPP_LIB; see the CMake coexistence block).
#include "device_io.hpp"  // UmdDevice, DriscSpareCore, drisc_service_cores
#include "drisc_device_io.hpp"  // DriscSocketLink (disaggregation)

// UMD host-memory (sysmem) mapping for the Metal-coexistence IOVA reservation.
#include "umd/device/chip/chip.hpp"
#include "umd/device/chip_helpers/sysmem_buffer.hpp"
#include "umd/device/chip_helpers/sysmem_manager.hpp"
#endif

namespace tt::transport {

#ifdef USE_METAL_CPP_LIB

namespace {
namespace dis = tt::tt_metal::experimental::disaggregation::detail;

uint32_t envU32(const char* key, uint32_t fallback) {
  const char* v = std::getenv(key);
  if (v == nullptr || *v == '\0') return fallback;
  try {
    return static_cast<uint32_t>(std::stoul(v));
  } catch (...) {
    return fallback;
  }
}
}  // namespace

struct DriscDeviceIo::Impl {
  // One launched DRISC link per device. `views` must outlive `link` (the link's
  // Config holds a non-owning pointer into it), so PerDevice is heap-stable
  // (held by unique_ptr in the map) — a rehash must not move the vector.
  struct PerDevice {
    std::unique_ptr<dis::UmdDevice> umd;  // owns the shared Cluster ref + chip
    std::vector<std::pair<uint64_t, uint64_t>> views;  // encoded (base, offset)
    std::shared_ptr<dis::DriscSocketLink> link;
  };
  std::unordered_map<LocalDeviceId, std::unique_ptr<PerDevice>> devices;

  // Metal-coexistence IOVA reservations (dummy NOC-mapped buffers holding low
  // IOVA / pcie_base), one per unique chip. Held until
  // releaseIovaReservations(); their SysmemBuffer dtor unmaps and frees the
  // IOVA.
  std::vector<std::unique_ptr<tt::umd::SysmemBuffer>> iova_reservations;
  std::unordered_set<uint64_t> reserved_chips;
};

DriscDeviceIo::DriscDeviceIo() : impl_(std::make_unique<Impl>()) {}
DriscDeviceIo::~DriscDeviceIo() = default;

bool DriscDeviceIo::addDevice(LocalDeviceId device, int deviceId) {
  const char* elf = std::getenv("MIGRATION_DRISC_SERVICE_ELF");
  if (elf == nullptr || *elf == '\0') {
    TT_LOG_ERROR(
        "[DriscDeviceIo] addDevice(device={:#x}): MIGRATION_DRISC_SERVICE_ELF "
        "unset; cannot launch the DRISC service kernel",
        device);
    return false;
  }
  try {
    auto pd = std::make_unique<Impl::PerDevice>();
    pd->umd = dis::UmdDevice::open(deviceId);

    // Metal coexistence: reserve the low IOVA on this chip BEFORE the
    // link pins its control page/arena (so DRISC buffers land above pcie_base).
    // Env-gated; unset => model-first ordering. Once per unique chip.
    if (const char* mb = std::getenv("MIGRATION_IOVA_RESERVE_MB")) {
      const uint64_t bytes =
          static_cast<uint64_t>(std::strtoull(mb, nullptr, 10)) * 1024ull *
          1024ull;
      const uint64_t chip = static_cast<uint64_t>(pd->umd->chip_id);
      if (bytes > 0 && impl_->reserved_chips.insert(chip).second) {
        auto* sm = pd->umd->cluster
                       ->get_chip(static_cast<tt::ChipId>(pd->umd->chip_id))
                       ->get_sysmem_manager();
        auto buf = sm->allocate_sysmem_buffer(bytes, /*map_to_noc=*/true);
        const auto noc = buf->get_noc_addr();
        TT_LOG_INFO(
            "[DriscDeviceIo] IOVA-reserve chip={} {} bytes at noc={:#x} (holds "
            "pcie_base for Metal; DRISC lands above)",
            chip, bytes, noc ? *noc : 0);
        impl_->iova_reservations.push_back(std::move(buf));
      }
    }

    // Encode each dram_view's TRANSLATED core to a NOC base ((y<<6|x)<<36); the
    // address_offset is added to the local address. This is the EXACT mapping
    // the coexistence UmdDevice uses (main.cpp:288), so DRISC reads/writes the
    // same bytes a Metal/UMD-addressed peer sees.
    pd->views.reserve(pd->umd->dram_views.size());
    for (const auto& dv : pd->umd->dram_views) {
      const uint64_t base =
          (static_cast<uint64_t>((static_cast<uint32_t>(dv.core.y) << 6) |
                                 static_cast<uint32_t>(dv.core.x)))
          << 36;
      pd->views.emplace_back(base, static_cast<uint64_t>(dv.address_offset));
    }

    // Pick a spare DRISC core — one that is NOT a dram_view NOC0 endpoint, so
    // the stream-mode service kernel can't collide with that bank's DRAM-
    // controller traffic. Derived + validated from the SOC descriptor (NOT
    // hand-mapped); env overrides per field. The kernel runs on NOC0.
    const auto& soc = pd->umd->cluster->get_soc_descriptor(pd->umd->chip_id);
    const auto cores = dis::drisc_service_cores(soc, /*kernel_noc=*/0);
    if (cores.empty()) {
      TT_LOG_ERROR(
          "[DriscDeviceIo] addDevice(device={:#x}): no spare DRISC cores on "
          "chip {}",
          device, pd->umd->chip_id);
      return false;
    }

    dis::DriscSocketLink::Config cfg;
    cfg.chip = pd->umd->chip_id;
    cfg.dram_bank = envU32("MIGRATION_DRISC_BANK", cores.front().bank);
    cfg.dram_subchannel =
        envU32("MIGRATION_DRISC_SUBCHANNEL", cores.front().subchannel);
    cfg.service_elf = elf;
    cfg.dram_views = &pd->views;

    pd->link = std::make_shared<dis::DriscSocketLink>(pd->umd->cluster, cfg);
    pd->link->setup();  // launch the persistent service kernel (touches HW)

    TT_LOG_INFO(
        "[DriscDeviceIo] device={:#x} -> chip {} DRISC link (bank {}, sub {}), "
        "{} dram views",
        device, pd->umd->chip_id, cfg.dram_bank, cfg.dram_subchannel,
        pd->views.size());
    impl_->devices[device] = std::move(pd);
    return true;
  } catch (const std::exception& e) {
    TT_LOG_ERROR(
        "[DriscDeviceIo] addDevice(device={:#x}, deviceId={}) failed: {}",
        device, deviceId, e.what());
    return false;
  }
}

bool DriscDeviceIo::hasDevice(LocalDeviceId device) const {
  const auto it = impl_->devices.find(device);
  return it != impl_->devices.end() && it->second && it->second->link;
}

std::size_t DriscDeviceIo::numDevices() const { return impl_->devices.size(); }

uint32_t DriscDeviceIo::numDramChannels(LocalDeviceId device) const {
  const auto it = impl_->devices.find(device);
  if (it == impl_->devices.end() || !it->second || !it->second->umd) return 0u;
  return it->second->umd->num_dram_channels();
}

// Enumerate every visible chip once (open by 0-based index, read its ASIC
// unique_id) and return unique_id -> index. Same eager enumeration the
// disaggregation worker does in open_all_devices; the handles are dropped at
// return (the shared per-process Cluster stays alive). Lets the composition
// root translate a device-map unique_id into the index UmdDevice::open() /
// addDevice() expect.
std::unordered_map<uint64_t, int> enumerateDevicesByUniqueId() {
  std::unordered_map<uint64_t, int> byUniqueId;
  const int count = dis::UmdDevice::count();
  for (int i = 0; i < count; ++i) {
    try {
      auto dev = dis::UmdDevice::open(i);
      if (dev == nullptr) {
        TT_LOG_WARN("[device_enum] open({}) returned null", i);
        continue;
      }
      const uint64_t uid = dev->unique_id();
      const auto [it, inserted] = byUniqueId.emplace(uid, i);
      if (!inserted) {
        TT_LOG_WARN(
            "[device_enum] duplicate unique_id {} at indices {} and {}; "
            "keeping {}",
            uid, it->second, i, it->second);
      }
    } catch (const std::exception& e) {
      TT_LOG_WARN("[device_enum] open({}) failed: {}", i, e.what());
    }
  }
  TT_LOG_INFO(
      "[device_enum] enumerated {} of {} visible device(s) by unique_id",
      byUniqueId.size(), count);
  return byUniqueId;
}

void DriscDeviceIo::registerHostRegion(void* va, std::size_t bytes) {
  for (auto& [device, pd] : impl_->devices) {
    (void)device;
    if (pd && pd->link) pd->link->register_host_region(va, bytes);
  }
}

std::size_t DriscDeviceIo::numIovaReservations() const {
  return impl_->iova_reservations.size();
}

void DriscDeviceIo::releaseIovaReservations() {
  if (!impl_->iova_reservations.empty()) {
    TT_LOG_INFO(
        "[DriscDeviceIo] releasing {} IOVA reservation(s) (pcie_base freed for "
        "a "
        "co-resident Metal model)",
        impl_->iova_reservations.size());
  }
  impl_->iova_reservations
      .clear();  // SysmemBuffer dtor unmaps + frees the IOVA
  impl_->reserved_chips.clear();
}

bool DriscDeviceIo::read(LocalDeviceId device, NocAddr nocAddr,
                         std::size_t size, void* hostBuffer) {
  const auto it = impl_->devices.find(device);
  if (it == impl_->devices.end() || !it->second || !it->second->link) {
    TT_LOG_ERROR("[DriscDeviceIo] read: no DRISC link for device {:#x}",
                 device);
    return false;
  }
  if (size == 0) return true;
  if (hostBuffer == nullptr) {
    TT_LOG_ERROR("[DriscDeviceIo] read: null hostBuffer (device={:#x})",
                 device);
    return false;
  }
  if (size > UINT32_MAX) {
    TT_LOG_ERROR("[DriscDeviceIo] read: size {} exceeds 32-bit limit", size);
    return false;
  }
  try {
    it->second->link->read_region(nocChannel(nocAddr), nocLocalAddr(nocAddr),
                                  static_cast<uint32_t>(size), hostBuffer);
    return true;
  } catch (const std::exception& e) {
    TT_LOG_ERROR(
        "[DriscDeviceIo] read(device={:#x}, channel={}, local={:#x}, size={}) "
        "failed: {}",
        device, nocChannel(nocAddr), nocLocalAddr(nocAddr), size, e.what());
    return false;
  }
}

bool DriscDeviceIo::write(LocalDeviceId device, NocAddr nocAddr,
                          const void* hostBuffer, std::size_t size) {
  const auto it = impl_->devices.find(device);
  if (it == impl_->devices.end() || !it->second || !it->second->link) {
    TT_LOG_ERROR("[DriscDeviceIo] write: no DRISC link for device {:#x}",
                 device);
    return false;
  }
  if (size == 0) return true;
  if (hostBuffer == nullptr) {
    TT_LOG_ERROR("[DriscDeviceIo] write: null hostBuffer (device={:#x})",
                 device);
    return false;
  }
  if (size > UINT32_MAX) {
    TT_LOG_ERROR("[DriscDeviceIo] write: size {} exceeds 32-bit limit", size);
    return false;
  }
  try {
    it->second->link->write_region(nocChannel(nocAddr), nocLocalAddr(nocAddr),
                                   static_cast<uint32_t>(size), hostBuffer);
    return true;
  } catch (const std::exception& e) {
    TT_LOG_ERROR(
        "[DriscDeviceIo] write(device={:#x}, channel={}, local={:#x}, size={}) "
        "failed: {}",
        device, nocChannel(nocAddr), nocLocalAddr(nocAddr), size, e.what());
    return false;
  }
}

bool DriscDeviceIo::readAsync(LocalDeviceId device, NocAddr nocAddr,
                              std::size_t size, void* hostBuffer) {
  const auto it = impl_->devices.find(device);
  if (it == impl_->devices.end() || !it->second || !it->second->link) {
    return false;
  }
  if (size == 0) return true;
  if (hostBuffer == nullptr || size > UINT32_MAX) return false;
  try {
    // false here means the link is BUSY (one request in flight) — the caller
    // tryPopCompleted()s and retries. A hard error throws.
    return it->second->link->launch_read(
        nocChannel(nocAddr), nocLocalAddr(nocAddr), static_cast<uint32_t>(size),
        hostBuffer);
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[DriscDeviceIo] readAsync(device={:#x}) failed: {}", device,
                 e.what());
    return false;
  }
}

bool DriscDeviceIo::writeAsync(LocalDeviceId device, NocAddr nocAddr,
                               const void* hostBuffer, std::size_t size) {
  const auto it = impl_->devices.find(device);
  if (it == impl_->devices.end() || !it->second || !it->second->link) {
    return false;
  }
  if (size == 0) return true;
  if (hostBuffer == nullptr || size > UINT32_MAX) return false;
  try {
    return it->second->link->launch_write(
        nocChannel(nocAddr), nocLocalAddr(nocAddr), static_cast<uint32_t>(size),
        hostBuffer);
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[DriscDeviceIo] writeAsync(device={:#x}) failed: {}", device,
                 e.what());
    return false;
  }
}

bool DriscDeviceIo::tryPopCompleted() {
  bool any = false;
  for (auto& [device, pd] : impl_->devices) {
    (void)device;
    if (pd && pd->link && pd->link->busy()) {
      try {
        if (pd->link->poll()) any = true;  // poll() retires a finished op
      } catch (const std::exception& e) {
        TT_LOG_ERROR("[DriscDeviceIo] poll failed: {}", e.what());
      }
    }
  }
  return any;
}

uint32_t DriscDeviceIo::asyncInFlight() const {
  uint32_t n = 0;
  for (const auto& [device, pd] : impl_->devices) {
    (void)device;
    if (pd && pd->link && pd->link->busy()) ++n;
  }
  return n;
}

#else  // !USE_METAL_CPP_LIB — no-op fallback (transport_lib builds everywhere)

struct DriscDeviceIo::Impl {};

DriscDeviceIo::DriscDeviceIo() : impl_(std::make_unique<Impl>()) {}
DriscDeviceIo::~DriscDeviceIo() = default;

bool DriscDeviceIo::addDevice(LocalDeviceId device, int /*deviceId*/) {
  TT_LOG_WARN(
      "[DriscDeviceIo] addDevice(device={:#x}) unavailable (built without "
      "tt-metal / DRISC; USE_METAL_CPP_LIB undefined)",
      device);
  return false;
}

bool DriscDeviceIo::hasDevice(LocalDeviceId /*device*/) const { return false; }
std::size_t DriscDeviceIo::numDevices() const { return 0; }
uint32_t DriscDeviceIo::numDramChannels(LocalDeviceId /*device*/) const {
  return 0u;
}

// No devices without tt-metal: return an empty map so a non-empty DeviceMap
// resolves to a clean miss (the composition root then fails without opening).
std::unordered_map<uint64_t, int> enumerateDevicesByUniqueId() {
  TT_LOG_WARN(
      "[device_enum] enumerate by unique_id unavailable (built without "
      "tt-metal); returning empty map");
  return {};
}

void DriscDeviceIo::registerHostRegion(void* /*va*/, std::size_t /*bytes*/) {}
std::size_t DriscDeviceIo::numIovaReservations() const { return 0; }
void DriscDeviceIo::releaseIovaReservations() {}

bool DriscDeviceIo::read(LocalDeviceId device, NocAddr /*nocAddr*/,
                         std::size_t /*size*/, void* /*hostBuffer*/) {
  TT_LOG_WARN("[DriscDeviceIo] read(device={:#x}) unavailable (no DRISC build)",
              device);
  return false;
}

bool DriscDeviceIo::write(LocalDeviceId device, NocAddr /*nocAddr*/,
                          const void* /*hostBuffer*/, std::size_t /*size*/) {
  TT_LOG_WARN(
      "[DriscDeviceIo] write(device={:#x}) unavailable (no DRISC build)",
      device);
  return false;
}

bool DriscDeviceIo::readAsync(LocalDeviceId device, NocAddr nocAddr,
                              std::size_t size, void* hostBuffer) {
  return read(device, nocAddr, size, hostBuffer);  // no-op read -> false
}
bool DriscDeviceIo::writeAsync(LocalDeviceId device, NocAddr nocAddr,
                               const void* hostBuffer, std::size_t size) {
  return write(device, nocAddr, hostBuffer, size);
}
bool DriscDeviceIo::tryPopCompleted() { return false; }
uint32_t DriscDeviceIo::asyncInFlight() const { return 0; }

#endif  // USE_METAL_CPP_LIB

}  // namespace tt::transport
