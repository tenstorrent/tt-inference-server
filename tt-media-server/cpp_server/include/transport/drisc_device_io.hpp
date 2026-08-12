// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <memory>

#include "transport/i_device_io.hpp"
#include "transport/kv_cache_layout.hpp"  // LocalDeviceId
#include "transport/transfer_types.hpp"   // NocAddr

namespace tt::transport {

/**
 * @brief DRISC (DRAM-RISC) NOC-DMA device I/O — the migration data plane's
 *        IDeviceIo.
 *
 * Each device gets a persistent on-device DRISC service kernel that NOC-DMAs
 * device DRAM <-> host at ~30 GB/s. It wraps the disaggregation DriscSocketLink
 * over the coexistence UMD path (raw Cluster, no start_device / no
 * libtt_metal), so it shares a chip with a live inference engine. read/write
 * are synchronous (read_region/write_region); registerHostRegion NOC-maps a
 * host buffer for zero-copy DMA (the same buffer is also ibv_reg_mr'd by the
 * transfer engine).
 *
 * Real DRISC I/O under USE_METAL_CPP_LIB, a no-op that reports failure
 * otherwise (transport_lib builds everywhere). Launch touches hardware;
 * addDevice returns false (logged) with no HW / no service-kernel ELF, and
 * read/write then report failure for that device.
 */
class DriscDeviceIo : public IDeviceIo {
 public:
  DriscDeviceIo();
  ~DriscDeviceIo() override;

  DriscDeviceIo(const DriscDeviceIo&) = delete;
  DriscDeviceIo& operator=(const DriscDeviceIo&) = delete;

  /**
   * @brief Open `deviceId`'s coexistence UMD handle, launch a DRISC service
   *        link on a spare DRAM-RISC core, and map it to `device`.
   *
   * Requires MIGRATION_DRISC_SERVICE_ELF (the freestanding service-kernel ELF).
   * The spare core is derived from the SOC descriptor (validated, not
   * hand-mapped); MIGRATION_DRISC_BANK / MIGRATION_DRISC_SUBCHANNEL override.
   * @return true iff the device opened and the service kernel launched (HW).
   */
  bool addDevice(LocalDeviceId device, int deviceId);

  /// True if a launched DRISC link is registered for `device`.
  bool hasDevice(LocalDeviceId device) const;

  std::size_t numDevices() const;

  /// DRAM-channel count of the opened `device` (0 if not opened / no HW / no
  /// tt-metal build). Lets a caller reject KV locations whose NoC channel is
  /// out of range for the chip before any transfer.
  uint32_t numDramChannels(LocalDeviceId device) const;

  /**
   * @brief NOC-map a host region for zero-copy DMA on every launched link.
   *
   * A subsequent read/write whose host buffer falls inside `[va, va+bytes)`
   * DMAs device DRAM <-> that buffer directly (no bounce memcpy). The region
   * must outlive the links and not move. The bounce buffer / staging is
   * registered here, the same buffer the transfer engine ibv_reg_mr's.
   */
  void registerHostRegion(void* va, std::size_t bytes);

  /**
   * @brief Metal-coexistence IOVA reservations currently held (0 unless
   *        MIGRATION_IOVA_RESERVE_MB is set and devices were added on HW).
   *
   * When set, each addDevice reserves that many MiB of low IOVA (incl. IOVA 0 /
   * `pcie_base`) as a dummy NOC-mapped buffer on the chip BEFORE its DRISC
   * links pin anything, so every DRISC NOC buffer (control page, bounce arena,
   * and the registerHostRegion bounce buffer/staging) lands ABOVE it. Unset =>
   * model-first ordering (a co-resident Metal engine opens first and takes
   * pcie_base) — the default. (The disaggregation MPI-barrier coordination for
   * two co-located A/B workers sharing one IOVA domain is intentionally NOT
   * lifted — cpp_server workers are separate MPI-free processes.)
   */
  std::size_t numIovaReservations() const;

  /**
   * @brief Release the IOVA reservations, freeing pcie_base / IOVA 0 for a
   *        co-resident Metal model.
   *
   * Call from the composition root AFTER DRISC setup + host-region pinning and
   * BEFORE the engine maps pcie_base (production: before signaling READY, so
   * the model launched after takes IOVA 0).
   */
  void releaseIovaReservations();

  bool read(LocalDeviceId device, NocAddr nocAddr, std::size_t size,
            void* hostBuffer) override;
  bool write(LocalDeviceId device, NocAddr nocAddr, const void* hostBuffer,
             std::size_t size) override;

  // Async interface: DRISC launches the NOC-DMA and returns; the
  // device does the transfer in the background while the caller drives the
  // network. One request in flight per device link (launch* returns false when
  // that link is busy — the caller tryPopCompleted()s and retries).
  bool readAsync(LocalDeviceId device, NocAddr nocAddr, std::size_t size,
                 void* hostBuffer) override;
  bool writeAsync(LocalDeviceId device, NocAddr nocAddr, const void* hostBuffer,
                  std::size_t size) override;
  bool tryPopCompleted() override;
  uint32_t asyncInFlight() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tt::transport
