// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// Build + link + graceful-degradation coverage for DriscDeviceIo.
// The DRISC data path (launch a service kernel, NOC-DMA device DRAM) needs real
// hardware, so this does NOT exercise a transfer — that is the HW gate.
// It asserts the no-hardware contract: an unopened device reports failure
// cleanly (never crashes), and addDevice fails (no ELF / no HW) rather than
// throwing. Linking this pulls the DRISC translation units, so it also verifies
// B1: drisc_device_io.cpp + multi_device_reader/writer.cpp link into
// transport_lib (under USE_METAL_CPP_LIB) with no undefined symbols.

#include "transport/drisc_device_io.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "transport/transfer_types.hpp"

namespace tt::transport {
namespace {

// A device that was never added reports failure on read/write (no crash),
// regardless of build config.
TEST(DriscDeviceIo, UnknownDeviceFailsCleanly) {
  DriscDeviceIo io;
  EXPECT_EQ(io.numDevices(), 0u);
  EXPECT_FALSE(io.hasDevice(0));

  std::vector<uint8_t> buf(64, 0);
  EXPECT_FALSE(
      io.read(/*device=*/0, makeNocAddr(0, 0x1000), buf.size(), buf.data()));
  EXPECT_FALSE(
      io.write(/*device=*/0, makeNocAddr(0, 0x1000), buf.data(), buf.size()));

  // Any other unopened device id also reports absent (not just id 0).
  EXPECT_FALSE(io.hasDevice(7));
}

// registerHostRegion with no launched links is a harmless no-op.
TEST(DriscDeviceIo, RegisterHostRegionNoDevicesIsNoop) {
  DriscDeviceIo io;
  std::vector<uint8_t> region(4096, 0);
  io.registerHostRegion(region.data(), region.size());  // must not crash
  EXPECT_EQ(io.numDevices(), 0u);
}

// addDevice must fail gracefully (return false, not throw) when it cannot
// launch — no service-kernel ELF and/or no hardware in this environment. This
// is the same degraded path the no-tt-metal fallback takes, so the assertion
// holds in every build config.
TEST(DriscDeviceIo, AddDeviceWithoutHardwareOrElfFailsGracefully) {
  ::unsetenv("MIGRATION_DRISC_SERVICE_ELF");
  DriscDeviceIo io;
  EXPECT_FALSE(io.addDevice(/*device=*/0, /*deviceId=*/0));
  EXPECT_FALSE(io.hasDevice(0));
  EXPECT_EQ(io.numDevices(), 0u);
}

// IOVA reservations: with the reservation env unset (the model-first
// default) and no devices added, none are held and release is a safe no-op.
// The actual reservation is HW (needs an opened chip's sysmem manager).
TEST(DriscDeviceIo, IovaReservationsEmptyByDefault) {
  ::unsetenv("MIGRATION_IOVA_RESERVE_MB");
  DriscDeviceIo io;
  EXPECT_EQ(io.numIovaReservations(), 0u);
  io.releaseIovaReservations();  // must not crash
  EXPECT_EQ(io.numIovaReservations(), 0u);
}

}  // namespace
}  // namespace tt::transport
