// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for HostInterface H2D/D2H socket loopback.
 * Mirrors tt-metal models/demos/deepseek_v3_b1/tests/unit_tests/test_host_io.py
 */

#include "runners/llm_engine/engine/host_interface.hpp"

#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>

namespace llm_engine {
namespace {

constexpr uint32_t kPageSize = 64;
constexpr uint32_t kFifoSize = 1024;
constexpr uint32_t kNumIterations = 64;

void ensure_tt_metal_runtime_root() {
    if (std::getenv("TT_METAL_RUNTIME_ROOT") == nullptr) {
        const char* home = std::getenv("TT_METAL_HOME");
        if (home) {
            setenv("TT_METAL_RUNTIME_ROOT", home, 1);
        } else {
            tt::tt_metal::SetRootDir(".");
        }
    }
}

}  // namespace

void run_host_io_loopback(tt::tt_metal::distributed::H2DMode h2d_mode) {
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);

    tt::tt_metal::distributed::MeshCoordinate device_coord{0, 0};
    tt::tt_metal::CoreCoord core_coord{0, 0};
    tt::tt_metal::distributed::MeshCoreCoord socket_core{device_coord, core_coord};

    auto h2d_socket = std::make_unique<tt::tt_metal::distributed::H2DSocket>(
        mesh_device,
        socket_core,
        tt::tt_metal::BufferType::L1,
        kFifoSize,
        h2d_mode);
    h2d_socket->set_page_size(kPageSize);

    auto d2h_socket = std::make_unique<tt::tt_metal::distributed::D2HSocket>(
        mesh_device, socket_core, kFifoSize);
    d2h_socket->set_page_size(kPageSize);

    HostInterface host_io;
    host_io.run(h2d_socket.get(), d2h_socket.get(), mesh_device.get());

    std::vector<uint32_t> input(kPageSize / sizeof(uint32_t));
    std::vector<uint32_t> output(kPageSize / sizeof(uint32_t));

    for (uint32_t i = 0; i < kNumIterations; ++i) {
        for (size_t j = 0; j < input.size(); ++j) {
            input[j] = static_cast<uint32_t>(i * input.size() + j);
        }
        std::memset(output.data(), 0, output.size() * sizeof(uint32_t));

        host_io.write(h2d_socket.get(), input.data(), 1);
        host_io.read(d2h_socket.get(), output.data(), 1);

        for (size_t j = 0; j < input.size(); ++j) {
            ASSERT_EQ(input[j], output[j])
                << "H2D → D2H loopback data mismatch at iteration " << i << ", element " << j;
        }
    }

    host_io.terminate();
}

TEST(HostIOLoopback, LoopbackHOST_PUSH) {
    ensure_tt_metal_runtime_root();
    if (tt::tt_metal::GetNumAvailableDevices() == 0) {
        GTEST_SKIP() << "No devices available; skipping HostIO loopback test";
    }
    run_host_io_loopback(tt::tt_metal::distributed::H2DMode::HOST_PUSH);
}

TEST(HostIOLoopback, LoopbackDEVICE_PULL) {
    ensure_tt_metal_runtime_root();
    if (tt::tt_metal::GetNumAvailableDevices() == 0) {
        GTEST_SKIP() << "No devices available; skipping HostIO loopback test";
    }
    run_host_io_loopback(tt::tt_metal::distributed::H2DMode::DEVICE_PULL);
}

}  // namespace llm_engine
