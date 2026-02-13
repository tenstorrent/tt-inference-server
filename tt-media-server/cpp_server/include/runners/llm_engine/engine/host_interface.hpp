// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace tt::tt_metal::distributed {
class H2DSocket;
class D2HSocket;
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace llm_engine {

/**
 * Host-Device Communication Interface (loopback mode).
 *
 * Mirrors the Python HostInterface from tt-metal models/demos/deepseek_v3_b1/
 * micro_ops/host_io/op.py. Provides bidirectional communication between host
 * and device using H2D (Host-to-Device) and D2H (Device-to-Host) sockets.
 *
 * In loopback mode: H2D receiver and D2H sender communicate via a circular
 * buffer (CB) for testing purposes. Data written to H2D is read back from D2H.
 *
 * Requires TT_METAL_HOME and tt-metal/ttnn libraries for compilation.
 */
class HostInterface {
public:
    struct Config {
        uint32_t page_size = 0;
        uint32_t loopback_cb_size = 1024;
        std::string kernel_base_path;  // e.g. ${TT_METAL_HOME}/models/demos/deepseek_v3_b1/micro_ops/host_io
    };

    HostInterface() = default;
    ~HostInterface();

    HostInterface(const HostInterface&) = delete;
    HostInterface& operator=(const HostInterface&) = delete;

    bool is_initialized() const { return initialized_; }

    void run(
        tt::tt_metal::distributed::H2DSocket* h2d_socket,
        tt::tt_metal::distributed::D2HSocket* d2h_socket,
        tt::tt_metal::distributed::MeshDevice* mesh_device);
    void terminate();
    void write(tt::tt_metal::distributed::H2DSocket* h2d_socket, const void* data, uint32_t num_pages);
    void read(tt::tt_metal::distributed::D2HSocket* d2h_socket, void* data, uint32_t num_pages);

private:
    bool initialized_ = false;
    void* termination_semaphore_ = nullptr;
    void* mesh_device_ = nullptr;
};

}  // namespace llm_engine
