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
 * Uses the pcie_socket_loopback kernel from tt-metal test suite to bridge
 * H2D and D2H sockets directly in a single data-movement kernel.
 * Data written to H2D is read back from D2H for the configured number
 * of iterations, after which the kernel exits naturally.
 */
class HostInterface {
public:
    HostInterface() = default;
    ~HostInterface();

    HostInterface(const HostInterface&) = delete;
    HostInterface& operator=(const HostInterface&) = delete;

    bool is_initialized() const { return initialized_; }

    void run(
        tt::tt_metal::distributed::H2DSocket* h2d_socket,
        tt::tt_metal::distributed::D2HSocket* d2h_socket,
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        uint32_t num_iterations);
    void terminate();
    void write(tt::tt_metal::distributed::H2DSocket* h2d_socket, const void* data, uint32_t num_pages);
    void read(tt::tt_metal::distributed::D2HSocket* d2h_socket, void* data, uint32_t num_pages);

private:
    bool initialized_ = false;
    void* mesh_device_ = nullptr;
};

}  // namespace llm_engine
