// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <tt-metalium/global_semaphore.hpp>

namespace tt::tt_metal::distributed {
class H2DSocket;
class D2HSocket;
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace llm_engine {

/**
 * Host-Device Communication Interface (loopback mode).
 *
 * Uses H2D receiver and D2H sender kernels (same as deepseek_v3_b1 host_io op)
 * with a local circular buffer for loopback. Data written to H2D is forwarded
 * to the CB; D2H sender reads from the CB and pushes to D2H. Both kernels
 * run until terminate() sets the global semaphore, then exit cleanly.
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
        tt::tt_metal::distributed::MeshDevice* mesh_device);
    void terminate();
    void write(tt::tt_metal::distributed::H2DSocket* h2d_socket, const void* data, uint32_t num_pages);
    void read(tt::tt_metal::distributed::D2HSocket* d2h_socket, void* data, uint32_t num_pages);

private:
    bool initialized_ = false;
    void* mesh_device_ = nullptr;
    std::optional<tt::tt_metal::GlobalSemaphore> termination_semaphore_;
};

}  // namespace llm_engine
