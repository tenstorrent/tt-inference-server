// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "runners/llm_engine/engine/host_interface.hpp"

#include <iostream>
#include <stdexcept>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>

namespace llm_engine {

namespace {

const char* kLoopbackKernelPath =
    "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_loopback.cpp";

}  // namespace

HostInterface::~HostInterface() {
    if (initialized_) {
        terminate();
    }
}

void HostInterface::run(
    tt::tt_metal::distributed::H2DSocket* h2d_socket,
    tt::tt_metal::distributed::D2HSocket* d2h_socket,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint32_t num_iterations) {
    if (!h2d_socket || !d2h_socket || !mesh_device) {
        throw std::invalid_argument("HostInterface::run: h2d_socket, d2h_socket, mesh_device must be non-null");
    }

    auto core_coords = h2d_socket->get_active_cores();
    if (core_coords.empty()) {
        throw std::runtime_error("HostInterface: H2D socket has no active cores");
    }
    const auto& mesh_core_coord = core_coords[0];
    const auto core_coord = mesh_core_coord.core_coord;

    const uint32_t page_size = h2d_socket->get_page_size();
    const bool pull_from_host =
        (h2d_socket->get_h2d_mode() == tt::tt_metal::distributed::H2DMode::DEVICE_PULL);

    mesh_device_ = mesh_device;
    initialized_ = true;

    auto program = tt::tt_metal::CreateProgram();

    std::cout << "[host_interface] CreateKernel pcie_socket_loopback..." << std::endl;
    tt::tt_metal::CreateKernel(
        program,
        kLoopbackKernelPath,
        core_coord,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(h2d_socket->get_config_buffer_address()),
                static_cast<uint32_t>(d2h_socket->get_config_buffer_address()),
                page_size,
                page_size,
                num_iterations,
                static_cast<uint32_t>(pull_from_host),
            }});
    std::cout << "[host_interface] CreateKernel done" << std::endl;

    const auto device_coord = mesh_core_coord.device_coord;
    tt::tt_metal::distributed::MeshCoordinate mesh_coord{device_coord[0], device_coord[1]};
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    mesh_workload.add_program(
        tt::tt_metal::distributed::MeshCoordinateRange{mesh_coord}, std::move(program));

    std::cout << "[host_interface] EnqueueMeshWorkload..." << std::endl;
    tt::tt_metal::distributed::EnqueueMeshWorkload(
        mesh_device->mesh_command_queue(), mesh_workload, false);
    std::cout << "[host_interface] EnqueueMeshWorkload done" << std::endl;
}

void HostInterface::terminate() {
    if (!initialized_) return;

    if (mesh_device_) {
        auto* dev = static_cast<tt::tt_metal::distributed::MeshDevice*>(mesh_device_);
        std::cout << "[host_interface] Finish..." << std::endl;
        tt::tt_metal::distributed::Finish(dev->mesh_command_queue());
        std::cout << "[host_interface] Finish done" << std::endl;
    }
    mesh_device_ = nullptr;
    initialized_ = false;
}

void HostInterface::write(
    tt::tt_metal::distributed::H2DSocket* h2d_socket,
    const void* data,
    uint32_t num_pages) {
    if (h2d_socket) {
        h2d_socket->write(const_cast<void*>(data), num_pages);
    }
}

void HostInterface::read(
    tt::tt_metal::distributed::D2HSocket* d2h_socket,
    void* data,
    uint32_t num_pages) {
    if (d2h_socket) {
        d2h_socket->read(data, num_pages);
    }
}

}  // namespace llm_engine
