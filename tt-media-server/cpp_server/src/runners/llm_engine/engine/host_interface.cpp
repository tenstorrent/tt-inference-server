// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "runners/llm_engine/engine/host_interface.hpp"

#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/kernel_types.hpp>

namespace llm_engine {

namespace {

constexpr uint32_t kIntermedCbIndex = 0;

std::string get_kernel_path(const std::string& name) {
    const char* base = std::getenv("TT_METAL_HOME");
    if (base) {
        return std::string{base} + "/models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/" + name;
    }
    return "models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/" + name;
}

}  // namespace

HostInterface::~HostInterface() {
    if (initialized_) {
        terminate();
    }
}

void HostInterface::run(
    tt::tt_metal::distributed::H2DSocket* h2d_socket,
    tt::tt_metal::distributed::D2HSocket* d2h_socket,
    tt::tt_metal::distributed::MeshDevice* mesh_device) {
    if (!h2d_socket || !d2h_socket || !mesh_device) {
        throw std::invalid_argument("HostInterface::run: h2d_socket, d2h_socket, mesh_device must be non-null");
    }

    auto core_coords = h2d_socket->get_active_cores();
    if (core_coords.empty()) {
        throw std::runtime_error("HostInterface: H2D socket has no active cores");
    }
    const auto& mesh_core_coord = core_coords[0];
    const auto core_coord = mesh_core_coord.core_coord;
    const auto core_range = tt::tt_metal::CoreRange{core_coord, core_coord};
    const tt::tt_metal::CoreRangeSet core_range_set{core_range};

    const uint32_t page_size = h2d_socket->get_page_size();
    const uint32_t loopback_cb_size = 1024;
    const bool pull_from_host =
        (h2d_socket->get_h2d_mode() == tt::tt_metal::distributed::H2DMode::DEVICE_PULL);

    auto termination_semaphore =
        tt::tt_metal::CreateGlobalSemaphore(mesh_device, core_range_set, 0, tt::tt_metal::BufferType::L1);
    termination_semaphore_ = new tt::tt_metal::GlobalSemaphore{std::move(termination_semaphore)};
    mesh_device_ = mesh_device;
    initialized_ = true;

    const uint32_t term_sem_addr =
        static_cast<uint32_t>(static_cast<tt::tt_metal::GlobalSemaphore*>(termination_semaphore_)->address());

    auto program = tt::tt_metal::CreateProgram();

    std::map<uint8_t, tt::DataFormat> cb_format{{kIntermedCbIndex, tt::DataFormat::UInt32}};
    tt::tt_metal::CircularBufferConfig cb_config{loopback_cb_size, cb_format};
    cb_config.set_page_size(kIntermedCbIndex, page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_config);

    std::vector<uint32_t> h2d_ct_args = {
        h2d_socket->get_config_buffer_address(),
        term_sem_addr,
        page_size,
        static_cast<uint32_t>(pull_from_host),
        static_cast<uint32_t>(true),
        kIntermedCbIndex,
    };
    tt::tt_metal::CreateKernel(
        program,
        get_kernel_path("h2d_receiver.cpp"),
        core_coord,
        tt::tt_metal::WriterDataMovementConfig{h2d_ct_args});

    std::vector<uint32_t> d2h_ct_args = {
        d2h_socket->get_config_buffer_address(),
        term_sem_addr,
        page_size,
        static_cast<uint32_t>(true),
        kIntermedCbIndex,
    };
    tt::tt_metal::CreateKernel(
        program,
        get_kernel_path("d2h_sender.cpp"),
        core_coord,
        tt::tt_metal::ReaderDataMovementConfig{d2h_ct_args});

    const auto device_coord = mesh_core_coord.device_coord;
    tt::tt_metal::distributed::MeshCoordinate mesh_coord{device_coord[0], device_coord[1]};
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    mesh_workload.add_program(
        tt::tt_metal::distributed::MeshCoordinateRange{mesh_coord}, std::move(program));

    tt::tt_metal::distributed::EnqueueMeshWorkload(
        mesh_device->mesh_command_queue(), mesh_workload, false);
}

void HostInterface::terminate() {
    if (!initialized_) return;

    auto* sem = static_cast<tt::tt_metal::GlobalSemaphore*>(termination_semaphore_);
    if (sem) {
        sem->reset_semaphore_value(1);
    }
    if (mesh_device_) {
        tt::tt_metal::distributed::Synchronize(
            static_cast<tt::tt_metal::distributed::MeshDevice*>(mesh_device_), std::nullopt, {});
    }
    delete static_cast<tt::tt_metal::GlobalSemaphore*>(termination_semaphore_);
    termination_semaphore_ = nullptr;
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
