// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "runners/llm_engine/engine/host_interface.hpp"

#include <stdexcept>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace llm_engine {

namespace {

constexpr uint32_t kLoopbackCbSize = 1024;
const char* kH2dReceiverKernelPath =
    "models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/h2d_receiver.cpp";
const char* kD2hSenderKernelPath =
    "models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/d2h_sender.cpp";
constexpr uint8_t kIntermedCbIndex = 0;

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

    const uint32_t page_size = h2d_socket->get_page_size();
    const bool pull_from_host =
        (h2d_socket->get_h2d_mode() == tt::tt_metal::distributed::H2DMode::DEVICE_PULL);

    tt::tt_metal::CoreRangeSet core_range_set{tt::tt_metal::CoreRange{core_coord, core_coord}};
    // GlobalSemaphore must get MeshDevice* (as IDevice*) so AnyBuffer::create builds a MeshBuffer;
    // get_device(coord) can return a non-MeshDevice and would create a plain Buffer, then
    // reset_semaphore_value would dereference null in mesh_buffer->device().
    termination_semaphore_.emplace(
        static_cast<tt::tt_metal::IDevice*>(mesh_device),
        core_range_set,
        0,
        tt::tt_metal::BufferType::L1);
    const uint32_t semaphore_addr = static_cast<uint32_t>(termination_semaphore_->address());

    mesh_device_ = mesh_device;
    initialized_ = true;

    auto program = tt::tt_metal::CreateProgram();

    tt::tt_metal::CircularBufferConfig cb_config{kLoopbackCbSize};
    cb_config.index(kIntermedCbIndex)
        .set_page_size(page_size)
        .set_data_format(tt::DataFormat::UInt32);
    tt::tt_metal::CreateCircularBuffer(program, core_coord, cb_config);

    std::vector<uint32_t> h2d_compile_args = {
        h2d_socket->get_config_buffer_address(),
        semaphore_addr,
        page_size,
        static_cast<uint32_t>(pull_from_host),
        static_cast<uint32_t>(true),   // loopback_mode
        kIntermedCbIndex,
    };
    tt::tt_metal::CreateKernel(
        program,
        kH2dReceiverKernelPath,
        core_coord,
        tt::tt_metal::WriterDataMovementConfig{h2d_compile_args});

    std::vector<uint32_t> d2h_compile_args = {
        d2h_socket->get_config_buffer_address(),
        semaphore_addr,
        page_size,
        static_cast<uint32_t>(true),   // loopback_mode
        kIntermedCbIndex,
    };
    tt::tt_metal::CreateKernel(
        program,
        kD2hSenderKernelPath,
        core_coord,
        tt::tt_metal::ReaderDataMovementConfig{d2h_compile_args});

    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    mesh_workload.add_program(
        tt::tt_metal::distributed::MeshCoordinateRange{mesh_core_coord.device_coord},
        std::move(program));

    tt::tt_metal::distributed::EnqueueMeshWorkload(
        mesh_device->mesh_command_queue(), mesh_workload, false);
}

void HostInterface::terminate() {
    if (!initialized_) return;

    if (termination_semaphore_) {
        termination_semaphore_->reset_semaphore_value(1);
    }
    if (mesh_device_) {
        auto* dev = static_cast<tt::tt_metal::distributed::MeshDevice*>(mesh_device_);
        tt::tt_metal::distributed::Finish(dev->mesh_command_queue());
    }
    mesh_device_ = nullptr;
    termination_semaphore_.reset();
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
