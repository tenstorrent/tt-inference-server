// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "runners/llm_engine/engine/host_interface.hpp"

#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <ttnn/operations/generic/generic_op.hpp>
#include <ttnn/global_semaphore.hpp>
#include <ttnn/tensor/tensor_ops.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <tt-metalium/distributed.hpp>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace llm_engine {

namespace {

constexpr uint32_t kIntermedCbIndex = 0;

std::string get_kernel_base_path() {
    const char* base = std::getenv("TT_METAL_HOME");
    if (base) {
        return std::string{base} + "/models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/";
    }
    return "models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/";
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
    const auto core_range = tt::tt_metal::CoreRange{mesh_core_coord.core_coord, mesh_core_coord.core_coord};
    const tt::tt_metal::CoreRangeSet core_range_set{core_range};

    const uint32_t page_size = h2d_socket->get_page_size();
    const uint32_t loopback_cb_size = 1024;
    const bool pull_from_host = (h2d_socket->get_h2d_mode() == tt::tt_metal::distributed::H2DMode::DEVICE_PULL);

    auto termination_semaphore = ttnn::global_semaphore::create_global_semaphore(
        mesh_device, core_range_set, 0, tt::tt_metal::BufferType::L1);
    termination_semaphore_ = new tt::tt_metal::GlobalSemaphore{std::move(termination_semaphore)};
    mesh_device_ = mesh_device;
    initialized_ = true;

    const uint32_t term_sem_addr = ttnn::global_semaphore::get_global_semaphore_address(
        *static_cast<tt::tt_metal::GlobalSemaphore*>(termination_semaphore_));

    const std::string kernel_base = get_kernel_base_path();

    tt::tt_metal::KernelDescriptor::CompileTimeArgs h2d_ct_args = {
        h2d_socket->get_config_buffer_address(),
        term_sem_addr,
        page_size,
        static_cast<uint32_t>(pull_from_host),
        static_cast<uint32_t>(true),
        kIntermedCbIndex,
    };

    tt::tt_metal::KernelDescriptor h2d_kernel{
        .kernel_source = kernel_base + "h2d_receiver.cpp",
        .source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = core_range_set,
        .compile_time_args = h2d_ct_args,
        .config = tt::tt_metal::WriterConfigDescriptor{},
    };

    tt::tt_metal::KernelDescriptor::CompileTimeArgs d2h_ct_args = {
        d2h_socket->get_config_buffer_address(),
        term_sem_addr,
        page_size,
        static_cast<uint32_t>(true),
        kIntermedCbIndex,
    };

    tt::tt_metal::KernelDescriptor d2h_kernel{
        .kernel_source = kernel_base + "d2h_sender.cpp",
        .source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = core_range_set,
        .compile_time_args = d2h_ct_args,
        .config = tt::tt_metal::ReaderConfigDescriptor{},
    };

    tt::tt_metal::CBDescriptor intermed_cb{
        .total_size = loopback_cb_size,
        .core_ranges = core_range_set,
        .format_descriptors = {tt::tt_metal::CBFormatDescriptor{
            .buffer_index = kIntermedCbIndex,
            .data_format = tt::DataFormat::UInt32,
            .page_size = page_size,
        }},
    };

    tt::tt_metal::ProgramDescriptor program_descriptor{
        .kernels = {h2d_kernel, d2h_kernel},
        .semaphores = {},
        .cbs = {intermed_cb},
    };

    const auto device_coord = mesh_core_coord.device_coord;
    tt::tt_metal::distributed::MeshCoordinate mesh_coord{device_coord[0], device_coord[1]};
    tt::tt_metal::experimental::MeshProgramDescriptor mesh_program_desc;
    mesh_program_desc.mesh_programs.push_back({
        tt::tt_metal::distributed::MeshCoordinateRange{mesh_coord},
        program_descriptor,
    });

    auto dummy_spec = tt::tt_metal::TensorSpec(
        tt::tt_metal::Shape{0, 0, 0, 0},
        tt::tt_metal::TensorLayout{
            tt::DataType::UInt32,
            tt::tt_metal::PageConfig{tt::tt_metal::Layout::ROW_MAJOR},
            tt::tt_metal::MemoryConfig{}});
    auto dummy_tensor = tt::tt_metal::create_device_tensor(dummy_spec, mesh_device);

    std::vector<tt::tt_metal::Tensor> io_tensors = {dummy_tensor, dummy_tensor};
    ttnn::generic_op(io_tensors, mesh_program_desc);
}

void HostInterface::terminate() {
    if (!initialized_) return;

    auto* sem = static_cast<tt::tt_metal::GlobalSemaphore*>(termination_semaphore_);
    if (sem) {
        ttnn::global_semaphore::reset_global_semaphore_value(*sem, 1);
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
