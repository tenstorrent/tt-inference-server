#include "llm_engine/engine/device_backend.hpp"
#include "llm_engine/engine/sequence.hpp"

#include <cstring>
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

constexpr uint32_t kFifoSize = 1024 * 1024;
constexpr uint32_t kLoopbackCbSize = 1024;
const char* kH2dReceiverKernelPath =
    "models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/h2d_receiver.cpp";
const char* kD2hSenderKernelPath =
    "models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/d2h_sender.cpp";
constexpr uint8_t kIntermedCbIndex = 0;

class SocketsDeviceBackend : public IDeviceBackend {
 public:
  explicit SocketsDeviceBackend(const Config&) {
    mesh_device_ = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    tt::tt_metal::distributed::MeshCoordinate device_coord{0, 0};
    tt::tt_metal::CoreCoord core_coord{0, 0};
    tt::tt_metal::distributed::MeshCoreCoord socket_core{device_coord, core_coord};

    h2d_socket_ = std::make_unique<tt::tt_metal::distributed::H2DSocket>(
        mesh_device_,
        socket_core,
        tt::tt_metal::BufferType::L1,
        kFifoSize,
        tt::tt_metal::distributed::H2DMode::HOST_PUSH);
    h2d_socket_->set_page_size(Sequence::page_size());

    d2h_socket_ = std::make_unique<tt::tt_metal::distributed::D2HSocket>(
        mesh_device_, socket_core, kFifoSize);
    d2h_socket_->set_page_size(Sequence::page_size());
  }

  void init() override {
    auto core_coords = h2d_socket_->get_active_cores();
    if (core_coords.empty()) {
      throw std::runtime_error("SocketsDeviceBackend: H2D socket has no active cores");
    }
    const auto& mesh_core_coord = core_coords[0];
    const auto core_coord = mesh_core_coord.core_coord;

    const uint32_t page_size = h2d_socket_->get_page_size();
    const bool pull_from_host =
        (h2d_socket_->get_h2d_mode() == tt::tt_metal::distributed::H2DMode::DEVICE_PULL);

    tt::tt_metal::CoreRangeSet core_range_set{tt::tt_metal::CoreRange{core_coord, core_coord}};
    termination_semaphore_.emplace(
        static_cast<tt::tt_metal::IDevice*>(mesh_device_.get()),
        core_range_set,
        0,
        tt::tt_metal::BufferType::L1);
    const uint32_t semaphore_addr = static_cast<uint32_t>(termination_semaphore_->address());

    auto program = tt::tt_metal::CreateProgram();

    tt::tt_metal::CircularBufferConfig cb_config{kLoopbackCbSize};
    cb_config.index(kIntermedCbIndex)
        .set_page_size(page_size)
        .set_data_format(tt::DataFormat::UInt32);
    tt::tt_metal::CreateCircularBuffer(program, core_coord, cb_config);

    std::vector<uint32_t> h2d_compile_args = {
        h2d_socket_->get_config_buffer_address(),
        semaphore_addr,
        page_size,
        static_cast<uint32_t>(pull_from_host),
        static_cast<uint32_t>(true),
        kIntermedCbIndex,
    };
    tt::tt_metal::CreateKernel(
        program,
        kH2dReceiverKernelPath,
        core_coord,
        tt::tt_metal::WriterDataMovementConfig{h2d_compile_args});

    std::vector<uint32_t> d2h_compile_args = {
        d2h_socket_->get_config_buffer_address(),
        semaphore_addr,
        page_size,
        static_cast<uint32_t>(true),
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
        mesh_device_->mesh_command_queue(), mesh_workload, false);
    initialized_ = true;
  }

  void write(const void* data, uint32_t num_pages) override {
    h2d_socket_->write(const_cast<void*>(data), num_pages);
  }

  bool read(void* data, uint32_t num_pages) override {
    d2h_socket_->read(data, num_pages);
    return true;
  }

  void terminate() override {
    if (!initialized_) return;
    if (termination_semaphore_) {
      termination_semaphore_->reset_semaphore_value(1);
    }
    if (mesh_device_) {
      tt::tt_metal::distributed::Finish(mesh_device_->mesh_command_queue());
    }
    termination_semaphore_.reset();
    initialized_ = false;
  }

 private:
  bool initialized_ = false;
  std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;
  std::unique_ptr<tt::tt_metal::distributed::H2DSocket> h2d_socket_;
  std::unique_ptr<tt::tt_metal::distributed::D2HSocket> d2h_socket_;
  std::optional<tt::tt_metal::GlobalSemaphore> termination_semaphore_;
};

}  // namespace

std::unique_ptr<IDeviceBackend> make_device_backend_sockets(const Config& config) {
  return std::make_unique<SocketsDeviceBackend>(config);
}

}  // namespace llm_engine
