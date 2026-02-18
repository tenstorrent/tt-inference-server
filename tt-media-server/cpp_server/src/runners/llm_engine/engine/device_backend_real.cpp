#include "llm_engine/engine/device_backend.hpp"
#include "llm_engine/engine/host_interface.hpp"
#include "llm_engine/engine/sequence.hpp"

#include <cstring>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/distributed.hpp>

namespace llm_engine {

namespace {

constexpr uint32_t kFifoSize = 1024 * 1024;

class RealDeviceBackend : public IDeviceBackend {
 public:
  explicit RealDeviceBackend(const Config&) {
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

    host_io_ = std::make_unique<HostInterface>();
  }

  void init() override {
    host_io_->run(h2d_socket_.get(), d2h_socket_.get(), mesh_device_.get());
  }

  void write(const void* data, uint32_t num_pages) override {
    host_io_->write(h2d_socket_.get(), data, num_pages);
  }

  bool read(void* data, uint32_t num_pages) override {
    host_io_->read(d2h_socket_.get(), data, num_pages);
    return true;
  }

  void terminate() override {
    host_io_->terminate();
  }

 private:
  std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;
  std::unique_ptr<tt::tt_metal::distributed::H2DSocket> h2d_socket_;
  std::unique_ptr<tt::tt_metal::distributed::D2HSocket> d2h_socket_;
  std::unique_ptr<HostInterface> host_io_;
};

}  // namespace

std::unique_ptr<IDeviceBackend> make_device_backend_real(const Config& config) {
  return std::make_unique<RealDeviceBackend>(config);
}

}  // namespace llm_engine
