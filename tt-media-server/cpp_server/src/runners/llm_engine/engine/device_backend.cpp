#include "llm_engine/engine/device_backend.hpp"
#include "llm_engine/config.hpp"

namespace llm_engine {

std::unique_ptr<IDeviceBackend> make_device_backend_sockets(const Config& config);
std::unique_ptr<IDeviceBackend> make_device_backend_mock(const Config& config);

std::unique_ptr<IDeviceBackend> make_device_backend(const Config& config) {
  if (config.device == DeviceBackend::Sockets) {
    std::cout << "Using sockets device backend" << std::endl;
    return make_device_backend_sockets(config);
  }
  std::cout << "Using mock device backend" << std::endl;
  return make_device_backend_mock(config);
}

}  // namespace llm_engine
