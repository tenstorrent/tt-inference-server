#include "runners/llm_runner/device_backend.hpp"
#include "runners/llm_runner/config.hpp"

#include <iostream>

namespace llm_engine {

std::unique_ptr<IDeviceBackend> make_device_backend_mock(const Config& config);

#ifdef USE_METAL_CPP_LIB
std::unique_ptr<IDeviceBackend> make_device_backend_sockets(const Config& config);
#endif

std::unique_ptr<IDeviceBackend> make_device_backend(const Config& config) {
#ifdef USE_METAL_CPP_LIB
  if (config.device == DeviceBackend::Sockets) {
    std::cout << "Using sockets device backend" << std::endl;
    return make_device_backend_sockets(config);
  }
#endif
  std::cout << "Using mock device backend" << std::endl;
  return make_device_backend_mock(config);
}

}  // namespace llm_engine
