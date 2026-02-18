#include "llm_engine/engine/device_backend.hpp"
#include "llm_engine/config.hpp"

namespace llm_engine {

std::unique_ptr<IDeviceBackend> make_device_backend_real(const Config& config);
std::unique_ptr<IDeviceBackend> make_device_backend_mock(const Config& config);

std::unique_ptr<IDeviceBackend> make_device_backend(const Config& config, bool use_real_device) {
  if (use_real_device) {
    return make_device_backend_real(config);
  }
  return make_device_backend_mock(config);
}

}  // namespace llm_engine
