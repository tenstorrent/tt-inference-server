#include "runners/llm_runner/backend/device_backend.hpp"
#include "runners/llm_runner/config.hpp"

#include <iostream>

namespace llm_engine::backend {

std::unique_ptr<IDeviceBackend> make_device_backend_mock(const Config& config);
std::unique_ptr<IDeviceBackend> make_device_backend_ttrun(const Config& config);

std::unique_ptr<IDeviceBackend> make_device_backend(const Config& config) {
  switch (config.device) {
    case DeviceBackend::TtRun:
      return make_device_backend_ttrun(config);
    case DeviceBackend::Mock:
      return make_device_backend_mock(config);
    default:
      throw std::invalid_argument("Invalid device backend");
  }
}

}  // namespace llm_engine
