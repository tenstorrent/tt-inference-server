#pragma once

#include <cstdint>
#include <memory>

#include "llm_engine/config.hpp"

namespace llm_engine {

/**
 * Abstraction for host–device communication (init, write to device, read from device).
 * Real implementation uses TT device and sockets; mock implementation echoes written
 * data back as read data for testing without hardware.
 */
class IDeviceBackend {
 public:
  virtual ~IDeviceBackend() = default;
  virtual void init() = 0;
  virtual void write(const void* data, uint32_t num_pages) = 0;
  /** Returns true if data was read, false on shutdown or no data. */
  virtual bool read(void* data, uint32_t num_pages) = 0;
  virtual void terminate() = 0;
};

std::unique_ptr<IDeviceBackend> make_device_backend(const Config& config, bool use_real_device);

}  // namespace llm_engine
