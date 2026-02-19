#pragma once

#include <memory>

#include "llm_engine/config.hpp"
#include "llm_engine/engine/sequence.hpp"

namespace llm_engine {

/**
 * Abstraction for host–device communication (init, write sequence, read decode result).
 * Real implementation uses TT device and sockets; mock queues DecodeResult without serialization.
 */
class IDeviceBackend {
 public:
  virtual ~IDeviceBackend() = default;
  virtual void init() = 0;
  virtual void write(const Sequence& seq) = 0;
  /** Returns true if result was filled, false on shutdown or no data. */
  virtual bool read(DecodeResult* result) = 0;
  virtual void terminate() = 0;
};

std::unique_ptr<IDeviceBackend> make_device_backend(const Config& config);

}  // namespace llm_engine
