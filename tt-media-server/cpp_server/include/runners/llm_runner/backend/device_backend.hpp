#pragma once

#include <memory>

#include "runners/llm_runner/config.hpp"
#include "runners/llm_runner/sequence.hpp"

namespace llm_engine::backend {

/**
 * Abstraction for host–device communication (init, write sequence, read token result).
 * Real implementation uses TT device and sockets; mock queues TokenResult without serialization.
 */
class IDeviceBackend {
 public:
  virtual ~IDeviceBackend() = default;
  virtual void init() = 0;
  virtual void write(const Sequence& seq) = 0;
  /** Returns true if result was filled, false on shutdown or no data. */
  virtual bool read(TokenResult* result) = 0;
  virtual void terminate() = 0;
};

std::unique_ptr<IDeviceBackend> make_device_backend(const Config& config);

}  // namespace llm_engine
