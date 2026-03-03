#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "runners/llm_runner/config.hpp"
#include "runners/llm_runner/backend/device_backend.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "profiling/tracy.hpp"

namespace llm_engine {

// Invoked from the device-to-host reader thread when a token is generated.
using DecodeCallback = std::function<void(const TokenResult&)>;

class DecodeQueue {
 public:
  void push(const TokenResult& result);
  std::vector<TokenResult> drain();

 private:
  TracyLockable(std::mutex, mutex_);
  std::vector<TokenResult> pending_;
};

class IModelRunner {
 public:
  virtual ~IModelRunner() = default;
  virtual void run(const std::vector<Sequence*>& seqs, bool is_prefill) = 0;
  virtual void exit() = 0;
};

class ModelRunnerStub : public IModelRunner {
 public:
  ModelRunnerStub(const Config& config, DecodeCallback callback,
                  std::unique_ptr<backend::IDeviceBackend> backend);
  ~ModelRunnerStub() override;
  void run(const std::vector<Sequence*>& seqs, bool is_prefill) override;
  void exit() override;

 private:
  void reader_loop();

  Config config_;
  DecodeCallback decode_callback_;
  std::unique_ptr<backend::IDeviceBackend> backend_;
  std::atomic<bool> stop_{false};
  std::thread reader_thread_;
};

std::unique_ptr<IModelRunner> make_model_runner(const Config& config,
                                                DecodeCallback callback);

}  // namespace llm_engine
