#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "llm_engine/config.hpp"
#include "llm_engine/engine/sequence.hpp"
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/distributed.hpp>

namespace llm_engine {

struct DecodeResult {
  SequenceID seq_id;
  int64_t token_id;
};

// Invoked from the device-to-host reader thread when a token is generated.
using DecodeCallback = std::function<void(const DecodeResult&)>;

class DecodeQueue {
 public:
  void push(const DecodeResult& result);
  std::vector<DecodeResult> drain();

 private:
  std::mutex mutex_;
  std::vector<DecodeResult> pending_;
};

class IModelRunner {
 public:
  virtual ~IModelRunner() = default;
  virtual void run(const std::vector<Sequence*>& seqs, bool is_prefill) = 0;
  virtual void exit() = 0;
};

class ModelRunnerStub : public IModelRunner {
 public:
  ModelRunnerStub(const Config& config, DecodeCallback callback);
  ~ModelRunnerStub() override;
  void run(const std::vector<Sequence*>& seqs, bool is_prefill) override;
  void exit() override;

 private:
  void reader_loop();

  Config config_;
  int64_t dummy_token_;
  DecodeCallback decode_callback_;
  std::mutex work_mutex_;
  std::vector<DecodeResult> work_queue_;
  std::atomic<bool> stop_{false};
  std::thread reader_thread_;  // must be last: uses all members above
  std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;
  std::unique_ptr<tt::tt_metal::distributed::H2DSocket> h2d_socket_;
  std::unique_ptr<tt::tt_metal::distributed::D2HSocket> d2h_socket_;
  std::mutex batch_mutex_;
  std::vector<std::vector<Sequence*>> batch_queue_;
};

std::unique_ptr<IModelRunner> make_model_runner(const Config& config,
                                                DecodeCallback callback);

}  // namespace llm_engine
