#pragma once

#include <atomic>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "llm_engine/config.hpp"
#include "llm_engine/engine/model_runner.hpp"
#include "runners/llm_engine/engine/sequence.hpp"

namespace llm_engine {

class SpoofedBlitzDecode {
 public:
  explicit SpoofedBlitzDecode(const Config& config);
  ~SpoofedBlitzDecode();

  void run();
  void decode(const std::vector<Sequence*>& seqs, DecodeCallback callback);
  void exit();

 private:
  void receiver_loop();

  Config config_;
  void* device_ctx_ = nullptr;

  struct PendingCallback {
    DecodeCallback cb;
    SequenceID seq_id;
  };
  std::mutex pending_mutex_;
  std::queue<PendingCallback> pending_;

  std::thread receiver_thread_;
  std::atomic<bool> receiver_shutdown_{false};
};

}  // namespace llm_engine
