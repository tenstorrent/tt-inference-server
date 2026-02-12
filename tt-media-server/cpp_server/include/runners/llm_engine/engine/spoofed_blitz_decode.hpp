#pragma once

#include <atomic>
#include <thread>
#include <vector>

#include "llm_engine/config.hpp"
#include "llm_engine/engine/model_runner.hpp"
#include "runners/llm_engine/engine/sequence.hpp"

namespace llm_engine {

class SpoofedBlitzDecode {
 public:
  SpoofedBlitzDecode(const Config& config, DecodeCallback decode_callback);
  ~SpoofedBlitzDecode();

  void run();
  void decode(const std::vector<Sequence*>& seqs);
  void exit();

 private:
  void receiver_loop();

  Config config_;
  DecodeCallback decode_callback_;
  void* device_ctx_ = nullptr;

  std::thread receiver_thread_;
  std::atomic<bool> receiver_shutdown_{false};
};

}  // namespace llm_engine
