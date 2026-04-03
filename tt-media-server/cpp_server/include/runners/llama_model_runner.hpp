// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "config/runner_config.hpp"
#include "runners/kv_cache_migrator.hpp"
#include "runners/llm_runner/model_runner.hpp"

namespace llm_engine {

/**
 * IModelRunner that runs Llama-3.1-8B-Instruct via embedded Python interpreter
 * (pybind11). Calls tt_model_runners.llama_runner.Llama31_8BRunner methods
 * directly in-process.
 */
class LlamaModelRunner : public IModelRunner {
 public:
  LlamaModelRunner(const tt::config::LLMConfig& config, DecodeCallback callback,
                   std::unique_ptr<IKVCacheMigrator> migrator = nullptr);
  ~LlamaModelRunner() override;
  void run(const std::vector<Sequence*>& seqs, bool isPrefill) override;
  void exit() override;

  bool isReady() const { return initialized_; }

 private:
  bool initialize();
  void failSequences(const std::vector<Sequence*>& seqs);

  void writePageTable(const KVCacheMigrationData& data);
  KVCacheMigrationData waitForKV(uint32_t taskId);

  tt::config::LLMConfig config_;
  DecodeCallback decode_callback_;
  std::unique_ptr<IKVCacheMigrator> migrator_;
  std::atomic<bool> stop_{false};
  bool initialized_ = false;
  bool lastStepWasPrefill_ = true;

  std::mutex kv_mutex_;
  std::condition_variable kv_cv_;
  std::unordered_map<uint32_t, KVCacheMigrationData> pending_kv_data_;
  std::unordered_set<uint32_t> kv_written_tasks_;
};

std::unique_ptr<IModelRunner> makeLlamaModelRunner(
    const tt::config::LLMConfig& config, DecodeCallback callback,
    std::unique_ptr<IKVCacheMigrator> migrator = nullptr);

}  // namespace llm_engine
