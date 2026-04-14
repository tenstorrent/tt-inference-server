// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "runners/llm_runner/sampling_params.hpp"

namespace tt::services {

class GuidedDecoderManager {
 public:
  explicit GuidedDecoderManager(const std::vector<std::string>& encodedVocab,
                                int vocabSize);
  ~GuidedDecoderManager();

  GuidedDecoderManager(const GuidedDecoderManager&) = delete;
  GuidedDecoderManager& operator=(const GuidedDecoderManager&) = delete;

  void initRequest(uint32_t taskId,
                   const tt::runners::llm_engine::SamplingParams& params);

  std::vector<int32_t> getNextAllowedTokenIds(uint32_t taskId);

  bool acceptToken(uint32_t taskId, int32_t tokenId);

  bool isCompleted(uint32_t taskId) const;

  bool hasGuidedDecoding(uint32_t taskId) const;

  void removeRequest(uint32_t taskId);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

}  // namespace tt::services
