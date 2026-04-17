// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "runners/llm_runner/sampling_params.hpp"

namespace tt::services {

struct TokenAcceptResult {
  bool accepted = true;
  bool completed = false;
};

struct BatchAllowedResult {
  uint32_t taskId;
  std::vector<int32_t> allowedTokenIds;
};

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

  std::vector<BatchAllowedResult> getNextAllowedTokenIdsBatch(
      const std::vector<uint32_t>& taskIds);

  void fillNextBitmask(uint32_t taskId, std::vector<int32_t>& bitmask);

  int vocabSize() const;
  int bitmaskSize() const;

  TokenAcceptResult acceptToken(uint32_t taskId, int32_t tokenId);

  bool hasGuidedDecoding(uint32_t taskId) const;

  void removeRequest(uint32_t taskId);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

}  // namespace tt::services
