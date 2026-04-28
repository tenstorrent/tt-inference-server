// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "domain/sampling_params.hpp"

namespace tt::runners {

struct TokenAcceptResult {
  bool accepted = true;
  bool completed = false;
};

class GuidedDecoderManager {
 public:
  /**
   * @param encodedVocab Byte-level encoded vocabulary used by xgrammar.
   * @param vocabSize Total vocabulary size, including special tokens.
   * @param deterministicSelect When true, fillNextBitmask reduces the bitmask
   *   to a single best token that drives the grammar toward termination. This
   *   is intended for mock runners that lack real logits and would otherwise
   *   pick tokens that prevent the grammar from ever completing (e.g. always
   *   picking digits inside an integer, or content characters inside a
   *   string). Real model runners must leave this off so that they receive
   *   the full bitmask and can sample over it normally.
   */
  explicit GuidedDecoderManager(const std::vector<std::string>& encodedVocab,
                                int vocabSize,
                                bool deterministicSelect = false);
  ~GuidedDecoderManager();

  GuidedDecoderManager(const GuidedDecoderManager&) = delete;
  GuidedDecoderManager& operator=(const GuidedDecoderManager&) = delete;

  void initRequest(uint32_t taskId, const tt::domain::SamplingParams& params);

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

}  // namespace tt::runners
