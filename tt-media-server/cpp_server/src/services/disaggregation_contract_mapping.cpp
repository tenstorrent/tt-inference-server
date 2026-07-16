// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/disaggregation_contract_mapping.hpp"

#include <vector>

namespace tt::services {

tt::domain::llm::LLMRequest buildDecodeRequestFromPrefillResult(
    const tt::sockets::PrefillResultMessage& message,
    DisaggregatedDecodeRequestOptions options) {
  auto request = tt::domain::llm::LLMRequest(message.taskId);
  request.disaggregated = true;
  request.migrationId = message.migrationId;
  request.skip_apply_chat_template = options.skip_apply_chat_template;
  request.skip_text_decode = options.skip_text_decode;
  if (!message.tokenIds.empty()) {
    request.kv_position_id = static_cast<uint32_t>(message.tokenIds.size() - 1);
    request.prompt.emplace<std::vector<uint32_t>>(message.tokenIds.end() - 1,
                                                  message.tokenIds.end());
    if (options.populate_token_counts) {
      request.prompt_tokens_count = 1;
      request.full_prompt_tokens_count =
          static_cast<int>(message.tokenIds.size());
    }
  } else {
    request.prompt = std::vector<uint32_t>{};
  }
  request.max_tokens = message.remainingTokens;
  request.slotId = message.slotId;
  request.temperature = message.temperature;
  request.top_p = message.topP;
  request.top_k = message.topK;
  request.fast_mode = message.fastMode;
  return request;
}

}  // namespace tt::services
