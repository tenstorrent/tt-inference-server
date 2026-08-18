// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "domain/llm/llm_request.hpp"
#include "sockets/socket_messages.hpp"

namespace tt::services {

struct DisaggregatedDecodeRequestOptions {
  bool skip_apply_chat_template = false;
  bool skip_text_decode = false;
  bool populate_token_counts = false;
};

tt::domain::llm::LLMRequest buildDecodeRequestFromPrefillResult(
    const tt::sockets::PrefillResultMessage& message,
    DisaggregatedDecodeRequestOptions options = {});

}  // namespace tt::services
