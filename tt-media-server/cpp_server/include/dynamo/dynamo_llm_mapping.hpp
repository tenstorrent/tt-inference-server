// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>

#include "domain/llm/llm_request.hpp"
#include "domain/llm/llm_response.hpp"
#include "dynamo/dynamo_protocol.hpp"
#include "sockets/socket_messages.hpp"

namespace tt::dynamo {

std::shared_ptr<tt::domain::llm::LLMRequest> buildLLMRequestFromGenerateRequest(
    const GenerateRequest& dyn);

tt::sockets::PrefillRequestMessage buildPrefillRequestMessage(
    const GenerateRequest& dyn);

tt::domain::llm::LLMRequest buildDecodeRequestFromPrefillResult(
    const tt::sockets::PrefillResultMessage& message);

TokenChunk tokenChunkFromStreamChunk(
    const tt::domain::llm::LLMStreamChunk& chunk, bool isFinal);

}  // namespace tt::dynamo
