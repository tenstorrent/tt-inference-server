// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <string>
#include <string_view>

namespace tt::domain::llm {

enum class LLMErrorReason { GENERIC, TIMEOUT };

inline constexpr std::string_view GENERIC_ERROR_FINISH_REASON = "error";
inline constexpr std::string_view TIMEOUT_ERROR_FINISH_REASON = "timeout_error";

inline std::string finishReasonForError(LLMErrorReason reason) {
  switch (reason) {
    case LLMErrorReason::TIMEOUT:
      return std::string(TIMEOUT_ERROR_FINISH_REASON);
    case LLMErrorReason::GENERIC:
      return std::string(GENERIC_ERROR_FINISH_REASON);
  }
  return std::string(GENERIC_ERROR_FINISH_REASON);
}

inline bool isErrorFinishReason(std::string_view finishReason) {
  return finishReason == GENERIC_ERROR_FINISH_REASON ||
         finishReason == TIMEOUT_ERROR_FINISH_REASON;
}

inline LLMErrorReason errorReasonFromFinishReason(
    std::string_view finishReason) {
  return finishReason == TIMEOUT_ERROR_FINISH_REASON ? LLMErrorReason::TIMEOUT
                                                     : LLMErrorReason::GENERIC;
}

inline bool isTimeoutError(LLMErrorReason reason) {
  return reason == LLMErrorReason::TIMEOUT;
}

}  // namespace tt::domain::llm
