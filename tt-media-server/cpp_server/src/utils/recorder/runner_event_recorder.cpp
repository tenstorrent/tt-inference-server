// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/recorder/runner_event_recorder.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#include "config/types.hpp"
#include "domain/sampling_params.hpp"
#include "domain/sequence.hpp"
#include "domain/slot_types.hpp"
#include "utils/logger.hpp"

#define XXH_INLINE_ALL
#include "xxhash.h"

namespace tt::utils::recorder {

namespace {

constexpr const char* MODE_ENV = "TT_RUNNER_RECORDER_MODE";
constexpr const char* PATH_ENV = "TT_RUNNER_RECORDER_PATH";

Mode parseMode(const char* value) {
  if (value == nullptr) {
    return Mode::OFF;
  }
  std::string s(value);
  for (auto& c : s) c = static_cast<char>(std::tolower(c));
  if (s == "record") return Mode::RECORD;
  if (s == "assert") return Mode::ASSERT;
  return Mode::OFF;
}

const char* responseFormatToString(tt::config::ResponseFormatType type) {
  switch (type) {
    case tt::config::ResponseFormatType::TEXT:
      return "TEXT";
    case tt::config::ResponseFormatType::JSON_OBJECT:
      return "JSON_OBJECT";
    case tt::config::ResponseFormatType::JSON_SCHEMA:
      return "JSON_SCHEMA";
  }
  return "UNKNOWN";
}

std::string tokensFingerprint(const std::vector<int64_t>& tokens) {
  if (tokens.empty()) return "0000000000000000";
  uint64_t h = XXH64(tokens.data(), tokens.size() * sizeof(int64_t), 0);
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(h));
  return std::string(buf, 16);
}

/** Append a quoted JSON string literal. We only need to escape ", \\, and
 *  control chars; recorded fields are model identifiers, fingerprints,
 *  and enum strings, so the simple escape set is sufficient. */
void appendJsonString(std::string& out, const std::string& value) {
  out.push_back('"');
  for (char c : value) {
    switch (c) {
      case '"':
        out.append("\\\"");
        break;
      case '\\':
        out.append("\\\\");
        break;
      case '\n':
        out.append("\\n");
        break;
      case '\r':
        out.append("\\r");
        break;
      case '\t':
        out.append("\\t");
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char esc[8];
          std::snprintf(esc, sizeof(esc), "\\u%04x",
                        static_cast<unsigned int>(c));
          out.append(esc);
        } else {
          out.push_back(c);
        }
    }
  }
  out.push_back('"');
}

void appendKey(std::string& out, const char* key) {
  out.push_back('"');
  out.append(key);
  out.append("\":");
}

void appendBool(std::string& out, const char* key, bool value, bool first) {
  if (!first) out.push_back(',');
  appendKey(out, key);
  out.append(value ? "true" : "false");
}

void appendInt(std::string& out, const char* key, long long value, bool first) {
  if (!first) out.push_back(',');
  appendKey(out, key);
  out.append(std::to_string(value));
}

void appendUInt(std::string& out, const char* key, unsigned long long value,
                bool first) {
  if (!first) out.push_back(',');
  appendKey(out, key);
  out.append(std::to_string(value));
}

void appendFloat(std::string& out, const char* key, double value, bool first) {
  if (!first) out.push_back(',');
  appendKey(out, key);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%g", value);
  out.append(buf);
}

void appendString(std::string& out, const char* key, const std::string& value,
                  bool first) {
  if (!first) out.push_back(',');
  appendKey(out, key);
  appendJsonString(out, value);
}

void appendNull(std::string& out, const char* key, bool first) {
  if (!first) out.push_back(',');
  appendKey(out, key);
  out.append("null");
}

std::string serializeSamplingParams(const tt::domain::SamplingParams& sp) {
  std::string out;
  out.reserve(256);
  out.push_back('{');
  bool first = true;

  if (sp.max_tokens.has_value()) {
    appendInt(out, "max_tokens", sp.max_tokens.value(), first);
  } else {
    appendNull(out, "max_tokens", first);
  }
  first = false;

  appendFloat(out, "temperature", static_cast<double>(sp.temperature), first);

  if (sp.top_p.has_value()) {
    appendFloat(out, "top_p", static_cast<double>(sp.top_p.value()), first);
  } else {
    appendNull(out, "top_p", first);
  }

  if (sp.top_k.has_value()) {
    appendInt(out, "top_k", sp.top_k.value(), first);
  } else {
    appendNull(out, "top_k", first);
  }

  appendBool(out, "ignore_eos", sp.ignore_eos, first);
  appendBool(out, "fast_mode", sp.fast_mode, first);
  appendBool(out, "skip_special_tokens", sp.skip_special_tokens, first);
  appendString(out, "response_format",
               responseFormatToString(sp.response_format_type), first);
  appendBool(out, "json_schema_present", sp.json_schema_str.has_value(), first);

  size_t toolsCount = sp.tools.has_value() ? sp.tools->size() : 0;
  appendUInt(out, "tools_count", toolsCount, first);

  if (sp.tool_choice.has_value()) {
    appendString(out, "tool_choice_type", sp.tool_choice->type, first);
  } else {
    appendNull(out, "tool_choice_type", first);
  }

  out.push_back('}');
  return out;
}

}  // namespace

RunnerEventRecorder::RunnerEventRecorder() {
  mode_ = parseMode(std::getenv(MODE_ENV));
  if (mode_ == Mode::OFF) {
    return;
  }

  const char* pathEnv = std::getenv(PATH_ENV);
  if (pathEnv == nullptr || pathEnv[0] == '\0') {
    TT_LOG_ERROR(
        "[RunnerEventRecorder] {} is set but {} is empty; disabling recorder",
        MODE_ENV, PATH_ENV);
    mode_ = Mode::OFF;
    return;
  }
  path_ = pathEnv;

  if (mode_ == Mode::RECORD) {
    out_.open(path_, std::ios::out | std::ios::trunc);
    if (!out_.is_open()) {
      TT_LOG_ERROR("[RunnerEventRecorder] Failed to open {} for writing",
                   path_);
      mode_ = Mode::OFF;
      return;
    }
    TT_LOG_INFO("[RunnerEventRecorder] RECORD mode -> {}", path_);
  } else {
    std::ifstream in(path_);
    if (!in.is_open()) {
      TT_LOG_ERROR(
          "[RunnerEventRecorder] ASSERT mode but cannot open fixture {}",
          path_);
      mode_ = Mode::OFF;
      return;
    }
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      expected_.push_back(std::move(line));
    }
    TT_LOG_INFO("[RunnerEventRecorder] ASSERT mode <- {} ({} expected events)",
                path_, expected_.size());
  }
}

RunnerEventRecorder::~RunnerEventRecorder() { finalize(); }

RunnerEventRecorder& RunnerEventRecorder::instance() {
  static RunnerEventRecorder kInstance;
  return kInstance;
}

void RunnerEventRecorder::onTaskSubmitted(
    const tt::domain::Sequence& sequence) {
  if (mode_ == Mode::OFF) return;

  std::string line;
  line.reserve(384);
  line.push_back('{');
  bool first = true;

  appendString(line, "kind", "task_submitted", first);
  first = false;
  appendUInt(line, "task_id", sequence.taskId, first);
  appendBool(line, "continuation", sequence.isContinuation(), first);
  appendBool(line, "disaggregated", sequence.isDisaggregated(), first);
  appendUInt(line, "num_prompt_tokens", sequence.getNumPromptTokens(), first);
  appendUInt(line, "tokens_len", sequence.getTokenIds().size(), first);
  appendString(line, "tokens_xx64", tokensFingerprint(sequence.getTokenIds()),
               first);

  bool slotIdSet = sequence.getKVCacheSlot() != tt::domain::INVALID_SLOT_ID;
  appendBool(line, "slot_id_set", slotIdSet, first);
  if (slotIdSet) {
    appendUInt(line, "slot_id", sequence.getKVCacheSlot(), first);
  } else {
    appendNull(line, "slot_id", first);
  }

  line.append(",\"sampling\":");
  line.append(serializeSamplingParams(sequence.getSamplingParams()));
  line.push_back('}');

  emit(line);
}

void RunnerEventRecorder::onCancelRequested(uint32_t taskId) {
  if (mode_ == Mode::OFF) return;

  std::string line;
  line.reserve(48);
  line.push_back('{');
  appendString(line, "kind", "cancel_requested", true);
  appendUInt(line, "task_id", taskId, false);
  line.push_back('}');

  emit(line);
}

void RunnerEventRecorder::emit(const std::string& jsonLine) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (mode_ == Mode::RECORD) {
    out_ << jsonLine << '\n';
    return;
  }

  // ASSERT
  if (expectedCursor_ >= expected_.size()) {
    mismatchCount_.fetch_add(1);
    TT_LOG_ERROR(
        "[RunnerEventRecorder] ASSERT: extra event beyond fixture (idx={}): "
        "{}",
        expectedCursor_, jsonLine);
    return;
  }

  const std::string& expected = expected_[expectedCursor_];
  if (expected != jsonLine) {
    mismatchCount_.fetch_add(1);
    TT_LOG_ERROR(
        "[RunnerEventRecorder] ASSERT: mismatch at idx={}\n  expected: {}\n  "
        "actual:   {}",
        expectedCursor_, expected, jsonLine);
  }
  ++expectedCursor_;
}

bool RunnerEventRecorder::finalize() {
  if (mode_ == Mode::OFF) return true;
  if (finalized_.exchange(true)) return mismatchCount_.load() == 0;

  std::lock_guard<std::mutex> lock(mutex_);

  if (mode_ == Mode::RECORD) {
    if (out_.is_open()) {
      out_.flush();
      out_.close();
    }
    TT_LOG_INFO("[RunnerEventRecorder] Recording complete: {}", path_);
    return true;
  }

  // ASSERT: ensure we consumed every expected event.
  if (expectedCursor_ < expected_.size()) {
    size_t missing = expected_.size() - expectedCursor_;
    mismatchCount_.fetch_add(missing);
    TT_LOG_ERROR(
        "[RunnerEventRecorder] ASSERT: {} expected event(s) not produced "
        "(consumed {}/{})",
        missing, expectedCursor_, expected_.size());
  }
  size_t mismatches = mismatchCount_.load();
  if (mismatches == 0) {
    TT_LOG_INFO("[RunnerEventRecorder] ASSERT PASS ({} events)",
                expected_.size());
    return true;
  }
  TT_LOG_ERROR("[RunnerEventRecorder] ASSERT FAIL ({} mismatches)", mismatches);
  return false;
}

}  // namespace tt::utils::recorder
