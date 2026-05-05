// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/recorder/runner_event_recorder.hpp"

#include <strings.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

constexpr const char* ENABLED_ENV = "TT_RUNNER_RECORDER_ENABLED";

bool parseEnabled(const char* value) {
  if (value == nullptr || value[0] == '\0') return false;
  if (std::strcmp(value, "0") == 0) return false;
  if (strcasecmp(value, "false") == 0) return false;
  if (strcasecmp(value, "off") == 0) return false;
  return true;
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
  bool enabled = parseEnabled(std::getenv(ENABLED_ENV));
  enabled_.store(enabled, std::memory_order_release);
  if (enabled) {
    TT_LOG_INFO(
        "[RunnerEventRecorder] enabled (in-memory log, max {} events). "
        "Debug API: GET/DELETE /debug/runner-events",
        MAX_EVENTS);
  }
}

RunnerEventRecorder& RunnerEventRecorder::instance() {
  static RunnerEventRecorder kInstance;
  return kInstance;
}

void RunnerEventRecorder::onTaskSubmitted(
    const tt::domain::Sequence& sequence) {
  if (!isEnabled()) return;

  std::string body;
  body.reserve(384);
  body.push_back('{');
  bool first = true;

  appendString(body, "kind", "task_submitted", first);
  first = false;
  appendUInt(body, "task_id", sequence.taskId, first);
  appendBool(body, "continuation", sequence.isContinuation(), first);
  appendBool(body, "disaggregated", sequence.isDisaggregated(), first);
  appendUInt(body, "num_prompt_tokens", sequence.getNumPromptTokens(), first);
  appendUInt(body, "tokens_len", sequence.getTokenIds().size(), first);
  appendString(body, "tokens_xx64", tokensFingerprint(sequence.getTokenIds()),
               first);

  bool slotIdSet = sequence.getKVCacheSlot() != tt::domain::INVALID_SLOT_ID;
  appendBool(body, "slot_id_set", slotIdSet, first);
  if (slotIdSet) {
    appendUInt(body, "slot_id", sequence.getKVCacheSlot(), first);
  } else {
    appendNull(body, "slot_id", first);
  }

  body.append(",\"sampling\":");
  body.append(serializeSamplingParams(sequence.getSamplingParams()));
  body.push_back('}');

  append(std::move(body));
}

void RunnerEventRecorder::onCancelRequested(uint32_t taskId) {
  if (!isEnabled()) return;

  std::string body;
  body.reserve(48);
  body.push_back('{');
  appendString(body, "kind", "cancel_requested", true);
  appendUInt(body, "task_id", taskId, false);
  body.push_back('}');

  append(std::move(body));
}

void RunnerEventRecorder::append(std::string json) {
  uint64_t seq = seqCounter_.fetch_add(1, std::memory_order_acq_rel) + 1;

  std::lock_guard<std::mutex> lock(mutex_);
  if (events_.size() >= MAX_EVENTS) {
    events_.pop_front();
    size_t dropped = droppedCount_.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (dropped == 1 || (dropped & (dropped - 1)) == 0) {
      // Log on the first drop and at every power-of-two boundary so a
      // misbehaving test that fails to clear the buffer still leaves a
      // visible breadcrumb without flooding the log.
      TT_LOG_WARN(
          "[RunnerEventRecorder] buffer at capacity ({}); dropping oldest "
          "event (total dropped: {})",
          MAX_EVENTS, dropped);
    }
  }
  events_.push_back(Event{seq, std::move(json)});
}

std::vector<RunnerEventRecorder::Event> RunnerEventRecorder::snapshot(
    uint64_t sinceSeq) const {
  std::vector<Event> out;
  std::lock_guard<std::mutex> lock(mutex_);
  out.reserve(events_.size());
  for (const auto& e : events_) {
    if (e.seq > sinceSeq) {
      out.push_back(e);
    }
  }
  return out;
}

void RunnerEventRecorder::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  events_.clear();
  // Keep seqCounter_ monotonic across clears so tests can reason about
  // strictly-increasing sequence numbers.
}

}  // namespace tt::utils::recorder
