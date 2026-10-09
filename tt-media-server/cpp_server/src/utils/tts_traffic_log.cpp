// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/tts_traffic_log.hpp"

#include <json/json.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/logger.hpp"

namespace tt::utils::tts_traffic {

namespace {

uint64_t monoUs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

uint64_t wallUs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

// FNV-1a. Not cryptographic and not meant to be -- it only has to be stable
// across processes and runs so a record captured with TTS_TRAFFIC_LOG_TEXT=0
// can still be matched to the same prompt captured elsewhere.
std::string stableHash(const std::string& value) {
  uint64_t h = 1469598103934665603ULL;
  for (const unsigned char c : value) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out(16, '0');
  for (int i = 15; i >= 0; --i) {
    out[i] = kHex[h & 0xF];
    h >>= 4;
  }
  return out;
}

// An unterminated request leaves its record open. That is a bug worth seeing
// rather than hiding, but it must not grow without bound on a long-lived
// server, so the oldest open records are dropped past this and counted.
constexpr size_t kMaxOpenRecords = 4096;

struct Record {
  uint64_t arrivalWallUs = 0;
  uint64_t arrivalMonoUs = 0;
  size_t inflightOnArrival = 0;
  size_t capacity = 0;
  std::string text;  // empty when TTS_TRAFFIC_LOG_TEXT=0
  std::string textHash;
  size_t textChars = 0;
  bool hasDescription = false;
  std::string description;
  size_t promptSpeechIds = 0;
  size_t promptTokens = 0;
  uint32_t sampleRateHz = 0;
  uint64_t ttfcUs = 0;
  std::vector<uint64_t> chunkOffsetsUs;
  std::vector<uint64_t> chunkSamples;
};

class TrafficLog {
 public:
  static TrafficLog& instance() {
    static TrafficLog log;
    return log;
  }

  bool enabled() const { return enabled_.load(std::memory_order_acquire); }
  const std::string& path() const { return path_; }
  bool captureText() const { return captureText_; }

  void open(const ArrivalInfo& info, Record record) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (open_.size() >= kMaxOpenRecords) {
      open_.clear();
      dropped_ += kMaxOpenRecords;
      TT_LOG_WARN(
          "[tts-traffic] {} records were still open; dropped them. A record "
          "stays open when its request produces no terminal event.",
          kMaxOpenRecords);
    }
    open_[info.taskId] = std::move(record);
  }

  void chunk(uint32_t taskId, uint32_t chunkIndex, size_t samples, uint32_t sampleRateHz) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = open_.find(taskId);
    if (it == open_.end()) {
      return;
    }
    Record& r = it->second;
    const uint64_t offset = monoUs() - r.arrivalMonoUs;
    if (r.chunkOffsetsUs.empty()) {
      r.ttfcUs = offset;
    }
    // chunkIndex is the producer's numbering; store the offsets positionally
    // and let the index gap, if any, show up as a short chunk list.
    (void)chunkIndex;
    r.chunkOffsetsUs.push_back(offset);
    r.chunkSamples.push_back(samples);
    if (sampleRateHz != 0) {
      r.sampleRateHz = sampleRateHz;
    }
  }

  void finish(uint32_t taskId, const char* outcome) {
    Record record;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = open_.find(taskId);
      if (it == open_.end()) {
        return;
      }
      record = std::move(it->second);
      open_.erase(it);
    }
    write(taskId, record, outcome, monoUs() - record.arrivalMonoUs);
  }

  void write(uint32_t taskId, const Record& r, const char* outcome, uint64_t totalUs) {
    Json::Value j(Json::objectValue);
    j["task_id"] = taskId;
    j["arrival_wall_us"] = static_cast<Json::UInt64>(r.arrivalWallUs);
    j["arrival_mono_us"] = static_cast<Json::UInt64>(r.arrivalMonoUs);
    j["inflight_on_arrival"] = static_cast<Json::UInt64>(r.inflightOnArrival);
    j["capacity"] = static_cast<Json::UInt64>(r.capacity);

    j["text_chars"] = static_cast<Json::UInt64>(r.textChars);
    j["text_sha"] = r.textHash;
    if (!r.text.empty()) {
      j["text"] = r.text;
    }
    if (r.hasDescription) {
      j["description"] = r.description;
    }
    j["prompt_speech_ids"] = static_cast<Json::UInt64>(r.promptSpeechIds);
    j["prompt_tokens"] = static_cast<Json::UInt64>(r.promptTokens);

    j["outcome"] = outcome;
    j["total_us"] = static_cast<Json::UInt64>(totalUs);
    j["chunks"] = static_cast<Json::UInt64>(r.chunkOffsetsUs.size());
    if (!r.chunkOffsetsUs.empty()) {
      j["ttfc_us"] = static_cast<Json::UInt64>(r.ttfcUs);
    }
    j["sample_rate_hz"] = r.sampleRateHz;

    Json::Value offsets(Json::arrayValue);
    Json::Value samples(Json::arrayValue);
    uint64_t totalSamples = 0;
    for (size_t i = 0; i < r.chunkOffsetsUs.size(); ++i) {
      offsets.append(static_cast<Json::UInt64>(r.chunkOffsetsUs[i]));
      samples.append(static_cast<Json::UInt64>(r.chunkSamples[i]));
      totalSamples += r.chunkSamples[i];
    }
    j["chunk_offsets_us"] = offsets;
    j["chunk_samples"] = samples;
    j["audio_samples"] = static_cast<Json::UInt64>(totalSamples);

    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";  // one record per line
    builder["commentStyle"] = "None";
    const std::string line = Json::writeString(builder, j);

    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!out_.is_open()) {
      return;
    }
    out_ << line << '\n';
    // Flushed per record: this is capture for later analysis, and one line per
    // request is nowhere near a throughput concern, so losing the tail of the
    // file to a crash would cost more than the flush does.
    out_.flush();
    if (!out_) {
      TT_LOG_ERROR("[tts-traffic] write to {} failed; disabling capture", path_);
      enabled_.store(false, std::memory_order_release);
      out_.close();
    }
  }

 private:
  TrafficLog() {
    const char* file = std::getenv("TTS_TRAFFIC_LOG");
    if (file == nullptr || *file == '\0') {
      return;
    }
    path_ = file;
    out_.open(path_, std::ios::out | std::ios::app);
    if (!out_.is_open()) {
      TT_LOG_ERROR("[tts-traffic] cannot open TTS_TRAFFIC_LOG={}; capture is off", path_);
      return;
    }
    const char* text = std::getenv("TTS_TRAFFIC_LOG_TEXT");
    captureText_ = !(text != nullptr && text[0] == '0' && text[1] == '\0');
    enabled_.store(true, std::memory_order_release);
    TT_LOG_INFO("[tts-traffic] capturing to {} (prompt text: {})", path_,
                captureText_ ? "included" : "hash and length only");
  }

  std::atomic<bool> enabled_{false};
  bool captureText_ = true;
  std::string path_;
  std::ofstream out_;
  std::mutex fileMutex_;
  std::mutex mutex_;
  std::unordered_map<uint32_t, Record> open_;
  size_t dropped_ = 0;
};

Record makeRecord(const ArrivalInfo& info) {
  TrafficLog& log = TrafficLog::instance();
  Record r;
  r.arrivalWallUs = wallUs();
  r.arrivalMonoUs = monoUs();
  r.inflightOnArrival = info.inflightOnArrival;
  r.capacity = info.capacity;
  if (info.text != nullptr) {
    r.textChars = info.text->size();
    r.textHash = stableHash(*info.text);
    if (log.captureText()) {
      r.text = *info.text;
    }
  }
  if (info.description != nullptr) {
    r.hasDescription = true;
    r.description = *info.description;
  }
  r.promptSpeechIds = info.promptSpeechIds;
  r.promptTokens = info.promptTokens;
  return r;
}

}  // namespace

bool enabled() { return TrafficLog::instance().enabled(); }

const std::string& path() { return TrafficLog::instance().path(); }

void onArrival(const ArrivalInfo& info) {
  TrafficLog& log = TrafficLog::instance();
  if (!log.enabled()) {
    return;
  }
  try {
    log.open(info, makeRecord(info));
  } catch (const std::exception& e) {
    TT_LOG_WARN("[tts-traffic] onArrival failed for task {}: {}", info.taskId, e.what());
  }
}

void onChunk(uint32_t taskId, uint32_t chunkIndex, size_t samples, uint32_t sampleRateHz) {
  TrafficLog& log = TrafficLog::instance();
  if (!log.enabled()) {
    return;
  }
  try {
    log.chunk(taskId, chunkIndex, samples, sampleRateHz);
  } catch (const std::exception& e) {
    TT_LOG_WARN("[tts-traffic] onChunk failed for task {}: {}", taskId, e.what());
  }
}

void onFinish(uint32_t taskId, const char* outcome) {
  TrafficLog& log = TrafficLog::instance();
  if (!log.enabled()) {
    return;
  }
  try {
    log.finish(taskId, outcome);
  } catch (const std::exception& e) {
    TT_LOG_WARN("[tts-traffic] onFinish failed for task {}: {}", taskId, e.what());
  }
}

void onRejected(const ArrivalInfo& info, const char* reason) {
  TrafficLog& log = TrafficLog::instance();
  if (!log.enabled()) {
    return;
  }
  try {
    log.write(info.taskId, makeRecord(info), reason, 0);
  } catch (const std::exception& e) {
    TT_LOG_WARN("[tts-traffic] onRejected failed for task {}: {}", info.taskId, e.what());
  }
}

}  // namespace tt::utils::tts_traffic
