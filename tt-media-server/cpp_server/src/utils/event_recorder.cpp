// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/event_recorder.hpp"

#include <cstdlib>

namespace tt::utils {

EventRecorder& EventRecorder::instance() {
  static EventRecorder recorder;
  return recorder;
}

EventRecorder::EventRecorder() {
  const char* path = std::getenv("SLOT_EVENT_LOG");
  if (!path || path[0] == '\0') {
    path = std::getenv("TT_SLOT_EVENT_LOG");
  }
  if (!path || path[0] == '\0') {
    return;
  }

  file.open(path, std::ios::out | std::ios::trunc);
  if (!file.is_open()) {
    return;
  }

  enabled = true;
  epoch = std::chrono::steady_clock::now();
}

EventRecorder::~EventRecorder() {
  if (file.is_open()) {
    file.close();
  }
}

void EventRecorder::record(std::string_view source, std::string_view event,
                           std::string_view payload) {
  if (!enabled) return;

  auto now = std::chrono::steady_clock::now();
  auto elapsedUs =
      std::chrono::duration_cast<std::chrono::microseconds>(now - epoch)
          .count();

  std::lock_guard lock(mutex);
  file << R"({"t_us":)" << elapsedUs << R"(,"src":")" << source
       << R"(","event":")" << event << '"';
  if (!payload.empty()) {
    file << ',' << payload;
  }
  file << "}\n";
  file.flush();
}

}  // namespace tt::utils
