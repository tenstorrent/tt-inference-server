// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <fstream>
#include <mutex>
#include <string_view>

namespace tt::utils {

class EventRecorder {
 public:
  static EventRecorder& instance();

  void record(std::string_view source, std::string_view event,
              std::string_view payload = "");

  bool isEnabled() const { return enabled; }

  EventRecorder(const EventRecorder&) = delete;
  EventRecorder& operator=(const EventRecorder&) = delete;

 private:
  EventRecorder();
  ~EventRecorder();

  bool enabled{false};
  std::ofstream file;
  std::mutex mutex;
  std::chrono::steady_clock::time_point epoch;
};

}  // namespace tt::utils
