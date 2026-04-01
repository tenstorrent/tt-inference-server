// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "domain/task_id.hpp"

namespace tt::sockets {

/**
 * @brief Prefill request message - sent from decode server to prefill server
 */
struct PrefillRequestMessage {
  tt::domain::TaskID task_id;
  std::string prompt;
  std::vector<int64_t> token_ids;
  std::optional<int> max_tokens;
  std::optional<uint32_t> slot_id;

  explicit PrefillRequestMessage(tt::domain::TaskID taskId)
      : task_id(std::move(taskId)) {}

  template <class Archive>
  void write(Archive& ar) const {
    int mt = max_tokens.has_value() ? max_tokens.value() : -1;
    uint32_t sid = slot_id.value_or(std::numeric_limits<uint32_t>::max());
    ar(task_id, prompt, token_ids, mt, sid);
  }

  template <class Archive>
  static PrefillRequestMessage read(Archive& ar) {
    uint32_t tid;
    std::string p;
    std::vector<int64_t> tids;
    int mt;
    uint32_t sid;
    ar(tid, p, tids, mt, sid);
    PrefillRequestMessage msg(tid);
    msg.prompt = std::move(p);
    msg.token_ids = std::move(tids);
    msg.max_tokens = (mt == -1) ? std::nullopt : std::optional<int>(mt);
    msg.slot_id = (sid == std::numeric_limits<uint32_t>::max())
                      ? std::nullopt
                      : std::optional<uint32_t>(sid);
    return msg;
  }
};

/**
 * @brief Prefill result message - sent from prefill server back to decode
 * server
 *
 * Contains the first token and updated sequence for decode server to continue
 * generation.
 */
struct PrefillResultMessage {
  tt::domain::TaskID task_id;
  std::string generated_text;
  bool finished = false;
  int tokens_generated = 0;
  double processing_time_ms = 0.0;
  std::vector<int64_t> token_ids;
  std::optional<int> remaining_tokens;
  std::optional<uint32_t> slot_id;

  explicit PrefillResultMessage(tt::domain::TaskID taskId)
      : task_id(std::move(taskId)) {}

  template <class Archive>
  void write(Archive& ar) const {
    int rt = remaining_tokens.has_value() ? remaining_tokens.value() : -1;
    uint32_t sid = slot_id.value_or(std::numeric_limits<uint32_t>::max());
    ar(task_id, generated_text, finished, tokens_generated,
       processing_time_ms, token_ids, rt, sid);
  }

  template <class Archive>
  static PrefillResultMessage read(Archive& ar) {
    uint32_t tid;
    std::string genText;
    bool fin;
    int tg;
    double pt;
    std::vector<int64_t> tids;
    int rt;
    uint32_t sid;
    ar(tid, genText, fin, tg, pt, tids, rt, sid);
    PrefillResultMessage msg(tid);
    msg.generated_text = std::move(genText);
    msg.finished = fin;
    msg.tokens_generated = tg;
    msg.processing_time_ms = pt;
    msg.token_ids = std::move(tids);
    msg.remaining_tokens = (rt == -1) ? std::nullopt : std::optional<int>(rt);
    msg.slot_id = (sid == std::numeric_limits<uint32_t>::max())
                      ? std::nullopt
                      : std::optional<uint32_t>(sid);
    return msg;
  }
};

/**
 * @brief Health check message
 */
struct HealthCheckMessage {
  std::string server_id;
  double cpu_usage = 0.0;
  double memory_usage = 0.0;
  int active_tasks = 0;

  template <class Archive>
  void write(Archive& ar) const {
    ar(server_id, cpu_usage, memory_usage, active_tasks);
  }

  template <class Archive>
  static HealthCheckMessage read(Archive& ar) {
    HealthCheckMessage msg;
    ar(msg.server_id, msg.cpu_usage, msg.memory_usage, msg.active_tasks);
    return msg;
  }
};

/**
 * @brief Load balancing info message
 */
struct LoadBalanceMessage {
  std::string server_id;
  int queue_size = 0;
  double avg_processing_time = 0.0;
  bool accepting_tasks = false;

  template <class Archive>
  void write(Archive& ar) const {
    ar(server_id, queue_size, avg_processing_time, accepting_tasks);
  }

  template <class Archive>
  static LoadBalanceMessage read(Archive& ar) {
    LoadBalanceMessage msg;
    ar(msg.server_id, msg.queue_size, msg.avg_processing_time,
       msg.accepting_tasks);
    return msg;
  }
};

}  // namespace tt::sockets
