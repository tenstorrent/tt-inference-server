// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cstdint>
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

  explicit PrefillRequestMessage(tt::domain::TaskID task_id)
      : task_id(std::move(task_id)) {}

  template <class Archive>
  void write(Archive& ar) const {
    int mt = max_tokens.has_value() ? max_tokens.value() : -1;
    ar(task_id.id, prompt, token_ids, mt);
  }

  template <class Archive>
  static PrefillRequestMessage read(Archive& ar) {
    std::string tid;
    std::string p;
    std::vector<int64_t> tids;
    int mt;
    ar(tid, p, tids, mt);
    PrefillRequestMessage msg(tt::domain::TaskID(std::move(tid)));
    msg.prompt = std::move(p);
    msg.token_ids = std::move(tids);
    msg.max_tokens = (mt == -1) ? std::nullopt : std::optional<int>(mt);
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

  explicit PrefillResultMessage(tt::domain::TaskID task_id)
      : task_id(std::move(task_id)) {}

  template <class Archive>
  void write(Archive& ar) const {
    int rt = remaining_tokens.has_value() ? remaining_tokens.value() : -1;
    ar(task_id.id, generated_text, finished, tokens_generated,
       processing_time_ms, token_ids, rt);
  }

  template <class Archive>
  static PrefillResultMessage read(Archive& ar) {
    std::string tid;
    std::string gen_text;
    bool fin;
    int tg;
    double pt;
    std::vector<int64_t> tids;
    int rt;
    ar(tid, gen_text, fin, tg, pt, tids, rt);
    PrefillResultMessage msg(tt::domain::TaskID(std::move(tid)));
    msg.generated_text = std::move(gen_text);
    msg.finished = fin;
    msg.tokens_generated = tg;
    msg.processing_time_ms = pt;
    msg.token_ids = std::move(tids);
    msg.remaining_tokens = (rt == -1) ? std::nullopt : std::optional<int>(rt);
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
