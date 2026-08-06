// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>

#include "tt_llm_engine/scheduler/scheduler_types.hpp"

namespace tt::runners::blaze {

namespace sch = tt_llm_engine::scheduler;

class IPrefillScheduler {
 public:
  virtual ~IPrefillScheduler() = default;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual bool push_request(const sch::ISRequest& request) = 0;
  virtual bool try_pop_response(sch::SchedulerResponse& response) = 0;
  virtual bool try_pop_output(sch::OutputMessage& output) = 0;
  void dump_diagnostics(std::ostream& os) const;
};

class IDecodeScheduler {
 public:
  virtual ~IDecodeScheduler() = default;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual bool push_request(const sch::ISRequest& request) = 0;
  virtual bool try_pop_response(sch::SchedulerResponse& response) = 0;
  virtual bool try_pop_output(sch::OutputMessage& output) = 0;
  virtual uint32_t get_in_flight_count(uint32_t slot_id) const = 0;
  virtual uint32_t get_tokens_generated(uint32_t slot_id) const = 0;
  virtual uint32_t get_max_new_tokens(uint32_t slot_id) const = 0;
  uint32_t get_current_position(uint32_t slot_id) const;
  virtual uint32_t get_generation(uint32_t slot_id) const = 0;
  virtual bool get_evict_pending(uint32_t slot_id) const = 0;
  virtual bool get_stop_pending(uint32_t slot_id) const = 0;
  virtual uint32_t get_spec_accepts(uint32_t slot_id) const = 0;
  virtual uint32_t get_spec_rejects(uint32_t slot_id) const = 0;
  virtual uint32_t get_decode_staging_size() const = 0;
  virtual uint32_t get_prefill_queue_size() const = 0;
  void dump_diagnostics(std::ostream& os) const;
};

}  // namespace tt::runners::blaze
