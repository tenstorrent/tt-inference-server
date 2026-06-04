// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>

namespace tt::utils {

/**
 * Utility class for generating unique task IDs.
 *
 * Task IDs are uint32_t values generated via atomic counter.
 * This provides:
 * - Thread-safe generation
 * - Deterministic sequential IDs (1, 2, 3, ...)
 * - Simple debugging (sequential IDs are easy to track)
 */
class TaskIDGenerator {
 public:
  /**
   * Generate a new unique task ID using atomic counter.
   * Thread-safe, starts at 1 and increments.
   */
  static uint32_t generate() {
    static std::atomic<uint32_t> counter{0};
    return ++counter;
  }
};

/**
 * Generates a node-prefixed trace id used to correlate one user request across
 * decode HTTP, decode worker, prefill HTTP, prefill worker, and inter-server
 * sockets.
 *
 * Format: "<role>-<8 hex chars>" (e.g. "decode-3a9f2b1c"). The role is the
 * server's LLM_MODE ("decode" / "prefill" / "regular") so a single grep for
 * the trace id surfaces every related log line, while the role/8-hex prefix
 * keeps the id short enough to fit on one log line and to be carried in the
 * `X-Request-Id` HTTP response header.
 *
 * The 8 hex chars are the high word of an additive golden-ratio (Weyl)
 * sequence seeded once per process with a random 64-bit value. Within a
 * process the ids stay distinct for far longer than a random 32-bit space
 * would (no birthday clustering until ~2^32 requests); the role prefix plus
 * the random seed keep them effectively unique for the lifetime of a request
 * even across nodes/restarts. The space is still only 32 bits, so de-duplicate
 * a fleet-wide log search with task_id/timestamps if you aggregate millions of
 * ids from same-role replicas.
 *
 * The generator does NOT enforce inbound format - if a client supplies an
 * `X-Request-Id` header (or OpenAI-compatible header) the controller stores
 * that value verbatim instead of generating a new one.
 */
class TraceIdGenerator {
 public:
  /** Generate a new trace id. Uses tt::config::llmMode() for the role prefix.
   */
  static std::string generate();

  /** Generate a new trace id using an explicit role prefix. Useful in tests or
   * non-LLM services that don't have an LLM_MODE. */
  static std::string generate(std::string_view role);

  /** Shared HTTP entry-point policy: returns `inboundRequestId` verbatim when
   * non-empty (honoring a caller- or gateway-supplied `X-Request-Id`),
   * otherwise mints a fresh trace id via generate(). */
  static std::string resolveOrGenerate(std::string_view inboundRequestId);
};

}  // namespace tt::utils
