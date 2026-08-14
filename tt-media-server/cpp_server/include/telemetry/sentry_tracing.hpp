// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

/**
 * Sentry distributed tracing (issue #4778).
 *
 * Thin wrapper around sentry-native so call sites never touch the SDK
 * directly. Every operation is a no-op unless init() found a SENTRY_DSN.
 *
 * The server only ever *continues* a trace: startTransaction() requires the
 * W3C `traceparent` value the Dynamo frontend forwards from the client, and
 * returns an inert transaction when it is absent or malformed — a request
 * without an upstream trace publishes nothing to Sentry. The frontend does
 * not forward `sentry-trace`/`baggage`, so those are deliberately not
 * handled. Transaction::traceparent() emits the header value for the next
 * hop (the disaggregated decode -> prefill ZMQ leg).
 *
 * Configuration comes from tt::config (env vars with compiled-in defaults
 * from config/defaults.hpp): SENTRY_DSN (default: the shared
 * tt-inference-server project; export it empty to disable tracing),
 * SENTRY_ENVIRONMENT, SENTRY_RELEASE, SENTRY_TRACES_SAMPLE_RATE (the
 * upstream sampling decision carried in traceparent is always honored) and
 * SENTRY_DEBUG.
 */

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace tt::telemetry {

using HeaderMap = std::unordered_map<std::string, std::string>;

namespace detail {
struct TxState;
}

/// One Sentry transaction (root span) per inbound request. Default
/// constructed instances are inert, so call sites work unchanged when
/// tracing is disabled or the request carried no traceparent.
/// finish()/finishError() are idempotent; whichever runs first wins.
class Transaction {
 public:
  Transaction() = default;
  Transaction(const Transaction&) = delete;
  Transaction& operator=(const Transaction&) = delete;
  Transaction(Transaction&& other) noexcept;
  Transaction& operator=(Transaction&& other) noexcept;
  ~Transaction();

  bool active() const { return raw_ != nullptr; }

  void setTag(const std::string& key, const std::string& value);
  void setData(const std::string& key, const std::string& value);
  void setData(const std::string& key, int64_t value);

  /// W3C `traceparent` value a downstream hop needs to continue this trace
  /// (decode -> prefill over ZMQ). Empty when inactive or already finished.
  std::string traceparent() const;

  void finish();
  void finishError(const std::string& message);

 private:
  friend Transaction startTransaction(const std::string&, const std::string&,
                                      const std::string&);
  explicit Transaction(void* raw);

  void* raw_ = nullptr;
  std::shared_ptr<detail::TxState> state_;
};

/// Initialize the SDK from the environment. No-op (tracing disabled) when
/// SENTRY_DSN is unset. `release` labels events when SENTRY_RELEASE is not
/// set; `instanceTag` keeps per-process SDK run directories apart (e.g.
/// "decode" / "prefill" sharing one working directory).
void init(const std::string& release, const std::string& instanceTag);

/// Flush queued envelopes and shut the SDK down.
void shutdown();

bool enabled();

/// Case-insensitive lookup of the `traceparent` transport header. Empty
/// string when the request carried none.
std::string traceparentFromHeaders(const HeaderMap& headers);

/// Start a transaction continuing the trace described by `traceparent`
/// ("00-<32 hex trace id>-<16 hex parent id>-<2 hex flags>"). Returns an
/// inert transaction when tracing is disabled or `traceparent` is
/// empty/malformed — this server never starts a trace of its own.
Transaction startTransaction(const std::string& name, const std::string& op,
                             const std::string& traceparent);

}  // namespace tt::telemetry
