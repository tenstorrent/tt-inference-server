// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "telemetry/sentry_tracing.hpp"

#include <sentry.h>

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <utility>

#include "utils/logger.hpp"

namespace tt::telemetry {

namespace detail {

/// Guards against a double finish when the transaction handle is shared
/// across stream callbacks (sentry_transaction_finish consumes the object).
struct TxState {
  std::atomic<bool> finished{false};
};

}  // namespace detail

namespace {

std::atomic<bool> gInitialized{false};

sentry_transaction_t* asTransaction(void* raw) {
  return static_cast<sentry_transaction_t*>(raw);
}

std::string lowercase(std::string value) {
  for (auto& c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return value;
}

bool isHex(const std::string& value) {
  for (const char c : value) {
    if (std::isxdigit(static_cast<unsigned char>(c)) == 0) {
      return false;
    }
  }
  return !value.empty();
}

/// Convert a W3C `traceparent` header ("00-<32 hex>-<16 hex>-<2 hex>") to
/// the equivalent `sentry-trace` value ("<32 hex>-<16 hex>-<0|1>") that the
/// SDK understands. Returns an empty string when the input does not look
/// like a traceparent.
std::string sentryTraceFromTraceparent(const std::string& traceparent) {
  // version(2)-trace_id(32)-parent_id(16)-flags(2)
  if (traceparent.size() < 2 + 1 + 32 + 1 + 16 + 1 + 2) {
    return {};
  }
  if (traceparent[2] != '-' || traceparent[35] != '-' ||
      traceparent[52] != '-') {
    return {};
  }
  const std::string traceId = traceparent.substr(3, 32);
  const std::string parentId = traceparent.substr(36, 16);
  const std::string flags = traceparent.substr(53, 2);
  if (!isHex(traceId) || !isHex(parentId) || !isHex(flags)) {
    return {};
  }
  const bool sampled =
      (std::stoi(flags, nullptr, 16) & 0x01) != 0;  // W3C sampled flag
  return traceId + "-" + parentId + "-" + (sampled ? "1" : "0");
}

void collectHeader(const char* key, const char* value, void* userdata) {
  auto* out = static_cast<HeaderMap*>(userdata);
  if (key != nullptr && value != nullptr) {
    (*out)[lowercase(key)] = value;
  }
}

double tracesSampleRateFromEnv() {
  const char* raw = std::getenv("SENTRY_TRACES_SAMPLE_RATE");
  if (raw == nullptr || raw[0] == '\0') {
    return 1.0;
  }
  try {
    const double rate = std::stod(raw);
    if (rate < 0.0) return 0.0;
    if (rate > 1.0) return 1.0;
    return rate;
  } catch (const std::exception&) {
    TT_LOG_WARN("[Telemetry] Invalid SENTRY_TRACES_SAMPLE_RATE='{}', using 1.0",
                raw);
    return 1.0;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Transaction
// ---------------------------------------------------------------------------

Transaction::Transaction(void* raw)
    : raw_(raw), state_(std::make_shared<detail::TxState>()) {}

Transaction::Transaction(Transaction&& other) noexcept
    : raw_(std::exchange(other.raw_, nullptr)),
      state_(std::move(other.state_)) {}

Transaction& Transaction::operator=(Transaction&& other) noexcept {
  if (this != &other) {
    finish();
    raw_ = std::exchange(other.raw_, nullptr);
    state_ = std::move(other.state_);
  }
  return *this;
}

Transaction::~Transaction() { finish(); }

void Transaction::setTag(const std::string& key, const std::string& value) {
  if (raw_ == nullptr || state_->finished.load()) return;
  sentry_transaction_set_tag(asTransaction(raw_), key.c_str(), value.c_str());
}

void Transaction::setData(const std::string& key, const std::string& value) {
  if (raw_ == nullptr || state_->finished.load()) return;
  sentry_transaction_set_data(asTransaction(raw_), key.c_str(),
                              sentry_value_new_string(value.c_str()));
}

void Transaction::setData(const std::string& key, int64_t value) {
  if (raw_ == nullptr || state_->finished.load()) return;
  sentry_transaction_set_data(
      asTransaction(raw_), key.c_str(),
      sentry_value_new_int32(static_cast<int32_t>(value)));
}

std::string Transaction::traceparent() const {
  if (raw_ == nullptr || state_->finished.load()) return {};
  HeaderMap headers;
  sentry_transaction_iter_headers(asTransaction(raw_), collectHeader, &headers);
  // The SDK emits `sentry-trace` as "<trace id>-<span id>[-<0|1>]"; reshape
  // it into the W3C form the receiving side expects.
  const auto it = headers.find("sentry-trace");
  if (it == headers.end()) return {};
  const std::string& st = it->second;
  if (st.size() < 32 + 1 + 16 || st[32] != '-') return {};
  const std::string traceId = st.substr(0, 32);
  const std::string spanId = st.substr(33, 16);
  if (!isHex(traceId) || !isHex(spanId)) return {};
  const bool sampled = st.size() < 51 || st[50] != '0';
  return "00-" + traceId + "-" + spanId + (sampled ? "-01" : "-00");
}

void Transaction::finish() {
  if (raw_ == nullptr) return;
  sentry_transaction_t* tx = asTransaction(std::exchange(raw_, nullptr));
  if (state_->finished.exchange(true)) return;
  sentry_transaction_finish(tx);
}

void Transaction::finishError(const std::string& message) {
  if (raw_ == nullptr) return;
  sentry_transaction_t* tx = asTransaction(std::exchange(raw_, nullptr));
  if (state_->finished.exchange(true)) return;
  sentry_transaction_set_tag(tx, "error.message", message.c_str());
  sentry_transaction_set_status(tx, SENTRY_SPAN_STATUS_INTERNAL_ERROR);
  sentry_transaction_finish(tx);
}

// ---------------------------------------------------------------------------
// SDK lifecycle
// ---------------------------------------------------------------------------

void init(const std::string& release, const std::string& instanceTag) {
  const char* dsn = std::getenv("SENTRY_DSN");
  if (dsn == nullptr || dsn[0] == '\0') {
    TT_LOG_INFO("[Telemetry] SENTRY_DSN not set; Sentry tracing disabled");
    return;
  }

  const char* environment = std::getenv("SENTRY_ENVIRONMENT");
  const char* releaseOverride = std::getenv("SENTRY_RELEASE");
  const char* debug = std::getenv("SENTRY_DEBUG");
  const double sampleRate = tracesSampleRateFromEnv();

  sentry_options_t* options = sentry_options_new();
  sentry_options_set_dsn(options, dsn);
  sentry_options_set_environment(
      options, (environment != nullptr && environment[0] != '\0')
                   ? environment
                   : "development");
  sentry_options_set_release(
      options, (releaseOverride != nullptr && releaseOverride[0] != '\0')
                   ? releaseOverride
                   : release.c_str());
  sentry_options_set_traces_sample_rate(options, sampleRate);
  // Per-role SDK run directory: decode and prefill share a working
  // directory in local/disaggregated deployments.
  sentry_options_set_database_path(
      options, ("./logs/.sentry-native-" + instanceTag).c_str());
  if (debug != nullptr && debug[0] == '1') {
    sentry_options_set_debug(options, 1);
  }

  if (sentry_init(options) != 0) {
    TT_LOG_ERROR("[Telemetry] sentry_init failed; tracing disabled");
    return;
  }
  gInitialized.store(true);
  TT_LOG_INFO(
      "[Telemetry] Sentry tracing enabled (environment={}, release={}, "
      "traces_sample_rate={})",
      (environment != nullptr && environment[0] != '\0') ? environment
                                                         : "development",
      (releaseOverride != nullptr && releaseOverride[0] != '\0')
          ? releaseOverride
          : release,
      sampleRate);
}

void shutdown() {
  if (!gInitialized.exchange(false)) return;
  sentry_close();
}

bool enabled() { return gInitialized.load(); }

// ---------------------------------------------------------------------------
// Transaction start / trace continuation
// ---------------------------------------------------------------------------

std::string traceparentFromHeaders(const HeaderMap& headers) {
  for (const auto& [key, value] : headers) {
    if (lowercase(key) == "traceparent") {
      return value;
    }
  }
  return {};
}

Transaction startTransaction(const std::string& name, const std::string& op,
                             const std::string& traceparent) {
  if (!gInitialized.load()) return {};

  const std::string sentryTrace = sentryTraceFromTraceparent(traceparent);
  if (sentryTrace.empty()) {
    // No (valid) upstream trace context: publish nothing rather than start
    // a disconnected server-side trace.
    return {};
  }

  sentry_transaction_context_t* txCtx =
      sentry_transaction_context_new(name.c_str(), op.c_str());
  sentry_transaction_context_update_from_header(txCtx, "sentry-trace",
                                                sentryTrace.c_str());
  sentry_transaction_t* tx =
      sentry_transaction_start(txCtx, sentry_value_new_null());
  return Transaction{tx};
}

}  // namespace tt::telemetry
