// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/worker_health.hpp"

#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/registry.h>
#include <prometheus/text_serializer.h>

#include <map>
#include <sstream>
#include <utility>

namespace tt::transport {

namespace {

/// Escape the minimal set of characters so a worker name is safe to embed
/// in our hand-built JSON. We never emit untrusted multi-byte content here, so
/// quote + backslash + control chars are sufficient.
std::string jsonEscape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 2);
  for (char c : in) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
    }
  }
  return out;
}

}  // namespace

WorkerHealth::WorkerHealth(std::string workerName)
    : workerName_(std::move(workerName)) {
  registry_ = std::make_shared<prometheus::Registry>();
  const std::map<std::string, std::string> label = {{"worker", workerName_}};

  up_ = &prometheus::BuildGauge()
             .Name("tt_migration_worker_up")
             .Help("1 if the worker process is healthy (liveness), else 0")
             .Register(*registry_)
             .Add(label);
  ready_ = &prometheus::BuildGauge()
                .Name("tt_migration_worker_ready")
                .Help("1 if the worker finished its own bring-up (readiness), "
                      "else 0")
                .Register(*registry_)
                .Add(label);
  transferFailures_ =
      &prometheus::BuildCounter()
           .Name("tt_migration_worker_transfer_failures_total")
           .Help("Transfers that failed because a peer was unreachable")
           .Register(*registry_)
           .Add(label);
  reresolveAttempts_ =
      &prometheus::BuildCounter()
           .Name("tt_migration_worker_peer_reresolve_attempts_total")
           .Help("Lazy peer re-resolution attempts after a transfer failure")
           .Register(*registry_)
           .Add(label);
  reresolveFailures_ =
      &prometheus::BuildCounter()
           .Name("tt_migration_worker_peer_reresolve_failures_total")
           .Help("Lazy peer re-resolution attempts that did not resolve")
           .Register(*registry_)
           .Add(label);

  std::lock_guard<std::mutex> lock(mutex_);
  refreshGaugesLocked();
}

WorkerHealth::~WorkerHealth() = default;

void WorkerHealth::setLifecycle(WorkerLifecycle state) {
  std::lock_guard<std::mutex> lock(mutex_);
  lifecycle_ = state;
  refreshGaugesLocked();
}

void WorkerHealth::setProcessHealthy(bool healthy, std::string reason) {
  std::lock_guard<std::mutex> lock(mutex_);
  processHealthy_ = healthy;
  unhealthyReason_ = healthy ? std::string{} : std::move(reason);
  refreshGaugesLocked();
}

void WorkerHealth::onTransferFailure() { transferFailures_->Increment(); }
void WorkerHealth::onReresolveAttempt() { reresolveAttempts_->Increment(); }
void WorkerHealth::onReresolveFailure() { reresolveFailures_->Increment(); }

bool WorkerHealth::isLive() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return processHealthy_ && lifecycle_ != WorkerLifecycle::ShuttingDown;
}

bool WorkerHealth::isReady() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return processHealthy_ && lifecycle_ == WorkerLifecycle::Ready;
}

void WorkerHealth::refreshGaugesLocked() {
  const bool live =
      processHealthy_ && lifecycle_ != WorkerLifecycle::ShuttingDown;
  const bool ready = processHealthy_ && lifecycle_ == WorkerLifecycle::Ready;
  up_->Set(live ? 1.0 : 0.0);
  ready_->Set(ready ? 1.0 : 0.0);
}

std::string WorkerHealth::healthJson() const {
  std::lock_guard<std::mutex> lock(mutex_);
  const bool live =
      processHealthy_ && lifecycle_ != WorkerLifecycle::ShuttingDown;
  std::ostringstream ss;
  ss << "{\"status\":\"" << (live ? "ok" : "fail") << "\",\"worker\":\""
     << jsonEscape(workerName_) << "\"";
  if (!live && !unhealthyReason_.empty()) {
    ss << ",\"reason\":\"" << jsonEscape(unhealthyReason_) << "\"";
  }
  ss << "}";
  return ss.str();
}

std::string WorkerHealth::readyJson() const {
  std::lock_guard<std::mutex> lock(mutex_);
  const bool ready = processHealthy_ && lifecycle_ == WorkerLifecycle::Ready;
  std::ostringstream ss;
  ss << "{\"ready\":" << (ready ? "true" : "false") << ",\"worker\":\""
     << jsonEscape(workerName_) << "\"}";
  return ss.str();
}

std::string WorkerHealth::metricsText() const {
  // prometheus' own locks make Collect() safe to call while the worker thread
  // mutates gauges/counters; no need to hold mutex_ here.
  prometheus::TextSerializer serializer;
  std::ostringstream ss;
  serializer.Serialize(ss, registry_->Collect());
  return ss.str();
}

}  // namespace tt::transport
