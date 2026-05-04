// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "config/types.hpp"

namespace tt::api {

/**
 * Per-modality HTTP route allow-list.
 *
 * Drogon auto-registers every linked `HttpController<>`, so a multi-modal
 * binary cannot disable a controller just by skipping a `#include`. main.cpp
 * installs a sync advice that 404s any request not on the active modality's
 * allow-list (paths registered via `registerAlwaysExempt(...)` are served
 * regardless of MODEL_SERVICE). The same registry drives the "Endpoints:"
 * startup log.
 */
class RouteRegistry {
 public:
  struct RouteSpec {
    std::string method;      /**< Uppercase HTTP verb, e.g. "POST". */
    std::string path;        /**< Exact path or a `{param}` template. */
    std::string description; /**< Free-form, used in startup logs. */
  };

  RouteRegistry(const RouteRegistry&) = delete;
  RouteRegistry& operator=(const RouteRegistry&) = delete;

  static RouteRegistry& instance();

  /** Register a route for a modality. */
  void registerRoute(config::ModelService service, std::string method,
                     std::string path, std::string description);

  /** Mark a path as always-allowed regardless of active modality (health,
   *  liveness, docs, metrics, …). */
  void registerAlwaysExempt(std::string path);

  /** Decide whether the active modality should serve `path`. Matches templated
   *  routes (e.g. "/v1/sessions/{session_id}") against concrete request paths
   *  by component. */
  bool isAllowed(config::ModelService activeService, std::string_view method,
                 std::string_view path) const;

  /** Routes registered for `service`, sorted in registration order. */
  std::vector<RouteSpec> routesFor(config::ModelService service) const;

  /** Always-exempt paths (sorted in registration order). */
  std::vector<std::string> alwaysExemptPaths() const;

  /** Remove all registrations. Test-only helper. */
  void clear();

 private:
  RouteRegistry() = default;

  /** True iff `templatePath` matches `requestPath`. `{name}` segments match
   *  any single non-empty segment; all other segments compare literally. */
  static bool pathMatches(std::string_view templatePath,
                          std::string_view requestPath);

  std::unordered_map<config::ModelService, std::vector<RouteSpec>> routes_;
  std::vector<std::string> alwaysExempt_;
};

}  // namespace tt::api
