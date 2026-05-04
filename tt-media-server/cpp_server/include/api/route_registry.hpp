// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "config/types.hpp"

namespace tt::api {

/**
 * Per-modality HTTP route allow-list.
 *
 * Drogon auto-registers every `HttpController<>` subclass linked into the
 * binary, so a multi-modal cpp_server cannot stop a controller from existing
 * by skipping a `#include`. Instead, modalities declare their public routes
 * here; main.cpp installs a pre-handling advice that 404s any request whose
 * path is not in the active modality's allow-list (modulo always-exempt
 * paths like /health, /metrics, /openapi.json).
 *
 * The same registry feeds the uniform "Endpoints:" startup log in main.cpp,
 * so adding a new modality adds an entry here once and gets routing + logging
 * for free.
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
  std::unordered_set<std::string> alwaysExemptIndex_;
};

}  // namespace tt::api
