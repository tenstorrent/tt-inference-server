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
 * Per-`ModelService` HTTP route allow-list. Drogon auto-registers every
 * linked controller, so a multi-service binary cannot disable one by
 * skipping a `#include`; main.cpp installs a sync advice that 404s any
 * request not allowed for the active `ModelService`. Paths registered via
 * `registerAlwaysExempt()` (health, docs, metrics, …) bypass the filter.
 */
class RouteRegistry {
 public:
  struct RouteSpec {
    std::string method;       /**< Uppercase HTTP verb. */
    std::string path;         /**< Exact path or a `{param}` template. */
    std::string description;  /**< Used in startup logs. */
  };

  RouteRegistry(const RouteRegistry&) = delete;
  RouteRegistry& operator=(const RouteRegistry&) = delete;

  static RouteRegistry& instance();

  void registerRoute(config::ModelService service, std::string method,
                     std::string path, std::string description);

  /** Always-allowed regardless of MODEL_SERVICE (health, docs, metrics, …). */
  void registerAlwaysExempt(std::string path);

  /** True if `service` should serve `method path`. Templated route segments
   *  (`{name}`) match any single non-empty segment. */
  bool isAllowed(config::ModelService activeService, std::string_view method,
                 std::string_view path) const;

  /** Routes for `service`, in registration order. */
  std::vector<RouteSpec> routesFor(config::ModelService service) const;

  /** Always-exempt paths, in registration order. */
  std::vector<std::string> alwaysExemptPaths() const;

  /** Test-only. */
  void clear();

 private:
  RouteRegistry() = default;

  static bool pathMatches(std::string_view templatePath,
                          std::string_view requestPath);

  std::unordered_map<config::ModelService, std::vector<RouteSpec>> routes_;
  std::vector<std::string> alwaysExempt_;
};

}  // namespace tt::api
