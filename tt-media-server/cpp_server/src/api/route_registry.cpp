// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/route_registry.hpp"

#include <algorithm>
#include <cctype>
#include <utility>

namespace tt::api {

RouteRegistry& RouteRegistry::instance() {
  static RouteRegistry registry;
  return registry;
}

void RouteRegistry::registerRoute(config::ModelService service,
                                  std::string method, std::string path,
                                  std::string description) {
  std::string upper;
  upper.reserve(method.size());
  for (char c : method) {
    upper.push_back(
        static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  routes_[service].push_back(
      RouteSpec{std::move(upper), std::move(path), std::move(description)});
}

void RouteRegistry::registerAlwaysExempt(std::string path) {
  if (std::find(alwaysExempt_.begin(), alwaysExempt_.end(), path) ==
      alwaysExempt_.end()) {
    alwaysExempt_.push_back(std::move(path));
  }
}

bool RouteRegistry::isAllowed(config::ModelService activeService,
                              std::string_view method,
                              std::string_view path) const {
  // Linear scan: < 10 entries, no allocation from string_view.
  for (const auto& exempt : alwaysExempt_) {
    if (exempt == path) return true;
  }
  auto it = routes_.find(activeService);
  if (it == routes_.end()) return false;
  // registerRoute() upper-cases stored methods and Drogon::methodString()
  // returns uppercase, so plain == is safe.
  for (const auto& spec : it->second) {
    if (spec.method != method) continue;
    if (pathMatches(spec.path, path)) return true;
  }
  return false;
}

std::vector<RouteRegistry::RouteSpec> RouteRegistry::routesFor(
    config::ModelService service) const {
  auto it = routes_.find(service);
  if (it == routes_.end()) return {};
  return it->second;
}

std::vector<std::string> RouteRegistry::alwaysExemptPaths() const {
  return alwaysExempt_;
}

void RouteRegistry::clear() {
  routes_.clear();
  alwaysExempt_.clear();
}

bool RouteRegistry::pathMatches(std::string_view templatePath,
                                std::string_view requestPath) {
  // Trailing-slash equivalence: "/x/" and "/x" match.
  if (templatePath.size() > 1 && templatePath.back() == '/')
    templatePath.remove_suffix(1);
  if (requestPath.size() > 1 && requestPath.back() == '/')
    requestPath.remove_suffix(1);

  // Walk segment-by-segment. std::min(find('/'), size()) folds the
  // "no more slashes" (npos) case into "rest of the view".
  while (!templatePath.empty() && !requestPath.empty()) {
    if (templatePath.front() != '/' || requestPath.front() != '/') return false;
    templatePath.remove_prefix(1);
    requestPath.remove_prefix(1);

    auto tEnd = std::min(templatePath.find('/'), templatePath.size());
    auto rEnd = std::min(requestPath.find('/'), requestPath.size());
    auto tSeg = templatePath.substr(0, tEnd);
    auto rSeg = requestPath.substr(0, rEnd);

    if (tSeg.empty() || rSeg.empty()) return false;
    const bool isWildcard =
        tSeg.size() >= 2 && tSeg.front() == '{' && tSeg.back() == '}';
    if (!isWildcard && tSeg != rSeg) return false;

    templatePath.remove_prefix(tEnd);
    requestPath.remove_prefix(rEnd);
  }
  return templatePath.empty() && requestPath.empty();
}

}  // namespace tt::api
