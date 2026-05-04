// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/route_registry.hpp"

#include <algorithm>
#include <cctype>
#include <utility>

namespace tt::api {

namespace {

bool iequalsAscii(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::toupper(static_cast<unsigned char>(a[i])) !=
        std::toupper(static_cast<unsigned char>(b[i]))) {
      return false;
    }
  }
  return true;
}

bool isTemplateSegment(std::string_view segment) {
  return segment.size() >= 2 && segment.front() == '{' && segment.back() == '}';
}

}  // namespace

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
  // Linear scan over alwaysExempt_ avoids allocating a std::string from
  // string_view per request; the list has < 10 entries so this is faster
  // than an unordered_set lookup at HTTP RPS scale.
  for (const auto& exempt : alwaysExempt_) {
    if (exempt == path) return true;
  }
  auto it = routes_.find(activeService);
  if (it == routes_.end()) return false;
  for (const auto& spec : it->second) {
    if (!iequalsAscii(spec.method, method)) continue;
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
  // Strip a single trailing slash on either side so that "/v1/sessions/" and
  // "/v1/sessions" compare equal.
  auto trim = [](std::string_view s) {
    if (s.size() > 1 && s.back() == '/') s.remove_suffix(1);
    return s;
  };
  auto tmpl = trim(templatePath);
  auto req = trim(requestPath);

  size_t ti = 0;
  size_t ri = 0;
  while (ti < tmpl.size() && ri < req.size()) {
    // Both must start with '/' at this position.
    if (tmpl[ti] != '/' || req[ri] != '/') return false;
    ++ti;
    ++ri;

    size_t tEnd = tmpl.find('/', ti);
    size_t rEnd = req.find('/', ri);
    if (tEnd == std::string_view::npos) tEnd = tmpl.size();
    if (rEnd == std::string_view::npos) rEnd = req.size();

    auto tSeg = tmpl.substr(ti, tEnd - ti);
    auto rSeg = req.substr(ri, rEnd - ri);

    if (tSeg.empty() || rSeg.empty()) return false;
    if (!isTemplateSegment(tSeg) && tSeg != rSeg) return false;

    ti = tEnd;
    ri = rEnd;
  }
  return ti == tmpl.size() && ri == req.size();
}

}  // namespace tt::api
