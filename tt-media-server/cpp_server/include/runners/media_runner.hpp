// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "runners/runner_base.hpp"

namespace tt::runners {

/**
 * Direct-call runner owned by an in-process service. Differs from `IRunner`
 * only in `run()` shape: media services synchronously call `run(request)` and
 * map exceptions to error responses.
 */
template <typename Request, typename Response>
class MediaRunner : public IRunnerBase {
 public:
  virtual Response run(const Request& request) = 0;
};

}  // namespace tt::runners
