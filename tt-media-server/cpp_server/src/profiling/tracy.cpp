// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Tracy config and helper implementations. Declarations in
// include/profiling/tracy.hpp.

#include "profiling/tracy.hpp"

#ifdef TRACY_ENABLE
#include <cstdio>
#include <cstdlib>
#endif

namespace tracy_config {

#ifdef TRACY_ENABLE
void tracySetPortForMain() {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%d", TRACY_MAIN_PORT);
  setenv("TRACY_PORT", buf, 1);
}

void tracySetPortForWorker(int workerId) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%d", TRACY_WORKER_PORT_BASE + workerId);
  setenv("TRACY_PORT", buf, 1);
}

void tracyStartMainProcess() {
  tracySetPortForMain();
  tracyStartupProfiler();
  tracySetThreadName("Main");
  tracyRegisterPlots();
}

void tracyStartupSchedulerParent() {
  TracyPlotConfig("pending_tasks", tracy::PlotFormatType::Number, true, true,
                  0);
}

void tracyRegisterPlots() {
  TracyPlotConfig("pending_tasks", tracy::PlotFormatType::Number, true, true,
                  0);
}

void tracyStartupWorker(int workerId) {
  tracySetPortForWorker(workerId);
  tracyStartupProfiler();
}
#else
void tracySetPortForMain() {}
void tracySetPortForWorker(int /*workerId*/) {}
void tracyStartMainProcess() {}
void tracyStartupSchedulerParent() {}
void tracyStartupWorker(int /*workerId*/) {}
void tracyRegisterPlots() {}
#endif

}  // namespace tracy_config
