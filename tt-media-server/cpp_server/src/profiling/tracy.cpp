// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
void TracySetPortForMain() {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%d", TRACY_MAIN_PORT);
  setenv("TRACY_PORT", buf, 1);
}

void TracySetPortForWorker(int worker_id) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%d", TRACY_WORKER_PORT_BASE + worker_id);
  setenv("TRACY_PORT", buf, 1);
}

void TracyStartMainProcess() {
  TracySetPortForMain();
  TracyStartupProfiler();
  TracySetThreadName("Main");
  TracyRegisterPlots();
}

void TracyStartupSchedulerParent() {
  TracyPlotConfig("pending_tasks", tracy::PlotFormatType::Number, true, true,
                  0);
}

void TracyRegisterPlots() {
  TracyPlotConfig("pending_tasks", tracy::PlotFormatType::Number, true, true,
                  0);
}

void TracyStartupWorker(int worker_id) {
  TracySetPortForWorker(worker_id);
  TracyStartupProfiler();
}
#else
void tracySetPortForMain() {}
void tracySetPortForWorker(int /*worker_id*/) {}
void tracyStartMainProcess() {}
void tracyStartupSchedulerParent() {}
void tracyStartupWorker(int /*worker_id*/) {}
void tracyRegisterPlots() {}
#endif

}  // namespace tracy_config
