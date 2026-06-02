// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Single project include for Tracy. Use this instead of <tracy/Tracy.hpp>.
// When TRACY_ENABLE is not defined, zone/plot macros are no-ops and helpers are
// stubs.

#ifndef TT_MEDIA_SERVER_PROFILING_TRACY_HPP
#define TT_MEDIA_SERVER_PROFILING_TRACY_HPP

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>

namespace tracy_config {
constexpr int TRACY_MAIN_PORT = 8086;
constexpr int TRACY_WORKER_PORT_BASE = 8087;

void tracySetPortForMain();
void tracySetPortForWorker(int workerId);
/** Start Tracy for the main process. Call once before any Tracy use (e.g. from
 * register_services). */
void tracyStartMainProcess();
void tracyStartupSchedulerParent();
void tracyStartupWorker(int workerId);
void tracyRegisterPlots();

inline void tracyFrameMark() { FrameMark; }
inline void tracyFrameMarkStart(const char* name) { FrameMarkStart(name); }
inline void tracyFrameMarkEnd(const char* name) { FrameMarkEnd(name); }
inline void tracySetThreadName(const char* name) { tracy::SetThreadName(name); }
#if defined(TRACY_DELAYED_INIT) && defined(TRACY_MANUAL_LIFETIME)
inline void tracyStartupProfiler() { tracy::StartupProfiler(); }
#else
inline void tracyStartupProfiler() {}
#endif
}  // namespace tracy_config

#else
#define ZONE_SCOPED_N(name) ((void)0)
#define ZoneScopedN(name) ZONE_SCOPED_N(name)
#define ZONE_SCOPED ((void)0)
#define TRACY_PLOT(name, value) ((void)0)
#define TRACY_PLOT_CONFIG(...) ((void)0)
#define TRACY_LOCKABLE(type, varname) type varname

namespace tracy_config {
constexpr int TRACY_MAIN_PORT = 8086;
constexpr int TRACY_WORKER_PORT_BASE = 8087;

void tracySetPortForMain();
void tracySetPortForWorker(int workerId);
void tracyStartMainProcess();
void tracyStartupSchedulerParent();
void tracyStartupWorker(int workerId);
void tracyRegisterPlots();

inline void tracyFrameMark() {}
inline void tracyFrameMarkStart(const char* /*name*/) {}
inline void tracyFrameMarkEnd(const char* /*name*/) {}
inline void tracySetThreadName(const char* /*name*/) {}
inline void tracyStartupProfiler() {}
}  // namespace tracy_config
#endif

#endif  // TT_MEDIA_SERVER_PROFILING_TRACY_HPP
