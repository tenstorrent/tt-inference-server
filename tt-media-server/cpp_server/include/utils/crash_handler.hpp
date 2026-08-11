// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <ostream>
#include <string>

namespace tt::utils {

// Renders crash-time diagnostics (e.g. the scheduler per-slot state) into an
// ostream. Plain function pointer + context so the registration can be read
// without allocation or locks.
using CrashStateDumpFn = void (*)(void* ctx, std::ostream& os);

/**
 * Installs process-wide crash observability so a fatal exit self-reports to
 * fd 2 instead of dying silently:
 *
 *   * a fatal-signal handler (SIGABRT/SIGSEGV/SIGBUS/SIGFPE/SIGILL) that
 *     writes "[<processTag> pid=N] FATAL signal=N — backtrace:" plus a native
 *     backtrace straight to stderr, then re-raises with the default
 *     disposition so the process still dies by the signal (and dumps core
 *     when RLIMIT_CORE allows);
 *   * a std::terminate handler that prints the active exception's type and
 *     what() (when there is one), runs the registered crash-state dump, then
 *     abort()s — which trips the signal handler above for the backtrace.
 *
 * Also unbuffers stderr so crash output is never lost to stdio buffering.
 *
 * Async-signal-safety: the signal handler uses only snprintf/write/backtrace/
 * backtrace_symbols_fd — no malloc, no iostream (the unwinder is pre-warmed at
 * install time because glibc's first backtrace() call can allocate). The dump
 * callback therefore runs ONLY on the std::terminate path (normal execution
 * context), never inside the signal handler: a dump that takes locks could
 * deadlock a thread that crashed while holding them.
 *
 * Idempotent: later calls re-install the same handlers and update the tag.
 * Call as early as possible in every process entrypoint (main and the forked
 * worker's --worker path); modeled on the migration worker's crash handler
 * (tt-llm-engine/disaggregation/migration/src/worker/main.cpp).
 */
void installCrashHandlers(const std::string& processTag);

/**
 * Registers the crash-state dump invoked by the terminate handler before the
 * process aborts (e.g. a runner registering its scheduler's dump_diagnostics).
 * Overwrites any previous registration — one runner per worker process.
 */
void setCrashStateDumpCallback(void* ctx, CrashStateDumpFn dumpFn);

/**
 * Clears the registration only if it still belongs to `ctx`; runners call
 * this from their destructor so a dead runner is never dereferenced.
 */
void clearCrashStateDumpCallback(void* ctx);

}  // namespace tt::utils
