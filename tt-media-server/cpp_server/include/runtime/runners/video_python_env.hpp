// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <filesystem>

namespace tt::runners {

/** Directory that holds the `cpp_dit_adapters` package: TT_VIDEO_PYTHON_DIR if
 *  set, else `<binary dir>/../python` (cpp_server/python). */
std::filesystem::path resolveVideoPythonDir();

/** Put tt-metal (TT_METAL_HOME, its python_env site-packages) and the
 *  cpp_dit_adapters directory on the embedded interpreter's sys.path. Needs
 *  the GIL. Throws when TT_METAL_HOME is unset or the package is missing.
 *  Nothing from the Python tt-media-server is added. */
void prepareVideoPythonPath();

/** Whether the embedded interpreter, when VideoDeviceRunner starts it, installs
 *  Python's own signal handlers (SIGINT, SIGPIPE). A worker process keeps the
 *  default (true). When the interpreter lives inside the server process (single
 *  process mode) it must be false: Python's SIGINT handler would replace the
 *  server's shutdown handler. Set before the runner's warmup(). */
void setVideoPythonSignalHandlers(bool enable);
bool videoPythonSignalHandlers();

}  // namespace tt::runners
