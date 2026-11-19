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

}  // namespace tt::runners
