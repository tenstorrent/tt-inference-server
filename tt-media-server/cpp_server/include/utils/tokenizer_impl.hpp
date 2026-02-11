// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#ifdef ENABLE_TOKENIZER
#include <memory>
#include <tokenizers_cpp.h>

namespace tt::utils {

struct TokenizerUtilImpl {
    std::unique_ptr<tokenizers::Tokenizer> tok;
};

}  // namespace tt::utils
#endif
