// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <xgrammar/xgrammar.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "utils/tokenizer.hpp"

using Clock = std::chrono::steady_clock;

int main() {
  const auto& tok = tt::utils::activeTokenizer();
  auto encodedVocab = tok.getEncodedVocab();
  int vocabSize = static_cast<int>(encodedVocab.size());

  printf("Vocab size: %d\n", vocabSize);

  xgrammar::TokenizerInfo tokInfo(encodedVocab, xgrammar::VocabType::BYTE_LEVEL,
                                  vocabSize);
  xgrammar::GrammarCompiler compiler(tokInfo);

  auto t0 = Clock::now();
  auto jsonGrammar = compiler.CompileBuiltinJSONGrammar();
  auto t1 = Clock::now();
  printf(
      "CompileBuiltinJSONGrammar: %ldus\n",
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

  std::string schema =
      R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"city":{"type":"string"}},"required":["name","age","city"],"additionalProperties":false})";

  t0 = Clock::now();
  auto schemaGrammar = compiler.CompileJSONSchema(schema);
  t1 = Clock::now();
  printf(
      "CompileJSONSchema: %ldus\n",
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

  int bitmaskSize = xgrammar::GetBitmaskSize(vocabSize);
  printf("Bitmask size: %d int32s (%d bytes)\n", bitmaskSize, bitmaskSize * 4);

  constexpr int numIterations = 1000;

  auto benchGrammar = [&](const char* name,
                          const xgrammar::CompiledGrammar& grammar) {
    std::vector<long> fillTimes, convertTimes, acceptTimes;
    fillTimes.reserve(numIterations);
    convertTimes.reserve(numIterations);
    acceptTimes.reserve(numIterations);

    xgrammar::GrammarMatcher matcher(grammar);
    std::vector<int32_t> bitmask(bitmaskSize, 0);

    for (int iter = 0; iter < numIterations; ++iter) {
      std::fill(bitmask.begin(), bitmask.end(), 0);

      DLTensor tensor;
      tensor.data = bitmask.data();
      tensor.device = {kDLCPU, 0};
      tensor.ndim = 1;
      tensor.dtype = xgrammar::GetBitmaskDLType();
      int64_t shape = bitmaskSize;
      tensor.shape = &shape;
      tensor.strides = nullptr;
      tensor.byte_offset = 0;

      auto ta = Clock::now();
      matcher.FillNextTokenBitmask(&tensor);
      auto tb = Clock::now();

      std::vector<int32_t> allowed;
      allowed.reserve(1024);
      for (int i = 0; i < vocabSize; ++i) {
        if (bitmask[i / 32] & (1 << (i % 32))) {
          allowed.push_back(i);
        }
      }
      auto tc = Clock::now();

      if (allowed.empty()) {
        printf("  [%s] iter %d: no allowed tokens, grammar terminated\n", name,
               iter);
        break;
      }

      int tokenId = allowed[iter % allowed.size()];

      auto td = Clock::now();
      bool accepted = matcher.AcceptToken(tokenId);
      auto te = Clock::now();

      if (!accepted) {
        printf("  [%s] iter %d: token %d rejected\n", name, iter, tokenId);
        break;
      }

      fillTimes.push_back(
          std::chrono::duration_cast<std::chrono::microseconds>(tb - ta)
              .count());
      convertTimes.push_back(
          std::chrono::duration_cast<std::chrono::microseconds>(tc - tb)
              .count());
      acceptTimes.push_back(
          std::chrono::duration_cast<std::chrono::microseconds>(te - td)
              .count());

      if (matcher.IsTerminated()) {
        printf("  [%s] grammar terminated after %d tokens\n", name, iter + 1);
        break;
      }
    }

    size_t n = fillTimes.size();
    if (n == 0) {
      printf("  [%s] No successful iterations\n", name);
      return;
    }

    auto avg = [](const std::vector<long>& v) {
      return std::accumulate(v.begin(), v.end(), 0L) /
             static_cast<double>(v.size());
    };

    printf(
        "\n=== %s (%zu tokens generated) ===\n"
        "  FillNextTokenBitmask: avg=%.1fus\n"
        "  BitmaskToAllowedIds:  avg=%.1fus\n"
        "  AcceptToken:          avg=%.1fus\n"
        "  Total per token:      avg=%.1fus (%.3fms)\n",
        name, n, avg(fillTimes), avg(convertTimes), avg(acceptTimes),
        avg(fillTimes) + avg(convertTimes) + avg(acceptTimes),
        (avg(fillTimes) + avg(convertTimes) + avg(acceptTimes)) / 1000.0);
  };

  printf("\n--- Benchmarking json_object grammar ---\n");
  benchGrammar("json_object", jsonGrammar);

  printf("\n--- Benchmarking json_schema grammar ---\n");
  benchGrammar("json_schema", schemaGrammar);

  return 0;
}
