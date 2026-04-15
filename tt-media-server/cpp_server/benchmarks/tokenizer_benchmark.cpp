// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "config/settings.hpp"
#include "utils/tokenizers/tokenizer.hpp"

using namespace tt::utils;

constexpr size_t NUM_WARMUP_ITERATIONS = 5;
constexpr size_t NUM_BENCHMARK_ITERATIONS = 50;

struct BenchmarkResult {
  size_t num_tokens;
  double latency_ms;
  double per_token_us;
  double throughput_tokens_per_sec;
};

std::string generateTextWithTokens(size_t targetTokens) {
  std::string text;
  text.reserve(targetTokens * 5);

  std::vector<std::string> words = {"The",
                                    "quick",
                                    "brown",
                                    "fox",
                                    "jumps",
                                    "over",
                                    "the",
                                    "lazy",
                                    "dog",
                                    "In",
                                    "a",
                                    "world",
                                    "where",
                                    "technology",
                                    "advances",
                                    "rapidly",
                                    "machine",
                                    "learning",
                                    "and",
                                    "artificial",
                                    "intelligence",
                                    "are",
                                    "transforming",
                                    "industries",
                                    "across",
                                    "the",
                                    "globe",
                                    "from",
                                    "healthcare",
                                    "to",
                                    "finance",
                                    "education",
                                    "and",
                                    "entertainment",
                                    "These",
                                    "innovations",
                                    "enable",
                                    "computers",
                                    "to",
                                    "learn",
                                    "patterns",
                                    "make",
                                    "predictions",
                                    "and",
                                    "solve",
                                    "complex",
                                    "problems",
                                    "at",
                                    "scale"};

  size_t wordIdx = 0;
  while (text.length() < targetTokens * 5) {
    if (!text.empty()) {
      text += " ";
    }
    text += words[wordIdx % words.size()];
    wordIdx++;
  }

  return text;
}

BenchmarkResult benchmarkEncode(const Tokenizer& tokenizer,
                                const std::string& text) {
  for (size_t i = 0; i < NUM_WARMUP_ITERATIONS; ++i) {
    tokenizer.encode(text);
  }

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<int> tokens;

  for (size_t i = 0; i < NUM_BENCHMARK_ITERATIONS; ++i) {
    tokens = tokenizer.encode(text);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  double latencyMs = duration.count() / 1e6 / NUM_BENCHMARK_ITERATIONS;
  size_t numTokens = tokens.size();
  double perTokenUs = (latencyMs * 1000.0) / numTokens;
  double throughput = numTokens / (latencyMs / 1000.0);

  return {numTokens, latencyMs, perTokenUs, throughput};
}

BenchmarkResult benchmarkDecode(const Tokenizer& tokenizer, size_t numTokens) {
  std::vector<int> tokens;
  for (size_t i = 0; i < numTokens; ++i) {
    tokens.push_back(100 + (i % 1000));
  }

  for (size_t i = 0; i < NUM_WARMUP_ITERATIONS; ++i) {
    tokenizer.decode(tokens);
  }

  auto start = std::chrono::high_resolution_clock::now();
  std::string decoded;

  for (size_t i = 0; i < NUM_BENCHMARK_ITERATIONS; ++i) {
    decoded = tokenizer.decode(tokens);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  double latencyMs = duration.count() / 1e6 / NUM_BENCHMARK_ITERATIONS;
  double perTokenUs = (latencyMs * 1000.0) / numTokens;
  double throughput = numTokens / (latencyMs / 1000.0);

  return {numTokens, latencyMs, perTokenUs, throughput};
}

void printHeader(std::ostream& out, const std::string& title) {
  out << "\n";
  out << "-------------------------------------------------------\n";
  out << title << "\n";
  out << "-------------------------------------------------------\n";
  out << std::setw(10) << "Tokens" << std::setw(17) << "Latency (ms)"
      << std::setw(17) << "Per-token (µs)" << std::setw(17) << "Throughput\n";
  out << "-------------------------------------------------------\n";
}

void printResult(std::ostream& out, const BenchmarkResult& result) {
  out << std::setw(10) << result.num_tokens << std::setw(17) << std::fixed
      << std::setprecision(3) << result.latency_ms << std::setw(17)
      << std::fixed << std::setprecision(3) << result.per_token_us
      << std::setw(12) << std::fixed << std::setprecision(0)
      << result.throughput_tokens_per_sec << " t/s\n";
}

int main(int argc, char* argv[]) {
  std::string tokenizerFilePath =
      argc >= 2 ? argv[1] : tt::config::tokenizerPath();

  if (tokenizerFilePath.empty()) {
    std::cerr << "Tokenizer not found at default location "
                 "(tokenizers/tokenizer.json)\n";
    std::cerr << "Usage: " << argv[0] << " [tokenizer_path]\n";
    std::cerr << "Example: " << argv[0] << " tokenizers/tokenizer.json\n";
    return 1;
  }

  std::cout << "Loading tokenizer from: " << tokenizerFilePath << "\n";
  auto tokenizer = createTokenizer(tt::config::modelType(), tokenizerFilePath);

  if (!tokenizer->isLoaded()) {
    std::cerr << "Failed to load tokenizer from: " << tokenizerFilePath << "\n";
    std::cerr << "Usage: " << argv[0] << " [tokenizer_path]\n";
    std::cerr << "Example: " << argv[0] << " tokenizers/tokenizer.json\n";
    return 1;
  }

  std::cout << "Tokenizer loaded successfully\n";
  std::cout << "Running benchmarks...\n";

  std::vector<size_t> encodeTargets = {50,   100,  200,   400,   800,  1600,
                                       3200, 6400, 12800, 25600, 51200};

  printHeader(std::cout, "TOKENIZATION (text -> tokens)");

  for (size_t target : encodeTargets) {
    std::string text = generateTextWithTokens(target);
    BenchmarkResult result = benchmarkEncode(*tokenizer, text);
    printResult(std::cout, result);
  }

  std::vector<size_t> decodeTargets = {128,  256,   512,   1024,  2048,  4096,
                                       8192, 16384, 32768, 65536, 131072};

  printHeader(std::cout, "DETOKENIZATION (tokens -> text)");

  for (size_t numTokens : decodeTargets) {
    BenchmarkResult result = benchmarkDecode(*tokenizer, numTokens);
    printResult(std::cout, result);
  }

  return 0;
}
