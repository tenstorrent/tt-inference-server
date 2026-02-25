// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"
#include "config/settings.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace tt::utils;

constexpr size_t NUM_WARMUP_ITERATIONS = 5;
constexpr size_t NUM_BENCHMARK_ITERATIONS = 50;

struct BenchmarkResult {
    size_t num_tokens;
    double latency_ms;
    double per_token_us;
    double throughput_tokens_per_sec;
};

std::string generate_text_with_tokens(size_t target_tokens) {
    std::string text;
    text.reserve(target_tokens * 5);

    std::vector<std::string> words = {
        "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
        "In", "a", "world", "where", "technology", "advances", "rapidly",
        "machine", "learning", "and", "artificial", "intelligence", "are",
        "transforming", "industries", "across", "the", "globe", "from",
        "healthcare", "to", "finance", "education", "and", "entertainment",
        "These", "innovations", "enable", "computers", "to", "learn", "patterns",
        "make", "predictions", "and", "solve", "complex", "problems", "at", "scale"
    };

    size_t word_idx = 0;
    while (text.length() < target_tokens * 5) {
        if (!text.empty()) {
            text += " ";
        }
        text += words[word_idx % words.size()];
        word_idx++;
    }

    return text;
}

BenchmarkResult benchmark_encode(const Tokenizer& tokenizer, const std::string& text) {
    for (size_t i = 0; i < NUM_WARMUP_ITERATIONS; ++i) {
        tokenizer.encode(text);
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> tokens;

    for (size_t i = 0; i < NUM_BENCHMARK_ITERATIONS; ++i) {
        tokens = tokenizer.encode(text);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    double latency_ms = duration.count() / 1e6 / NUM_BENCHMARK_ITERATIONS;
    size_t num_tokens = tokens.size();
    double per_token_us = (latency_ms * 1000.0) / num_tokens;
    double throughput = num_tokens / (latency_ms / 1000.0);

    return {num_tokens, latency_ms, per_token_us, throughput};
}

BenchmarkResult benchmark_decode(const Tokenizer& tokenizer, size_t num_tokens) {
    std::vector<int> tokens;
    for (size_t i = 0; i < num_tokens; ++i) {
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
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    double latency_ms = duration.count() / 1e6 / NUM_BENCHMARK_ITERATIONS;
    double per_token_us = (latency_ms * 1000.0) / num_tokens;
    double throughput = num_tokens / (latency_ms / 1000.0);

    return {num_tokens, latency_ms, per_token_us, throughput};
}

void print_header(std::ostream& out, const std::string& title) {
    out << "\n";
    out << "-------------------------------------------------------\n";
    out << title << "\n";
    out << "-------------------------------------------------------\n";
    out << std::setw(10) << "Tokens"
        << std::setw(17) << "Latency (ms)"
        << std::setw(17) << "Per-token (µs)"
        << std::setw(17) << "Throughput\n";
    out << "-------------------------------------------------------\n";
}

void print_result(std::ostream& out, const BenchmarkResult& result) {
    out << std::setw(10) << result.num_tokens
        << std::setw(17) << std::fixed << std::setprecision(3) << result.latency_ms
        << std::setw(17) << std::fixed << std::setprecision(3) << result.per_token_us
        << std::setw(12) << std::fixed << std::setprecision(0) << result.throughput_tokens_per_sec
        << " t/s\n";
}

int main(int argc, char* argv[]) {
    std::string tokenizer_file_path = argc >= 2 ? argv[1] : tt::config::tokenizer_path();

    if (tokenizer_file_path.empty()) {
        std::cerr << "Tokenizer not found at default location (tokenizers/tokenizer.json)\n";
        std::cerr << "Usage: " << argv[0] << " [tokenizer_path]\n";
        std::cerr << "Example: " << argv[0] << " tokenizers/tokenizer.json\n";
        return 1;
    }

    std::cout << "Loading tokenizer from: " << tokenizer_file_path << "\n";
    Tokenizer tokenizer(tokenizer_file_path);

    // Check if tokenizer loaded successfully
    if (!tokenizer.is_loaded()) {
        std::cerr << "Failed to load tokenizer from: " << tokenizer_file_path << "\n";
        std::cerr << "Usage: " << argv[0] << " [tokenizer_path]\n";
        std::cerr << "Example: " << argv[0] << " tokenizers/tokenizer.json\n";
        return 1;
    }

    std::cout << "Tokenizer loaded successfully\n";
    std::cout << "Running benchmarks...\n";

    std::vector<size_t> encode_targets = {
        50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200
    };

    print_header(std::cout, "TOKENIZATION (text -> tokens)");

    for (size_t target : encode_targets) {
        std::string text = generate_text_with_tokens(target);
        BenchmarkResult result = benchmark_encode(tokenizer, text);
        print_result(std::cout, result);
    }

    std::vector<size_t> decode_targets = {
        128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
    };

    print_header(std::cout, "DETOKENIZATION (tokens -> text)");

    for (size_t num_tokens : decode_targets) {
        BenchmarkResult result = benchmark_decode(tokenizer, num_tokens);
        print_result(std::cout, result);
    }

    return 0;
}
