// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include <drogon/drogon.h>
#include <iostream>
#include <csignal>
#include <cstdlib>

// Controllers are auto-registered via ADD_METHOD_TO macros
#include "api/llm_controller.hpp"

// Include OpenAPI controller (defined in openapi.cpp)
// The controller auto-registers itself with Drogon

namespace {
    volatile std::sig_atomic_t g_shutdown_requested = 0;

    void signal_handler(int signal) {
        std::cout << "\n[Main] Received signal " << signal << ", initiating shutdown..." << std::endl;
        g_shutdown_requested = 1;
        drogon::app().quit();
    }
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string host = "0.0.0.0";
    uint16_t port = 8000;
    int threads = std::thread::hardware_concurrency();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-h" || arg == "--host") && i + 1 < argc) {
            host = argv[++i];
        } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "TT Media Server (C++ Drogon)\n"
                      << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -h, --host HOST     Listen host (default: 0.0.0.0)\n"
                      << "  -p, --port PORT     Listen port (default: 8000)\n"
                      << "  -t, --threads N     Number of IO threads (default: CPU cores)\n"
                      << "  --help              Show this help message\n";
            return 0;
        }
    }

    // Setup signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "=================================================\n"
              << "  TT Media Server (C++ Drogon Implementation)\n"
              << "=================================================\n"
              << "  Host: " << host << "\n"
              << "  Port: " << port << "\n"
              << "  IO Threads: " << threads << "\n"
              << "  LLM Test Runner: 120,000 tokens/sec\n"
              << "=================================================\n"
              << std::endl;

    // Configure Drogon
    drogon::app()
        .setLogLevel(trantor::Logger::kWarn)
        .setLogPath("./logs")
        .addListener(host, port)
        .setThreadNum(threads)
        .setMaxConnectionNum(100000)
        .setMaxConnectionNumPerIP(0)  // No limit per IP
        .setIdleConnectionTimeout(60)
        .setKeepaliveRequestsNumber(0)  // No limit
        .setClientMaxBodySize(100 * 1024 * 1024)  // 100MB max body
        .setClientMaxMemoryBodySize(100 * 1024 * 1024)
        .setStaticFilesCacheTime(0);

    std::cout << "[Main] Starting Drogon server at http://" << host << ":" << port << std::endl;
    std::cout << "[Main] Endpoints:\n"
              << "  POST /v1/completions  - OpenAI-compatible completions\n"
              << "  GET  /health          - Health check\n"
              << "  GET  /ready           - Readiness check\n"
              << "  GET  /docs            - Swagger UI\n"
              << "  GET  /openapi.json    - OpenAPI specification\n"
              << std::endl;

    // Run the server
    drogon::app().run();

    std::cout << "[Main] Server shutdown complete" << std::endl;
    return 0;
}
