#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

set -e

echo "=============================================="
echo "TT-Inference-Server - Build and Run Example"
echo "=============================================="
echo ""

# Change to inference server directory
cd /workspace/tt-inference-server

echo "Prerequisites:"
echo "  - Python 3.8+ installed"
echo "  - Docker installed (for docker-server mode)"
echo "  - Tenstorrent hardware (n150, n300, t3k, galaxy, etc.)"
echo "  - HuggingFace token (for gated models)"
echo ""

# Display available models
echo "Available Models (examples):"
echo "  - Llama-3.2-1B-Instruct (smallest, fastest)"
echo "  - Llama-3.2-3B-Instruct"
echo "  - Llama-3.1-8B-Instruct"
echo "  - Llama-3.1-70B-Instruct"
echo "  - Qwen2.5-7B-Instruct"
echo "  - Mistral-7B-Instruct-v0.3"
echo ""

echo "Available Devices:"
echo "  - n150 (single Wormhole)"
echo "  - n300 (dual Wormhole)"
echo "  - t3k (8x Wormhole in T3K configuration)"
echo "  - galaxy (32x Wormhole)"
echo ""

echo "Available Workflows:"
echo "  - server     : Start inference server only"
echo "  - benchmarks : Run performance benchmarks"
echo "  - evals      : Run accuracy evaluations"
echo "  - reports    : Generate comparison reports"
echo "  - release    : Run all workflows (benchmarks + evals + reports)"
echo ""

echo "=============================================="
echo "Quick Start Examples"
echo "=============================================="
echo ""

echo "Example 1: Run Docker Server with Llama-3.2-1B (smallest model)"
echo "----------------------------------------------------------------"
echo "python3 run.py --model Llama-3.2-1B-Instruct --device n150 --workflow server --docker-server"
echo ""

echo "Example 2: Run Benchmarks with Docker Server"
echo "----------------------------------------------------------------"
echo "python3 run.py --model Llama-3.2-1B-Instruct --device n150 --workflow benchmarks --docker-server"
echo ""

echo "Example 3: Run Full Release Workflow (benchmarks + evals + reports)"
echo "----------------------------------------------------------------"
echo "python3 run.py --model Llama-3.2-1B-Instruct --device n150 --workflow release --docker-server"
echo ""

echo "Example 4: Run Server in Interactive Mode (for debugging)"
echo "----------------------------------------------------------------"
echo "python3 run.py --model Llama-3.2-1B-Instruct --device n150 --workflow server --docker-server -it"
echo ""

echo "=============================================="
echo "Setup Process"
echo "=============================================="
echo ""

# Check if Python 3.8+ is available
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo "✅ Python found: $PYTHON_VERSION"
else
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if python3-venv is available
if python3 -m venv --help &> /dev/null 2>&1; then
    echo "✅ python3-venv is available"
else
    echo "⚠️  python3-venv not found. Install with: apt install python3-venv"
fi

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "✅ Docker is installed"
else
    echo "⚠️  Docker not found. Install Docker for --docker-server mode"
fi

echo ""
echo "=============================================="
echo "Setup Instructions"
echo "=============================================="
echo ""
echo "Step 1: Prepare Environment"
echo "  The run.py script will automatically bootstrap virtual environments"
echo "  using 'uv' and 'venv'. No manual pip install needed!"
echo ""

echo "Step 2: Get HuggingFace Token (for gated models)"
echo "  - Visit https://huggingface.co"
echo "  - Go to Settings -> Access Tokens"
echo "  - Create a token with read access"
echo "  - Accept terms for model repositories (e.g., meta-llama models)"
echo ""

echo "Step 3: Run Your First Example"
echo "  For smallest model (Llama-3.2-1B on n150):"
echo ""
echo "  python3 run.py --model Llama-3.2-1B-Instruct --device n150 --workflow server --docker-server"
echo ""
echo "  On first run, you'll be prompted for:"
echo "    - HF_TOKEN (HuggingFace token)"
echo "    - JWT_SECRET (can be any string for authentication)"
echo ""

echo "=============================================="
echo "Model Setup (Optional Manual Setup)"
echo "=============================================="
echo ""
echo "For manual model weight setup, use setup.sh:"
echo ""
echo "  ./setup.sh Llama-3.2-1B-Instruct"
echo ""
echo "This will:"
echo "  - Check system requirements (disk space, RAM)"
echo "  - Download model weights from HuggingFace or Meta"
echo "  - Optionally repack weights for certain models"
echo "  - Store weights in persistent_volume/ directory"
echo ""

echo "=============================================="
echo "Testing with Prompt Client"
echo "=============================================="
echo ""
echo "After server is running, test with prompt client:"
echo ""
echo "# In another terminal, activate workflow venv:"
echo "source .workflow_venvs/.venv_benchmarks_run_script/bin/activate"
echo ""
echo "# Send a test prompt:"
echo "python3 utils/prompt_client_cli.py \\"
echo "  --service-url http://localhost:8000/v1/completions \\"
echo "  --prompt 'What is the capital of France?' \\"
echo "  --max-tokens 100"
echo ""

echo "=============================================="
echo "Monitoring and Logs"
echo "=============================================="
echo ""
echo "Logs are stored in workflow_logs/ directory:"
echo "  - run_logs/         : Main run.py execution logs"
echo "  - docker_server/    : Docker container server logs"
echo "  - benchmarks_output/: Benchmark results (JSON)"
echo "  - evals_output/     : Evaluation results"
echo "  - reports_output/   : Summary reports (markdown + data)"
echo ""

echo "To view Docker container logs:"
echo "  docker ps    # Find container ID"
echo "  docker logs -f <container-id>"
echo ""

echo "=============================================="
echo "Common Use Cases"
echo "=============================================="
echo ""
echo "1. Development Mode (mount local changes into Docker):"
echo "   python3 run.py --model Llama-3.2-1B-Instruct --device n150 --workflow server --docker-server --dev-mode"
echo ""

echo "2. Custom Service Port:"
echo "   python3 run.py --model Llama-3.2-1B-Instruct --device n150 --workflow server --docker-server --service-port 9000"
echo ""

echo "3. Skip Trace Capture (faster if already captured):"
echo "   python3 run.py --model Llama-3.2-1B-Instruct --device n150 --workflow benchmarks --docker-server --disable-trace-capture"
echo ""

echo "4. Override Docker Image:"
echo "   python3 run.py --model Llama-3.2-1B-Instruct --device n150 --workflow server --docker-server --override-docker-image my-custom-image:tag"
echo ""

echo "5. Specify Device IDs (for multi-device):"
echo "   python3 run.py --model Llama-3.1-70B-Instruct --device t3k --workflow server --docker-server --device-id 0,1,2,3,4,5,6,7"
echo ""

echo "=============================================="
echo "Troubleshooting"
echo "=============================================="
echo ""
echo "1. Python venv issues:"
echo "   python3 run.py --reset-venvs"
echo ""
echo "2. Docker permission errors:"
echo "   sudo usermod -aG docker \$USER"
echo "   newgrp docker"
echo ""
echo "3. Out of disk space:"
echo "   Check requirements in setup.sh (models need 4GB-350GB)"
echo ""
echo "4. HuggingFace 401 Unauthorized:"
echo "   - Check your token at https://huggingface.co"
echo "   - Accept terms for the specific model repository"
echo "   - Update token in .env file"
echo ""

echo "=============================================="
echo "Next Steps"
echo "=============================================="
echo ""
echo "1. Choose a model and device from the lists above"
echo "2. Ensure you have the required hardware"
echo "3. Get your HuggingFace token ready"
echo "4. Run the example command for your chosen model"
echo ""
echo "For more details, see:"
echo "  - docs/workflows_user_guide.md"
echo "  - docs/development.md"
echo "  - README.md"
echo ""
echo "=============================================="
echo "Ready to start!"
echo "=============================================="

