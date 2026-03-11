#!/bin/bash
# Debug script for HuggingFace connectivity issues on BH-QB-GE

echo "================================"
echo "HuggingFace Network Diagnostics"
echo "================================"
echo ""

echo "1. DNS Resolution Test"
echo "----------------------"
host huggingface.co
echo ""

echo "2. Ping Test"
echo "------------"
ping -c 3 huggingface.co || echo "Ping failed"
echo ""

echo "3. HTTP Connectivity Test"
echo "-------------------------"
curl -v -I --max-time 10 https://huggingface.co 2>&1 | head -20
echo ""

echo "4. HuggingFace Hub Status"
echo "------------------------"
curl --max-time 10 https://status.huggingface.co/api/v2/status.json
echo ""

echo "5. Proxy Environment Variables"
echo "------------------------------"
echo "HTTP_PROXY: ${HTTP_PROXY:-not set}"
echo "HTTPS_PROXY: ${HTTPS_PROXY:-not set}"
echo "NO_PROXY: ${NO_PROXY:-not set}"
echo "http_proxy: ${http_proxy:-not set}"
echo "https_proxy: ${https_proxy:-not set}"
echo "no_proxy: ${no_proxy:-not set}"
echo ""

echo "6. HuggingFace Cache Location"
echo "-----------------------------"
echo "HF_HOME: ${HF_HOME:-not set (default: ~/.cache/huggingface)}"
echo "HF_HUB_CACHE: ${HF_HUB_CACHE:-not set}"
ls -lah ~/.cache/huggingface/ 2>/dev/null || echo "Cache directory not found"
echo ""

echo "7. Test Download of Small File"
echo "------------------------------"
timeout 30 python3 << 'PYEOF'
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

from huggingface_hub import hf_hub_download
import time

print("Attempting to download a small config file...")
start = time.time()
try:
    path = hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-schnell",
        filename="model_index.json",
        timeout=30
    )
    elapsed = time.time() - start
    print(f"✓ Success! Downloaded in {elapsed:.2f}s")
    print(f"  File location: {path}")
except Exception as e:
    elapsed = time.time() - start
    print(f"✗ Failed after {elapsed:.2f}s")
    print(f"  Error: {e}")
PYEOF
echo ""

echo "8. Compare with Working Machine"
echo "-------------------------------"
echo "Run this script on Galaxy (working machine) and compare results"
echo ""

echo "================================"
echo "Diagnostics Complete"
echo "================================"
