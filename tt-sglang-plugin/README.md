 
1.Prepare sglang from source code for cpu ( untill there is proper support for cpu compatible sglang via pip install )

git clone https://github.com/sgl-project/sglang.git
cd sglang

2. Install SGLang Python Package and Dependencies

cd python
cp pyproject_cpu.toml pyproject.toml
pip install --upgrade pip setuptools
pip install .

3. Install CPU Versions of torch, torchvision, and triton

pip install torch==2.9.0+cpu torchvision==0.24.0+cpu triton==3.5.0 --index-url https://download.pytorch.org/whl/cpu

        If you get errors about missing wheels, check the PyTorch CPU install guide for the latest versions and commands.

4. Build the CPU Backend Kernels

cd ../sgl-kernel
cp pyproject_cpu.toml pyproject.toml
pip install .

5. Set Required Environment Variables

export SGLANG_USE_CPU_ENGINE=1
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=${LD_PRELOAD}:/opt/sglang-venv/lib/libiomp5.so:${LD_LIBRARY_PATH}/libtcmalloc.so.4:${LD_LIBRARY_PATH}/libtbbmalloc.so.2

Adjust /opt/sglang-venv if your venv is elsewhere. also ld_preload is not necessary if it fails

6. Launch the SGLang Server (CPU Mode)

python -m sglang.launch_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --trust-remote-code \
    --disable-overlap-schedule \
    --device cpu \
    --host 0.0.0.0 \
    --tp 6

SGLang TT-Metal Plugin Setup

1: Activate TT-Metal Environment

source localdev/.../tt-metal/python_env/bin/activate

2: Install the TT Plugin

cd ../sglang-plugin
pip install -e .

3: Create the .pth File for Subprocess Patching

echo 'import sglang_tt_plugin' > localdev/.../tt-metal/python_env/lib/python3.10/site-packages/sglang_tt_plugin.pth
cat /localdev/.../tt-metal/python_env/lib/python3.10/site-packages/sglang_tt_plugin.pth

4: Verify Installation

python -c "import sglang_tt_plugin; print('Plugin version:', sglang_tt_plugin.__version__)"
which sglang-tt-server

5: Run the Server

unset LD_PRELOAD
sglang-tt-server --model-path meta-llama/Llama-3.1-8B-Instruct --page-size 128

"test" with some curl command: 
curl -s http://127.0.0.1:30000/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "what is a butterfly?",
    "max_tokens": 64,
    "temperature": 0.7
  }' | jq -r '.choices[0].text'

