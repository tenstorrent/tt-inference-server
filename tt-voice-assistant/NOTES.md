# TT Voice Assistant - Build Notes

## Machine: QB2 (sjc2-t3019)
- 4x Blackhole chips (2x P300c cards, each with 2x P150 equivalent)
- AMD Ryzen 7 9700X, 256GB RAM

## Device Selection

To run on a **single P150** (one chip on P300c):

```bash
TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
TT_VISIBLE_DEVICES='0' \
<your_command>
```

### Device IDs on P300c

| Physical Card | Board ID | Chip Position | PCIe Bus | Device |
|---------------|----------|---------------|----------|--------|
| Card 1 | 0000046131925035 | ASIC 1 (0x1) | 01:00.0 | /dev/tenstorrent/0 |
| Card 1 | 0000046131925035 | ASIC 0 (0x0) | 02:00.0 | /dev/tenstorrent/1 |
| Card 2 | 0000046131925083 | ASIC 1 (0x1) | 03:00.0 | /dev/tenstorrent/2 |
| Card 2 | 0000046131925083 | ASIC 0 (0x0) | 04:00.0 | /dev/tenstorrent/3 |

**Summary:** 2 Physical P300c cards, 4 Blackhole ASICs 

### Running Multiple Models on Separate Devices

```bash
# Model A on device 0
TT_VISIBLE_DEVICES='0' python model_a.py &

# Model B on device 1
TT_VISIBLE_DEVICES='1' python model_b.py &
```

---

## Docker Workflow

### Container Management

```bash
# Start NEW container (fresh)
docker run -it --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G <image> bash

# Re-enter EXISTING stopped container (preserves changes!)
docker start -ai <container_id>

# List all containers (including stopped)
docker ps -a

# Commit container to new image (single layer)
docker commit <container_id> <new_image_name>:<tag>
```

### Current Build Container
```
Container ID: af22d9208393
Base Image: tt-metal-base_qb2:latest
```

---

## Model Setup Progress

### YuNet (Face Detection) ✅
- [x] Clone YUNet repo: `./models/experimental/yunet/setup.sh`
- [x] Apply Blackhole pool fix (patch ttnn_yunet.py)
- [x] Test: 
  ```bash
  TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  TT_VISIBLE_DEVICES='0' \
  pytest models/experimental/yunet/tests/pcc/test_pcc.py
  ```

### SFace (Face Recognition) ✅
- [x] Copy SFace model: `docker cp ~/tt-inference-server/gstreamer_face_matching_demo/models/sface <container_id>:/home/container_app_user/tt-metal/models/experimental/sface`
- [x] ONNX weights auto-download on first use
- [x] Test:
  ```bash
  TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  TT_VISIBLE_DEVICES='0' \
  pytest models/experimental/sface/tests/pcc/test_pcc.py -v
  ```

### Whisper (Speech-to-Text) ✅
- [x] Script: `test_whisper_single_device.py` (single P150 mode)
- [x] Model: `distil-whisper/distil-large-v3` (~1.5GB, auto-downloads)
- [x] Test:
  ```bash
  TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  TT_VISIBLE_DEVICES='0' \
  python test_whisper_single_device.py
  ```
- **Performance**: First run ~20s (compile), cached ~80ms, 539 t/s

### Llama 3.1 8B Instruct (Chat) ✅
- [x] Weights: `/home/container_app_user/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct`
- [x] Test (2 devices / full P300c):
  ```bash
  export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
  TT_VISIBLE_DEVICES="0,1" pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" --timeout=0
  ```
  - Device: P300
  - TTFT: 52.26ms
  - Speed: 25.98ms @ 38.49 tok/s/user

- [x] Test (single P150):
  ```bash
  export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
  TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  TT_VISIBLE_DEVICES="1" \
  pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" --timeout=0
  ```
  - Device: P150
  - TTFT: 85.52ms
  - Speed: 43.42ms @ 23.03 tok/s/user

### Qwen3 TTS (Text-to-Speech) ✅
- [x] Model: `models/demos/qwen3_tts/`
- [x] Weights: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (auto-downloads from HuggingFace)
- [x] **Blackhole fix applied**: CP prefill/decode run non-traced (trace capture doesn't support KV cache writes on Blackhole)
- [x] Test:
  ```bash
  TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  TT_VISIBLE_DEVICES='3' \
  python models/demos/qwen3_tts/demo/demo_full_ttnn_tts.py --text "Hello from Tenstorrent"
  ```
- **Performance** (single P150):
  - Decode throughput: 13.68 frames/sec
  - Avg decode step: 75.3 ms/frame (Talker: 24.9ms traced, CP: 50.1ms non-traced)
  - Output: `/tmp/ttnn_tts_output.wav`

---

## Blackhole Fixes Applied

### YuNet MaxPool Fix
The upstream tt-metal only applies pool workaround for Wormhole. Blackhole also needs it.

**File:** `models/experimental/yunet/tt/ttnn_yunet.py`

**Change:**
```python
# BEFORE (upstream - hangs on Blackhole):
self._is_wormhole = device.arch() == Arch.WORMHOLE_B0

# AFTER (fixed):
self._use_pool_workaround = device.arch() in (Arch.WORMHOLE_B0, Arch.BLACKHOLE)
```

Patch file location: `~/tt-inference-server/gstreamer_face_matching_demo/model_patches/ttnn_yunet.py`

---

## Docker Images Status

### Current Working Images

#### Production Ready (Single Layer)
```bash
tt-voice-assistant:clean-single-layer
├── Size: 105GB (38.3GB compressed)
├── Layers: 1 layer (flattened via export/import)
├── Contains: All models + fixes + weights + HF cache
├── Status: ✅ Production ready
└── Use: Final deployment, distribution
```

#### Development Backup (Multi Layer)
```bash
tt-voice-assistant:working-v1
├── Size: 126GB (43.7GB compressed)
├── Layers: 113 layers (112 base + 1 commit)
├── Contains: Same as clean version
├── Status: ✅ Development backup
└── Use: Quick testing, experiments
```

---

## Container Environment Setup

**Complete setup command** (run when entering container):

```bash
cd /home/container_app_user/tt-metal && \
source python_env/bin/activate && \
export HF_HOME=/home/container_app_user/.cache/huggingface && \
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:$PYTHONPATH" && \
export TT_METAL_HOME=/home/container_app_user/tt-metal && \
export TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto && \
echo "✅ Environment ready with system packages access!"
```

**Key Environment Variables:**
- `HF_HOME`: Points to Hugging Face model cache
- `PYTHONPATH`: Includes system packages for onnx/opencv access
- `TT_MESH_GRAPH_DESC_PATH`: P150 single-chip mesh descriptor
- `TT_METAL_HOME`: TT Metal installation path

---

## Final Commit

When all models are set up and tested:

```bash
docker commit af22d9208393 tt-voice-assistant:bh-v1
```


#docker play of tt metal base image
# checkout to any branch or commit
# build metal
# other terminal : 
docker exec -u root 4b1b5c0bcf0d rm -rf /home/container_app_user/tt-metal/python_env
# ./create_venv.sh
#commit to new docker image - so u can update everything in container 
Awesome !