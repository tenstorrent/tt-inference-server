# SDXL Accuracy Test

This script evaluates the accuracy and performance of the SDXL inference server by generating images from COCO dataset prompts and calculating CLIP score, FID score and performance.

## Prerequisites

1. Follow the instructions in [tt-media-server/README.md](../../tt-media-server/README.md)
2. **Start the SDXL inference server** MAX_QUEUE_SIZE=5000 source run_uvicorn.sh
3. Ensure the server is running and healthy before proceeding with accuracy tests (sometimes machine reset is neccessary)



## Setup

### 1. Navigate to the accuracy test directory

```bash
cd utils/sdxl_accuracy
```

### 2. Create Virtual Environment

Run the setup script to create a virtual environment and install dependencies:

```bash
./setup_env.sh
```

This will:
- Create a Python virtual environment in `utils/sdxl_accuracy/.sdxl_accuracy_env`
- Install all required dependencies from `utils/sdxl_accuracy/requirements.txt`

### 3. Activate Environment

```bash
source .sdxl_accuracy_env/bin/activate
```

## Usage

Run the accuracy test:

```bash
python3 sdxl_accuracy.py
```

## Configuration

### Server Settings
- **Port**: Hardcoded to `8000` (default inference server behavior)
- **Server Health Check**: Script automatically verifies server status before starting

### Test Parameters
- **N_PROMPTS**: Number of prompts from the dataset to generate images for (default: 2, max: 5000)
- **Dataset**: Uses COCO 2014 captions dataset
- **Image Generation**: 1 image per prompt with fixed parameters:
  - 20 inference steps
  - Guidance scale: 8
  - Seed: 0
  - Negative Prompt: normal quality, low quality, worst quality, low res, blurry, nsfw, nude

## Output

### Generated Images
- **Location**: `utils/sdxl_accuracy/output/`
- **Format**: PNG files named `generated_image_0.png`, `generated_image_1.png`, etc.

### Test Results
- **Location**: `utils/sdxl_accuracy/test_reports/sdxl_test_results.json`
- **Metrics**:
  - Average generation time
  - CLIP scores (average and standard deviation)
  - FID score

## Workflow

1. **Health Check**: Verifies inference server is running and responsive
2. **Prompt Loading**: Downloads COCO captions dataset if not present
3. **Image Generation**: Generates images for specified number of prompts
4. **Metric Calculation**: 
   - CLIP scores for text-image similarity
   - FID scores for image quality assessment
   - Performance timing metrics
5. **Results Export**: Saves detailed results in JSON format