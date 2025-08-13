# TT non-LLM inference server

This server is built to serve non-LLM models. Currently supported models:

1. SDXL
2. SD- 3.5

# Repo structure

1. Config - config files that can be overrriden by environment variables.
2. Domain - Domain and transfer objects
3. Model services - Services for processing models, scheduler for models and a runner
4. Open_ai_api - controllers in OpenAI flavor
5. Resolver - creator of scheduler and model, depending on the config creates singleton instances of scheduelr and model service
6. Security - Auth features
7. Tests - general end to end tests
8. tt_model_runners - runners for devices and models. Runner_fabric is responsible for creating a needed runner

More details about each folder will be provided below

# Installation instructions

To just run a server build a docker file and run it.

For development running:

1. Setup tt-metal and all the needed variables for it
2. Make sure you're in tt-metal's python env
3. Clone repo into the root of tt-metal
4. pip install -r requirements.txt
5. uvicorn main:app --lifespan on --port 8000 (lifespan methods are needed to init device and close the devices)

## SDXL setup

1. export MODEL_RUNNER=tt-sdxl
2. run the server uvicorn main:app --lifespan on --port 8000


## SD-3.5 setup

1. export MODEL_RUNNER=tt-sd3.5
2. Set device env variable export MESH_DEVICE=N150
3. Run the server uvicorn main:app --lifespan on --port 8000


# Environment variables

model_service - image
LOG_LEVEL - CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0


ENVIRONMENT - production / development

# Remaining work:

 1. Add uts
 2. add api tests
 3. Cleanup unused things in runers
 4. Put device specific things into a runner
