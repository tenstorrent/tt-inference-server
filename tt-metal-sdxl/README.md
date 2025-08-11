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
4. ```pip install -r requirements.txt```
5. ```uvicorn main:app --lifespan on --port 8000``` (lifespan methods are needed to init device and close the devices)

## SDXL setup

1. ```export MODEL_RUNNER=tt-sdxl```
2. run the server ```uvicorn main:app --lifespan on --port 8000```

## SD-3.5 setup

1. ```export MODEL_RUNNER=tt-sd3.5```
2. Set device env variable ```export MESH_DEVICE=N150```
3. Run the server ```uvicorn main:app --lifespan on --port 8000```

# Environment variables

```text
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
```

# Steps for Onboarding a Model to the Inference Server

If you're integrating a new model into the inference server, here’s a suggested workflow to help guide the process:

1. **Implement a Model Runner** Create a model runner by inheriting the *base_runner* class and implementing its abstract methods. You can find the relevant codebase here: [tt-inference-server/tt-metal-sdxl/tt_model_runners at dev · tenstorrent/tt-inference-server ](https://github.com/tenstorrent/tt-inference-server/tree/dev/tt-metal-sdxl/tt_model_runners)
(most likely a model runner is a *demo.py* file from a model in tt-metal broken down in methods of a class)
2. **Update Dependencies** If your runner relies on any additional libraries, please make sure to add them to the requirements.txt:  [tt-inference-server/tt-metal-sdxl/requirements.txt at dev · tenstorrent/tt-inference-server ](https://github.com/tenstorrent/tt-inference-server/blob/dev/tt-metal-sdxl/requirements.txt)
3. **Modify *runner_fabric.py*** Update *runner_fabric.py* to instantiate your runner based on the configuration: [tt-inference-server/tt-metal-sdxl/tt_model_runners/runner_fabric.py at dev · tenstorrent/tt-inference-server ](https://github.com/tenstorrent/tt-inference-server/blob/dev/tt-metal-sdxl/tt_model_runners/runner_fabric.py)
4. **Add a Dummy Config** Add a basic config entry to help instantiate your runner: [tt-inference-server/tt-metal-sdxl/config/settings.py at dev · tenstorrent/tt-inference-server ](https://github.com/tenstorrent/tt-inference-server/blob/dev/tt-metal-sdxl/config/settings.py)
Alternatively, you can use an environment variable:
```export MODEL_RUNNER=<your-model-runner-name>```
5. **Write a Unit Test** Please include a unit test in the *tests/* folder to verify your runner works as expected. This step is crucial—without it, it’s difficult to pinpoint issues if something breaks later
6. **Adjust the Service Configuration** Configure the service to use your runner by setting the *MODEL_SERVICE* environment variable accordingly.
```export MODEL_SERVICE={image,audio,base}```
7. **Open an Issue for CI Coverage** Kindly submit a GitHub issue for Igor Djuric to review your PR and to help cover end to end running, CI integration, or any missing service steps: [https://github.com/tenstorrent/tt-inference-server/issuesConnect your Github account ](https://github.com/tenstorrent/tt-inference-server/issues)
8. **Share Benchmarks (if available)** If you’ve run any benchmarks or evaluation tests, please share them. They’re very helpful for understanding performance and validating correctness.

# Remaining work:

 1. Add uts
 2. add api tests
 3. Cleanup unused things in runners
 4. Put device specific things into a runner
