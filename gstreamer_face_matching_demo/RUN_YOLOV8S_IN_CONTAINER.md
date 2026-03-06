# Run YOLOv8s on tt-metal-base-sface:blackhole Container

Use the [ultralytics-models](https://github.com/tenstorrent/tt-metal/tree/ultralytics-models) branch. Copy `models/demos/yolov8s` into the container and run tests/demo inside.

**Note:** The official [YOLOv8s README](https://github.com/tenstorrent/tt-metal/blob/ultralytics-models/models/demos/yolov8s/README.md) lists **Wormhole (n150, n300)**. On **Blackhole** the same code may run; if you see device/arch errors, you may need a Blackhole-specific build or branch.

---

## 1. Get `models/demos/yolov8s` from ultralytics-models branch

On your **host** (one-time):

```bash
# Option A: Clone only the branch and path (sparse)
cd /tmp  # or any dir
git clone --depth 1 --branch ultralytics-models --filter=blob:none --sparse \
  https://github.com/tenstorrent/tt-metal.git tt-metal-yolov8s
cd tt-metal-yolov8s
git sparse-checkout set models/demos/yolov8s

# Option B: Full clone then checkout
git clone https://github.com/tenstorrent/tt-metal.git tt-metal-yolov8s
cd tt-metal-yolov8s
git fetch origin ultralytics-models
git checkout ultralytics-models
```

You should have `models/demos/yolov8s/` with subdirs like `demo/`, `tests/`, `tt/`, etc.

---

## 2. Copy into the running container

Use your container ID (e.g. `3f2c1b8e91e8`). tt-metal in the base image is usually at `/home/container_app_user/tt-metal`.

```bash
# Create demos dir if needed and copy yolov8s into it
docker exec 3f2c1b8e91e8 mkdir -p /home/container_app_user/tt-metal/models/demos

docker cp /tmp/tt-metal-yolov8s/models/demos/yolov8s \
  3f2c1b8e91e8:/home/container_app_user/tt-metal/models/demos/
```

If your clone path differs, replace `/tmp/tt-metal-yolov8s` with the path that contains `models/demos/yolov8s`.

---

## 3. Enter the container

```bash
docker exec -it 3f2c1b8e91e8 bash
```

---

## 4. Inside the container: set up and run YOLOv8s

Assume you're inside the container as `container_app_user`, with `tt-metal` at `/home/container_app_user/tt-metal`.

```bash
# Go to tt-metal root (required for pytest and imports)
cd /home/container_app_user/tt-metal

# Use the env that has tt-metal/ttnn (e.g. default python or venv)
# If the image uses a venv, activate it first, e.g.:
# source /home/container_app_user/python_env_gst/bin/activate   # if present

# Optional: install pytest if not already
pip install pytest --quiet

# 1) Quick PCC test (validates model)
pytest --disable-warnings models/demos/yolov8s/tests/pcc/test_yolov8s.py::test_yolov8s_640

# 2) Demo (reads images from models/demos/yolov8s/demo/images, writes to demo/runs/)
pytest models/demos/yolov8s/demo/demo.py::test_demo[res0-True-tt_model-1-models/demos/yolov8s/demo/images-device_params0]

# 3) E2E performant test (FPS)
pytest --disable-warnings models/demos/yolov8s/tests/perf/test_e2e_performant.py
```

- **PCC test:** Confirms the model runs and passes correlation checks.
- **Demo:** Runs on sample images; output in `models/demos/yolov8s/demo/runs/` (e.g. `tt_model` subdir).
- **Perf test:** Measures FPS (README mentions ~215 FPS on N150; Blackhole may differ).

---

## 5. If the container has no tt-metal at that path

Find where tt-metal is:

```bash
docker exec 3f2c1b8e91e8 find /home -name "tt-metal" -type d 2>/dev/null
# or
docker exec 3f2c1b8e91e8 ls -la /home/container_app_user/
```

Then copy to that root as `models/demos/yolov8s`, e.g. if tt-metal is at `/opt/tt-metal`:

```bash
docker exec 3f2c1b8e91e8 mkdir -p /opt/tt-metal/models/demos
docker cp /tmp/tt-metal-yolov8s/models/demos/yolov8s 3f2c1b8e91e8:/opt/tt-metal/models/demos/
```

And run the same `pytest` commands from that root (`cd /opt/tt-metal` then the commands above).

---

## 6. Copying results out (optional)

```bash
docker cp 3f2c1b8e91e8:/home/container_app_user/tt-metal/models/demos/yolov8s/demo/runs ./yolov8s_runs
```

---

## Summary

| Step | Command |
|------|--------|
| Get code | `git clone --branch ultralytics-models ...` then use `models/demos/yolov8s` |
| Copy in | `docker cp .../yolov8s CONTAINER_ID:/home/container_app_user/tt-metal/models/demos/` |
| Enter | `docker exec -it 3f2c1b8e91e8 bash` |
| PCC test | `cd /home/container_app_user/tt-metal && pytest --disable-warnings models/demos/yolov8s/tests/pcc/test_yolov8s.py::test_yolov8s_640` |
| Demo | `pytest models/demos/yolov8s/demo/demo.py::test_demo[res0-True-tt_model-1-models/demos/yolov8s/demo/images-device_params0]` |
