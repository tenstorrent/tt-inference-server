# GStreamer Face Matching Demo — Blackhole only

This branch builds the demo **for Blackhole only**, on top of a tt-metal base image that already has SFace, YUNet, and the pool fix for yunet on Blackhole.

**TODO:**
- **USB webcam stream (`stream-webcam`):** Find alternatives — currently browser webcam (`webcam-server`) works; USB stream with `--device /dev/video0` and `--group-add` did not work in this setup.
- **8-device parallel path:** Keep/check `parallel-8device` mode (runs `parallel_8device_stream.py`) for Blackhole or multi-device setups.

## Base image

Use your committed Blackhole base (e.g. `tt-metal-base-sface:blackhole`). The Dockerfile does **not** copy sface/yunet or run `setup.sh` — everything comes from the base.

## Build

Run from the **gstreamer_face_matching_demo** folder (so the build context includes `plugins/`, `entrypoint.sh`, `demo/`, `stream_webcam_python.py`, etc.):

```bash
cd ~/sface_demo/tt-inference-server/gstreamer_face_matching_demo
docker build -f Dockerfile -t face-matching-demo:blackhole .
```

**If you changed entrypoint or added scripts:** Rebuild so the new image is used. If the container still runs the old code, force a clean build: `docker build --no-cache -f Dockerfile -t face-matching-demo:blackhole .`

Override base image if needed:

```bash
docker build -f Dockerfile \
  --build-arg BASE_IMAGE=your-tt-metal-base:blackhole \
  -t face-matching-demo:blackhole .
```

## Run

**Before first run — set up registered faces folder and permissions**

The demo saves registered faces under `/app/registered_faces` in the container. The container user (often uid 1000) must be able to write to the mounted folder. On the host:

```bash
mkdir -p ~/faces
# Required: give ownership to the container user (uid 1000) so "Permission denied" on embedding.npy goes away
sudo chown -R 1000:1000 ~/faces
chmod 755 ~/faces
```

If you see `[Errno 13] Permission denied: '/app/registered_faces/.../embedding.npy'`, fix with `sudo chown -R 1000:1000 ~/faces` on the host and re-run.

Then run:

```bash
docker run --rm -it --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v ~/faces:/app/registered_faces \
  -p 8080:8080 -p 8765:8765 \
  face-matching-demo:blackhole
```

Browser: http://localhost:8080 — use the **Register Face** flow to add faces; they are stored in the mounted `~/faces` folder.

## Recommended: webcam-server (browser camera)

Uses your **browser’s camera** (no USB device in the container). Same face recognition; works with port forwarding.

```bash
docker run --rm -it \
  --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v ~/faces:/app/registered_faces \
  -p 8080:8080 -p 8765:8765 \
  face-matching-demo:blackhole webcam-server
```

Open http://localhost:8080 and use the **Webcam** tab.

## USB webcam (stream-webcam)

To use a USB webcam attached to the host, pass the video device into the container and run in **stream-webcam** mode:

```bash
# Use default /dev/video0
docker run --rm -it --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v ~/faces:/app/registered_faces \
  --device /dev/video0:/dev/video0 \
  -p 8080:8080 -p 8765:8765 \
  face-matching-demo:blackhole stream-webcam
```

If the webcam is on a different device (e.g. `/dev/video2`):

```bash
  --device /dev/video2:/dev/video2 \
  face-matching-demo:blackhole stream-webcam /dev/video2
```

Then open http://localhost:8080 — the stream will be from the USB webcam with face detection/recognition overlay.

**If you see `[StreamWebcam] ERROR: Could not open /dev/video0`:** The container user needs access to the video device. Rebuild the image (the Dockerfile adds the user to the `video` group). If it still fails, add the host’s video group to the run:  
`--group-add $(getent group video | cut -d: -f3)`

**If you see “no element face_recognition”:** The entrypoint sets `PYTHONPATH` so the GStreamer Python plugin can import `ttnn`. Rebuild the image so the updated entrypoint is used. If it still fails, use the default mode (no args) or `webcam-server` — they use the Python WebSocket server and do not require the GStreamer plugin.

## Connection error / remote access (port forwarding)

If you’re on a **remote machine** (e.g. SSH) and open the app from your **local** browser, you need to forward the app’s ports from the remote host to your machine.

**On your local machine**, when you SSH in, forward port 8080 (and 8765 if you use WebSocket features):

```bash
ssh -L 8080:localhost:8080 -L 8765:localhost:8765 ttuser@qbge-01
```

Then on your **local** browser open: **http://localhost:8080** (not the remote hostname).

- **“Connection error” when the page won’t load:** Port 8080 isn’t reaching the app. Check that `-p 8080:8080` is in your `docker run` and that SSH uses `-L 8080:localhost:8080`.
- **“Connection error” or blank/broken stream on the Stream tab:** The stream source inside the container (port 8081) may not be running. For `stream-webcam`, if the script exits because it couldn’t open `/dev/video0`, nothing listens on 8081. After the latest script update, the server listens first so the Stream tab connects and shows \"Camera unavailable\" if the camera can't open. To get real USB video, add `--group-add $(getent group video | cut -d: -f3)`. Workaround: run with `webcam-server` and use the Webcam tab (browser camera). Fix webcam access as needed.
