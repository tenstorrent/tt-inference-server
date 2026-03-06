#!/bin/bash
# GStreamer Face Recognition Demo Entrypoint
set -e

# Activate the GStreamer-compatible venv
source /home/container_app_user/python_env_gst/bin/activate
cd /home/container_app_user/tt-metal

# Set environment (GStreamer Python plugin loader looks in GST_PLUGIN_PATH/python/ for .py files)
export GST_PLUGIN_PATH=/app/gstreamer_plugins/plugins
# PYTHONPATH so the face_recognition plugin can import ttnn when loaded by gst-launch (uses process env)
export PYTHONPATH=/app/models:/home/container_app_user/tt-metal:/home/container_app_user/python_env_gst/lib/python3.10/site-packages:${PYTHONPATH:-}
export GST_DEBUG=${GST_DEBUG:-1}

# Enable pipeline timing if BENCHMARK mode
if [ "$1" = "benchmark" ]; then
    export GST_DEBUG="GST_TRACER:7,GST_ELEMENT:4"
    export GST_TRACERS="latency(flags=pipeline+element);stats"
    echo "[BENCHMARK] GStreamer tracing ENABLED"
fi

# Clear GStreamer cache
rm -f ~/.cache/gstreamer-1.0/registry.x86_64.bin 2>/dev/null || true

# Create __init__.py files for imports
touch /home/container_app_user/tt-metal/tests/__init__.py 2>/dev/null || true
touch /home/container_app_user/tt-metal/tests/ttnn/__init__.py 2>/dev/null || true

STREAM_PORT=${STREAM_PORT:-8080}
TCP_PORT=8081

# Common pipeline for face recognition
FACE_PROCESS="queue ! videoconvert ! videoscale ! video/x-raw,format=BGRx,width=640,height=640 ! face_recognition"

# Output: MJPEG stream
STREAM_OUTPUT="videoconvert ! jpegenc quality=85 ! multipartmux boundary=frame ! tcpserversink host=0.0.0.0 port=$TCP_PORT"

# Function to start HTTP wrapper
start_http_wrapper() {
    echo "Starting HTTP server on port $STREAM_PORT..."
    python3 /app/http_stream.py &
    sleep 1
}

echo "============================================"
echo "  Face Recognition Demo on Tenstorrent"
echo "  YuNet (Detection) + SFace (Recognition)"
echo "============================================"

# Check plugin
if ! gst-inspect-1.0 face_recognition > /dev/null 2>&1; then
    echo "WARNING: face_recognition plugin not found - using Python mode"
fi

# Show help
if [ "$1" = "help" ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo ""
    echo "Usage: docker run ... face-matching-demo [MODE]"
    echo ""
    echo "  MODES:"
    echo "    (no args)        - Demo with built-in video"
    echo "    webcam-server    - Browser webcam mode (click Webcam tab)"
    echo "    stream-file      - Custom video file"
    echo "    stream-rtsp      - RTSP IP camera"
    echo "    stream-webcam    - USB webcam on server"
    echo "    parallel-8device   - 8-device parallel face recognition"
    echo "    benchmark          - Run pipeline with detailed timing breakdown"
    echo "    accuracy-benchmark - LFW accuracy & latency (PyTorch vs TTNN)"
    echo ""
    echo "  Examples:"
    echo "    docker run --network host ... face-matching-demo"
    echo "    docker run --network host ... face-matching-demo webcam-server"
    echo "    docker run --network host ... face-matching-demo stream-rtsp rtsp://camera/stream"
    echo ""
    echo "  Persist registered faces:"
    echo "    Add: -v ./faces:/app/registered_faces"
    echo "    Example: docker run --network host -v ./faces:/app/registered_faces ... face-matching-demo webcam-server"
    echo ""
    exit 0
fi

# Default: run demo (interactive video mode via webcam-server)
if [ $# -eq 0 ]; then
    echo ""
    echo "Starting Face Recognition Demo..."
    echo "  Browser: http://localhost:$STREAM_PORT"
    echo "  Click 'Play Test Video' or upload your own video"
    echo "  Click 'Webcam' tab to use browser webcam"
    echo "  Click 'Register Face' to add faces"
    echo "  (Model compiles on first run ~2 min)"
    echo ""
    echo "  Tip: Add -v ./faces:/app/registered_faces to persist faces"
    echo ""
    export HTTP_MODE=stream
    start_http_wrapper
    # Use webcam-server backend which handles video via WebSocket
    python3 /app/websocket_server.py
    exit 0
fi

MODE=$1
shift

case $MODE in
    "stream-test")
        echo "Streaming test pattern..."
        echo "  Browser: http://localhost:$STREAM_PORT"
        start_http_wrapper
        gst-launch-1.0 videotestsrc ! \
            "video/x-raw,width=640,height=640,framerate=30/1,format=UYVY" ! \
            $FACE_PROCESS ! $STREAM_OUTPUT
        ;;
    
    "stream-file")
        VIDEO_PATH=$1
        echo "Streaming video: $VIDEO_PATH"
        echo "  Browser: http://localhost:$STREAM_PORT"
        start_http_wrapper
        gst-launch-1.0 filesrc location="$VIDEO_PATH" ! \
            decodebin ! videoconvert ! videoscale ! videorate ! \
            "video/x-raw,width=640,height=640,framerate=30/1" ! \
            $FACE_PROCESS ! $STREAM_OUTPUT
        ;;
    
    "stream-rtsp")
        RTSP_URL=$1
        echo "Streaming from RTSP: $RTSP_URL"
        echo "  Browser: http://localhost:$STREAM_PORT"
        start_http_wrapper
        gst-launch-1.0 rtspsrc location="$RTSP_URL" latency=0 protocols=tcp ! \
            rtph264depay ! h264parse ! avdec_h264 ! \
            videoconvert ! videoscale ! videorate ! \
            "video/x-raw,width=640,height=640,framerate=30/1" ! \
            $FACE_PROCESS ! $STREAM_OUTPUT
        ;;
    
    "stream-webcam")
        DEVICE=${1:-/dev/video0}
        echo "Streaming from USB webcam: $DEVICE (Python mode)"
        echo "  Browser: http://localhost:$STREAM_PORT"
        start_http_wrapper
        # Use Python script (no GStreamer plugin required); streams MJPEG to TCP 8081
        exec python3 /app/stream_webcam_python.py "$DEVICE" --port $TCP_PORT
        ;;

    "parallel-8device")
        VIDEO_ARGS=""
        if [ $# -gt 0 ]; then
            VIDEO_ARGS="$@"
        else
            # Device 0: current video; Device 1: Mohamed+Jim+Jonathan; Device 2: attached photos+Teja+Jim; Device 3: current video
            if [ -f /app/demo/test_video.mp4 ]; then
                if [ ! -f /app/demo/video_device1.mp4 ] || [ ! -f /app/demo/video_device2.mp4 ]; then
                    echo "Creating device 1 & 2 videos from face photos..."
                    python3 /app/demo/create_device_videos_from_faces.py || true
                fi
                if [ -f /app/demo/video_device1.mp4 ] && [ -f /app/demo/video_device2.mp4 ]; then
                    VIDEO_ARGS="/app/demo/test_video.mp4 /app/demo/video_device1.mp4 /app/demo/video_device2.mp4 /app/demo/test_video.mp4"
                else
                    VIDEO_ARGS="/app/demo/test_video.mp4"
                fi
            else
                VIDEO_ARGS="/app/demo/test_video.mp4"
            fi
        fi
        echo ""
        echo "Starting 8-Device Parallel Face Recognition Demo..."
        echo "  Videos: $VIDEO_ARGS"
        echo "  Browser: http://localhost:$STREAM_PORT"
        echo ""
        python3 /app/parallel_8device_stream.py $VIDEO_ARGS --port $STREAM_PORT
        ;;

    "webcam-server")
        echo ""
        echo "Starting Browser Webcam Server..."
        echo "  Browser: http://localhost:$STREAM_PORT"
        echo "  Click 'Webcam' tab to start face recognition"
        echo "  Click 'Register Face' tab to add faces"
        echo "  (Model compiles on first run ~2 min)"
        echo ""
        echo "  Tip: Add -v ./faces:/app/registered_faces to persist faces"
        echo ""
        export HTTP_MODE=webcam
        start_http_wrapper
        python3 /app/websocket_server.py "$@"
        ;;

    "benchmark")
        echo ""
        echo "======================================================================"
        echo "  GSTREAMER PIPELINE BENCHMARK MODE"
        echo "======================================================================"
        echo ""
        echo "Measuring timing for EVERY GStreamer element:"
        echo "  - decodebin (video decoder)"
        echo "  - videoconvert (color conversion)"
        echo "  - videoscale (resize)"
        echo "  - face_recognition (YuNet + SFace on TTNN)"
        echo "  - jpegenc (JPEG encoder)"
        echo ""
        
        VIDEO_PATH=${1:-/app/demo/test_video.mp4}
        
        # Run Python script with pad probes on each element
        python3 /app/benchmark_gst_elements.py "$VIDEO_PATH"
        ;;

    "accuracy-benchmark")
        echo ""
        echo "======================================================================"
        echo "  LFW ACCURACY & LATENCY BENCHMARK"
        echo "======================================================================"
        echo ""
        echo "Running face recognition benchmark on LFW dataset:"
        echo "  - PyTorch (CPU) baseline"
        echo "  - TTNN (Tenstorrent) accelerated"
        echo ""
        echo "Metrics: TPR @ 0.1% FAR, P95 latency"
        echo ""
        
        PAIRS=${1:-500}
        python3 /app/benchmark_lfw.py --pairs $PAIRS
        ;;

    "custom")
        echo "Running custom command..."
        exec "$@"
        ;;
    
    *)
        echo "Unknown mode: $MODE"
        echo "Run with --help for usage"
        exit 1
        ;;
esac
