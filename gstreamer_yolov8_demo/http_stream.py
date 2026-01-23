# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
HTTP Stream Server - Unified interface for YOLOv8s demo on Tenstorrent.

Serves on port 8080:
- /           : Main interface with input source options
- /stream     : MJPEG stream from GStreamer pipeline
- /webcam     : Webcam mode (browser captures, sends to TT, receives detections)
- /ws         : WebSocket endpoint for webcam mode

Usage: python http_stream.py [--mode stream|webcam|unified]
"""

import socket
import threading
import json
import base64
import io
import time
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

TCP_HOST = 'localhost'
TCP_PORT = 8081
HTTP_PORT = int(os.environ.get('STREAM_PORT', 8080))

# Mode: 'stream' = GStreamer pipeline, 'webcam' = browser webcam, 'unified' = both
MODE = os.environ.get('HTTP_MODE', 'stream')

# HTML for unified interface
UNIFIED_HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>YOLOv8s Demo - Tenstorrent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            min-height: 100vh;
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #fff;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 30px;
        }
        h1 {
            font-size: 2.5rem;
            font-weight: 600;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #888; margin-top: 10px; }
        
        /* Tab Navigation */
        .tabs {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 30px;
        }
        .tab {
            padding: 12px 30px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 1rem;
            color: #888;
        }
        .tab:hover { background: rgba(255,255,255,0.1); color: #fff; }
        .tab.active {
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            border-color: transparent;
            color: #fff;
        }
        
        /* Content Panels */
        .panel { display: none; }
        .panel.active { display: block; }
        
        .main-content {
            display: flex;
            gap: 30px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .video-container {
            position: relative;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            background: #000;
        }
        .video-container img, .video-container canvas, .video-container video {
            display: block;
            border-radius: 16px;
        }
        
        .sidebar {
            background: rgba(255,255,255,0.03);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 25px;
            min-width: 320px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .sidebar h3 {
            color: #00d4ff;
            margin-bottom: 20px;
            font-size: 1.1rem;
        }
        
        .form-group { margin-bottom: 20px; }
        label {
            display: block;
            color: #888;
            font-size: 0.75rem;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        input[type="text"], input[type="file"] {
            width: 100%;
            padding: 12px 15px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 8px;
            color: #fff;
            font-size: 0.95rem;
        }
        input:focus { outline: none; border-color: #00d4ff; }
        
        button {
            width: 100%;
            padding: 14px 20px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            margin-top: 10px;
        }
        .btn-primary {
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            color: #fff;
        }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(0,212,255,0.4); }
        .btn-danger {
            background: rgba(255,50,50,0.2);
            color: #ff5555;
            border: 1px solid #ff5555;
        }
        
        .stats {
            margin-top: 25px;
            padding-top: 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .stat-label { color: #666; }
        .stat-value { color: #00d4ff; font-family: 'SF Mono', monospace; font-weight: 600; }
        
        .status {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 0.9rem;
        }
        .status.info { background: rgba(0,150,255,0.1); color: #00aaff; }
        .status.success { background: rgba(0,255,100,0.1); color: #00ff66; }
        .status.warning { background: rgba(255,200,0,0.1); color: #ffcc00; }
        .status.error { background: rgba(255,50,50,0.1); color: #ff5555; }
        .status-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: currentColor;
        }
        .status.warning .status-dot { animation: pulse 1s infinite; }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.3; } }
        
        /* Loading spinner */
        .spinner {
            border: 3px solid rgba(255,255,255,0.1);
            border-top: 3px solid #00d4ff;
            border-radius: 50%;
            width: 50px; height: 50px;
            animation: spin 1s linear infinite;
            margin: 100px auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .hidden { display: none !important; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>YOLOv8s Object Detection</h1>
            <p class="subtitle">Real-time inference powered by Tenstorrent AI Accelerators</p>
        </header>
        
        <!-- Tab Navigation -->
        <div class="tabs">
            <div class="tab active" onclick="showTab('stream')">Video Stream</div>
            <div class="tab" onclick="showTab('webcam')">Webcam</div>
            <div class="tab" onclick="showTab('rtsp')">RTSP Camera</div>
        </div>
        
        <!-- Stream Panel (GStreamer MJPEG) -->
        <div id="panel-stream" class="panel active">
            <div class="main-content">
                <div class="video-container">
                    <div id="stream-loading" class="spinner"></div>
                    <img id="stream-img" src="/stream" width="640" height="640" 
                         style="display:none;" onload="streamLoaded()" onerror="streamError()">
                </div>
                <div class="sidebar">
                    <h3>Stream Status</h3>
                    <div id="stream-status" class="status warning">
                        <span class="status-dot"></span>
                        <span>Waiting for stream...</span>
                    </div>
                    <div class="stats">
                        <div class="stat-row">
                            <span class="stat-label">Input</span>
                            <span class="stat-value">Video Stream</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Resolution</span>
                            <span class="stat-value">640 x 640</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Hardware</span>
                            <span class="stat-value">Tenstorrent Wormhole</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Webcam Panel -->
        <div id="panel-webcam" class="panel">
            <div class="main-content">
                <div class="video-container">
                    <video id="webcam-video" width="640" height="480" autoplay playsinline style="display:none;"></video>
                    <canvas id="webcam-canvas" width="640" height="480"></canvas>
                </div>
                <div class="sidebar">
                    <h3>Browser Webcam Mode</h3>
                    <div id="webcam-status" class="status info">
                        <span class="status-dot"></span>
                        <span id="webcam-status-text">Requires webcam-server mode</span>
                    </div>
                    <p style="color:#888; font-size:0.85rem; margin-bottom:15px;">
                        To use your laptop webcam, restart container with:
                    </p>
                    <div style="position:relative;">
                        <pre id="webcam-cmd" style="background:#1a1a2e; padding:12px; border-radius:6px; font-size:0.75rem; color:#00d4ff; margin-bottom:15px; overflow-x:auto; white-space:pre-wrap;">docker run --rm --network host \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  tt-gstreamer-yolov8 webcam-server</pre>
                        <button onclick="copyCmd('webcam-cmd')" style="position:absolute; top:5px; right:5px; background:#333; border:none; color:#888; padding:5px 8px; border-radius:4px; cursor:pointer; font-size:0.7rem;">📋 Copy</button>
                    </div>
                    <button id="webcam-start" class="btn-primary" onclick="startWebcam()">Try Connect</button>
                    <button id="webcam-stop" class="btn-danger hidden" onclick="stopWebcam()">Stop</button>
                    <div class="stats">
                        <div class="stat-row">
                            <span class="stat-label">Inference</span>
                            <span class="stat-value" id="wc-inference">-</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">FPS</span>
                            <span class="stat-value" id="wc-fps">-</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Detections</span>
                            <span class="stat-value" id="wc-detections">-</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Roundtrip</span>
                            <span class="stat-value" id="wc-roundtrip">-</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- RTSP Panel -->
        <div id="panel-rtsp" class="panel">
            <div class="main-content">
                <div class="video-container">
                    <div id="rtsp-placeholder" style="width:640px;height:480px;display:flex;align-items:center;justify-content:center;background:#111;border-radius:16px;flex-direction:column;">
                        <p style="color:#888; margin-bottom:20px;">RTSP IP Camera Mode</p>
                        <p style="color:#666; font-size:0.85rem;">Restart container with your camera URL</p>
                    </div>
                </div>
                <div class="sidebar">
                    <h3>RTSP IP Camera</h3>
                    <div id="rtsp-status" class="status info">
                        <span class="status-dot"></span>
                        <span>Requires stream-rtsp mode</span>
                    </div>
                    <p style="color:#888; font-size:0.85rem; margin-bottom:15px;">
                        To connect to an IP camera, restart container with:
                    </p>
                    <div style="position:relative;">
                        <pre id="rtsp-cmd" style="background:#1a1a2e; padding:12px; border-radius:6px; font-size:0.75rem; color:#00d4ff; margin-bottom:15px; overflow-x:auto; white-space:pre-wrap;">docker run --rm --network host \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  tt-gstreamer-yolov8 stream-rtsp rtsp://CAMERA_IP:554/stream</pre>
                        <button onclick="copyCmd('rtsp-cmd')" style="position:absolute; top:5px; right:5px; background:#333; border:none; color:#888; padding:5px 8px; border-radius:4px; cursor:pointer; font-size:0.7rem;">📋 Copy</button>
                    </div>
                    <div class="stats">
                        <div class="stat-row">
                            <span class="stat-label">Protocol</span>
                            <span class="stat-value">RTSP over TCP</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Typical Cameras</span>
                            <span class="stat-value">Hikvision, Dahua, etc.</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const COCO_NAMES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"];
        const COLORS = ['#FF0000','#00FF00','#0000FF','#FFFF00','#FF00FF','#00FFFF'];
        
        let webcamWs = null;
        let webcamRunning = false;
        let sendTime = 0;
        
        function showTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelector(`.tab:nth-child(${tab==='stream'?1:tab==='webcam'?2:3})`).classList.add('active');
            document.getElementById('panel-' + tab).classList.add('active');
        }
        
        function streamLoaded() {
            document.getElementById('stream-loading').style.display = 'none';
            document.getElementById('stream-img').style.display = 'block';
            document.getElementById('stream-status').className = 'status success';
            document.getElementById('stream-status').innerHTML = '<span class="status-dot"></span><span>Stream active</span>';
        }
        
        function streamError() {
            setTimeout(() => {
                document.getElementById('stream-img').src = '/stream?' + Date.now();
            }, 2000);
        }
        
        async function startWebcam() {
            const video = document.getElementById('webcam-video');
            const canvas = document.getElementById('webcam-canvas');
            const ctx = canvas.getContext('2d');
            
            setWebcamStatus('warning', 'Starting webcam...');
            
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                video.srcObject = stream;
                await video.play();
                
                setWebcamStatus('warning', 'Connecting to server...');
                
                // Connect WebSocket - try port 8765
                const wsUrl = 'ws://' + window.location.hostname + ':8765';
                webcamWs = new WebSocket(wsUrl);
                
                webcamWs.onopen = () => {
                    setWebcamStatus('success', 'Connected - Running');
                    webcamRunning = true;
                    document.getElementById('webcam-start').classList.add('hidden');
                    document.getElementById('webcam-stop').classList.remove('hidden');
                    sendWebcamFrame(video, canvas, ctx);
                };
                
                webcamWs.onmessage = (e) => {
                    try {
                        const data = JSON.parse(e.data);
                        const rt = Date.now() - sendTime;
                        document.getElementById('wc-inference').textContent = data.inference_ms.toFixed(1) + ' ms';
                        document.getElementById('wc-fps').textContent = (1000/data.inference_ms).toFixed(0);
                        document.getElementById('wc-detections').textContent = data.count;
                        document.getElementById('wc-roundtrip').textContent = rt + ' ms';
                        drawWebcamFrame(video, canvas, ctx, data.detections);
                        if (webcamRunning) {
                            requestAnimationFrame(() => sendWebcamFrame(video, canvas, ctx));
                        }
                    } catch (err) {
                        console.error('onmessage error:', err);
                        setWebcamStatus('error', 'Error: ' + err.message);
                    }
                };
                
                webcamWs.onerror = (e) => { 
                    console.error('WebSocket error:', e);
                    setWebcamStatus('warning', 'Connection error - check console'); 
                };
                webcamWs.onclose = (e) => { 
                    console.log('WebSocket closed:', e.code, e.reason);
                    if (webcamRunning) stopWebcam(); 
                };
            } catch (err) {
                setWebcamStatus('error', 'Error: ' + err.message);
            }
        }
        
        function stopWebcam() {
            webcamRunning = false;
            if (webcamWs) webcamWs.close();
            const video = document.getElementById('webcam-video');
            if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
            setWebcamStatus('info', 'Stopped');
            document.getElementById('webcam-start').classList.remove('hidden');
            document.getElementById('webcam-stop').classList.add('hidden');
        }
        
        function setWebcamStatus(type, text) {
            const el = document.getElementById('webcam-status');
            el.className = 'status ' + type;
            el.innerHTML = '<span class="status-dot"></span><span>' + text + '</span>';
        }
        
        function sendWebcamFrame(video, canvas, ctx) {
            if (!webcamRunning || !webcamWs || webcamWs.readyState !== 1) return;
            // Use a temporary canvas to capture frame WITHOUT overwriting display canvas
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 640;
            tempCanvas.height = 480;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(video, 0, 0, 640, 480);
            tempCanvas.toBlob((blob) => {
                const reader = new FileReader();
                reader.onload = () => {
                    sendTime = Date.now();
                    webcamWs.send(JSON.stringify({ frame: reader.result.split(',')[1] }));
                };
                reader.readAsDataURL(blob);
            }, 'image/jpeg', 0.7);
        }
        
        function drawWebcamFrame(video, canvas, ctx, detections) {
            ctx.drawImage(video, 0, 0, 640, 480);
            ctx.lineWidth = 3;
            ctx.font = '16px sans-serif';
            console.log('Drawing', detections.length, 'detections:', detections);
            for (const d of detections) {
                const color = COLORS[d.class_id % 6];
                const label = COCO_NAMES[d.class_id] + ': ' + (d.confidence*100).toFixed(0) + '%';
                console.log('Box:', d.x1, d.y1, d.x2, d.y2, label);
                ctx.strokeStyle = color;
                ctx.strokeRect(d.x1, d.y1, d.x2-d.x1, d.y2-d.y1);
                ctx.fillStyle = color;
                ctx.fillRect(d.x1, d.y1-18, ctx.measureText(label).width+8, 18);
                ctx.fillStyle = '#fff';
                ctx.fillText(label, d.x1+4, d.y1-4);
            }
        }
        
        function copyCmd(id) {
            const text = document.getElementById(id).innerText;
            navigator.clipboard.writeText(text).then(() => {
                const btn = event.target;
                btn.innerText = '✓ Copied!';
                setTimeout(() => { btn.innerText = '📋 Copy'; }, 2000);
            });
        }
        
        // Auto-retry stream
        setTimeout(() => {
            const img = document.getElementById('stream-img');
            if (img.complete && img.naturalHeight === 0) streamError();
        }, 3000);
    </script>
</body>
</html>
'''


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(UNIFIED_HTML.encode('utf-8'))
            
        elif self.path.startswith('/stream'):
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((TCP_HOST, TCP_PORT))
                sock.settimeout(5.0)
                
                while True:
                    data = sock.recv(65536)
                    if not data:
                        break
                    self.wfile.write(data)
                    self.wfile.flush()
            except Exception as e:
                print(f"Stream error: {e}")
            finally:
                sock.close()
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        pass  # Suppress logging


if __name__ == '__main__':
    print(f"=" * 50)
    print(f"  YOLOv8s Demo - HTTP Server")
    print(f"=" * 50)
    print(f"  Open: http://localhost:{HTTP_PORT}")
    print(f"  Stream from GStreamer TCP port: {TCP_PORT}")
    print(f"=" * 50)
    
    server = HTTPServer(('0.0.0.0', HTTP_PORT), StreamHandler)
    server.serve_forever()
