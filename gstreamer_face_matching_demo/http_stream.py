# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
HTTP Stream Server - Face Recognition Demo on Tenstorrent.

Serves on port 8080:
- /           : Main interface with input source options
- /stream     : MJPEG stream from GStreamer pipeline
- /webcam     : Webcam mode (browser captures, sends to TT, receives results)
- /ws         : WebSocket endpoint for webcam mode
- /register   : Register a new face
- /faces      : List registered faces
"""

import socket
import threading
import json
import base64
import io
import time
import os
import numpy as np
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

TCP_HOST = 'localhost'
TCP_PORT = 8081
HTTP_PORT = int(os.environ.get('STREAM_PORT', 8080))

# Face storage
FACES_DIR = Path("/app/registered_faces")
FACES_DIR.mkdir(exist_ok=True)

MODE = os.environ.get('HTTP_MODE', 'stream')

UNIFIED_HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition - Tenstorrent</title>
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
            background: linear-gradient(90deg, #ff6b6b, #feca57);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #888; margin-top: 10px; }
        
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
            background: linear-gradient(90deg, #ff6b6b, #feca57);
            border-color: transparent;
            color: #fff;
        }
        
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
            color: #ff6b6b;
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
        input:focus { outline: none; border-color: #ff6b6b; }
        
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
            background: linear-gradient(90deg, #ff6b6b, #feca57);
            color: #fff;
        }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(255,107,107,0.4); }
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
        .stat-value { color: #ff6b6b; font-family: 'SF Mono', monospace; font-weight: 600; }
        
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
        
        .spinner {
            border: 3px solid rgba(255,255,255,0.1);
            border-top: 3px solid #ff6b6b;
            border-radius: 50%;
            width: 50px; height: 50px;
            animation: spin 1s linear infinite;
            margin: 100px auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .hidden { display: none !important; }
        
        .face-list {
            max-height: 200px;
            overflow-y: auto;
            margin-top: 15px;
        }
        .face-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            margin-bottom: 8px;
        }
        .face-item span { color: #fff; }
        .face-item button {
            width: auto;
            padding: 5px 12px;
            font-size: 0.8rem;
            margin: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 Face Recognition</h1>
            <p class="subtitle">YuNet + SFace powered by Tenstorrent AI Accelerators</p>
        </header>
        
        <script>const SERVER_MODE = '{{MODE}}';</script>
        
        <div class="tabs">
            <div id="tab-stream" class="tab" onclick="showTab('stream')">Video Stream</div>
            <div id="tab-webcam" class="tab" onclick="showTab('webcam')">Webcam</div>
            <div id="tab-register" class="tab" onclick="showTab('register')">Register Face</div>
            <div id="tab-rtsp" class="tab" onclick="showTab('rtsp')">RTSP Camera</div>
        </div>
        
        <!-- Stream Panel -->
        <div id="panel-stream" class="panel">
            <div class="main-content">
                <div class="video-container">
                    <div id="stream-placeholder" style="width:640px;height:640px;display:flex;align-items:center;justify-content:center;background:#111;border-radius:16px;flex-direction:column;">
                        <p style="color:#888; font-size:1.2rem; margin-bottom:20px;">🎬 Video Stream</p>
                        <p style="color:#666; font-size:0.9rem;">Select a source to start</p>
                    </div>
                    <img id="stream-img" width="640" height="640" style="display:none;">
                    <video id="test-video" width="640" height="640" style="display:none;"></video>
                    <canvas id="test-canvas" width="640" height="640" style="display:none;"></canvas>
                </div>
                <div class="sidebar">
                    <h3>Video Source</h3>
                    <div id="stream-status" class="status info">
                        <span class="status-dot"></span>
                        <span>Select a video source</span>
                    </div>
                    
                    <div style="margin: 15px 0;">
                        <button id="test-video-btn" class="btn-primary" onclick="loadTestVideo()" style="width:100%; margin-bottom:10px;">
                            ▶ Play Test Video
                        </button>
                        <div style="text-align:center; color:#666; font-size:0.8rem; margin: 8px 0;">or</div>
                        <label style="display:block;">
                            <input type="file" id="video-upload" accept="video/*" style="display:none;" onchange="loadUploadedVideo(this)">
                            <span id="upload-btn" class="btn-primary" style="display:block; text-align:center; cursor:pointer;">
                                📁 Upload Video File
                            </span>
                        </label>
                        <button id="stop-video-btn" class="btn-danger hidden" onclick="stopVideoStream()" style="width:100%; margin-top:10px;">
                            ⏹ Stop
                        </button>
                    </div>
                    
                    <div class="stats" id="video-stats" style="display:none;">
                        <div class="stat-row">
                            <span class="stat-label">Inference</span>
                            <span class="stat-value" id="vid-inference">-</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">FPS</span>
                            <span class="stat-value" id="vid-fps">-</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Faces</span>
                            <span class="stat-value" id="vid-faces">-</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Webcam Panel -->
        <div id="panel-webcam" class="panel">
            <div class="main-content">
                <div class="video-container">
                    <video id="webcam-video" width="640" height="640" autoplay playsinline style="display:none;"></video>
                    <canvas id="webcam-canvas" width="640" height="640"></canvas>
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
                        <pre id="webcam-cmd" style="background:#1a1a2e; padding:12px; border-radius:6px; font-size:0.7rem; color:#ff6b6b; margin-bottom:15px; overflow-x:auto; white-space:pre-wrap;">docker run --rm --network host \\
  --device /dev/tenstorrent:/dev/tenstorrent \\
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \\
  ghcr.io/tenstorrent/tt-inference-server/face-matching-demo webcam-server</pre>
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
                            <span class="stat-label">Faces</span>
                            <span class="stat-value" id="wc-faces">-</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Register Face Panel -->
        <div id="panel-register" class="panel">
            <div class="main-content">
                <div class="sidebar" style="min-width: 450px;">
                    <h3>Register New Face</h3>
                    <p style="color:#888; font-size:0.8rem; margin-bottom:15px;">Upload 1-3 photos for better accuracy</p>
                    <div class="form-group">
                        <label>Name</label>
                        <input type="text" id="reg-name" placeholder="Enter person's name">
                    </div>
                    <div class="form-group">
                        <label>Photo 1 (required)</label>
                        <input type="file" id="reg-photo-1" accept="image/*">
                    </div>
                    <div class="form-group">
                        <label>Photo 2 (optional)</label>
                        <input type="file" id="reg-photo-2" accept="image/*">
                    </div>
                    <div class="form-group">
                        <label>Photo 3 (optional)</label>
                        <input type="file" id="reg-photo-3" accept="image/*">
                    </div>
                    <button class="btn-primary" onclick="registerFace()">Register Face</button>
                    <div id="reg-result" style="margin-top:15px; padding:10px; border-radius:6px; display:none;"></div>
                    
                    <h3 style="margin-top:30px;">Registered Faces (<span id="face-count">0</span>)</h3>
                    <div id="face-list" class="face-list" style="max-height:300px; overflow-y:auto;">
                        <p style="color:#666;">Loading...</p>
                    </div>
                    <button class="btn-danger" onclick="loadFaces()" style="margin-top:10px;">Refresh List</button>
                    
                    <div style="margin-top:20px; padding:12px; background:rgba(100,100,255,0.1); border-radius:8px; border:1px solid rgba(100,100,255,0.3);">
                        <p style="color:#888; font-size:0.75rem; margin:0;">
                            💡 <strong>Persist faces across restarts:</strong><br>
                            Add <code>-v ./faces:/app/registered_faces</code> to your docker run command
                        </p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- RTSP Panel -->
        <div id="panel-rtsp" class="panel">
            <div class="main-content">
                <div class="video-container">
                    <div style="width:640px;height:640px;display:flex;align-items:center;justify-content:center;background:#111;border-radius:16px;flex-direction:column;">
                        <p style="color:#888; margin-bottom:20px;">RTSP IP Camera Mode</p>
                        <p style="color:#666; font-size:0.85rem;">Restart container with your camera URL</p>
                    </div>
                </div>
                <div class="sidebar">
                    <h3>RTSP IP Camera</h3>
                    <div class="status info">
                        <span class="status-dot"></span>
                        <span>Requires stream-rtsp mode</span>
                    </div>
                    <p style="color:#888; font-size:0.85rem; margin-bottom:15px;">
                        To connect to an IP camera:
                    </p>
                    <div style="position:relative;">
                        <pre id="rtsp-cmd" style="background:#1a1a2e; padding:12px; border-radius:6px; font-size:0.7rem; color:#ff6b6b; margin-bottom:15px; overflow-x:auto; white-space:pre-wrap;">docker run --rm --network host \\
  --device /dev/tenstorrent:/dev/tenstorrent \\
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \\
  -v ./faces:/app/registered_faces \\
  ghcr.io/tenstorrent/tt-inference-server/face-matching-demo rtsp &lt;RTSP_URL&gt;</pre>
                        <button onclick="copyCmd('rtsp-cmd')" style="position:absolute; top:5px; right:5px; background:#333; border:none; color:#888; padding:5px 8px; border-radius:4px; cursor:pointer; font-size:0.7rem;">📋 Copy</button>
                    </div>
                    <p style="color:#666; font-size:0.75rem; margin-top:10px;">
                        💡 Use <code>-v ./faces:/app/registered_faces</code> to persist faces across restarts
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let webcamWs = null;
        let webcamRunning = false;
        let sendTime = 0;
        
        // Video file playback
        let videoWs = null;
        let videoRunning = false;
        let videoSendTime = 0;
        
        function showTab(tab) {
            document.querySelectorAll('.tab').forEach((t, i) => {
                t.classList.remove('active');
                if ((tab==='stream' && i===0) || (tab==='webcam' && i===1) || 
                    (tab==='register' && i===2) || (tab==='rtsp' && i===3)) {
                    t.classList.add('active');
                }
            });
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.getElementById('panel-' + tab).classList.add('active');
            
            if (tab === 'register') loadFaces();
        }
        
        // Video file playback via WebSocket
        let lastFrameTime = 0;
        let frameSkipInterval = 100; // Send frame every 100ms (10 fps input)
        let currentVideoName = '';
        
        function loadTestVideo() {
            const video = document.getElementById('test-video');
            
            // Show loading status
            document.getElementById('stream-status').className = 'status warning';
            document.getElementById('stream-status').innerHTML = '<span class="status-dot"></span><span>Loading test video...</span>';
            
            video.onerror = (e) => {
                console.error('Video load error:', e);
                document.getElementById('stream-status').className = 'status error';
                document.getElementById('stream-status').innerHTML = '<span class="status-dot"></span><span>Failed to load video</span>';
            };
            
            video.oncanplay = () => {
                console.log('Video can play, starting stream...');
                currentVideoName = 'test_video.mp4';
                startVideoStream(video, true);
            };
            
            video.src = '/demo/test_video.mp4';
            video.load();
        }
        
        function loadUploadedVideo(input) {
            if (input.files && input.files[0]) {
                const video = document.getElementById('test-video');
                video.src = URL.createObjectURL(input.files[0]);
                currentVideoName = input.files[0].name;
                video.load();
                video.onloadeddata = () => startVideoStream(video, false);
            }
        }
        
        function startVideoStream(video, isTestVideo) {
            const canvas = document.getElementById('test-canvas');
            const ctx = canvas.getContext('2d');
            
            // Hide placeholder, show canvas
            document.getElementById('stream-placeholder').style.display = 'none';
            document.getElementById('stream-img').style.display = 'none';
            video.style.display = 'none';
            canvas.style.display = 'block';
            document.getElementById('video-stats').style.display = 'block';
            document.getElementById('stop-video-btn').classList.remove('hidden');
            document.getElementById('test-video-btn').classList.add('hidden');
            document.getElementById('upload-btn').parentElement.classList.add('hidden');
            
            const wsUrl = 'ws://' + window.location.hostname + ':8765';
            videoWs = new WebSocket(wsUrl);
            
            videoWs.onopen = () => {
                document.getElementById('stream-status').className = 'status success';
                document.getElementById('stream-status').innerHTML = '<span class="status-dot"></span><span>Processing: ' + currentVideoName + '</span>';
                videoRunning = true;
                lastFrameTime = 0;
                video.play();
                sendVideoFrame(video, canvas, ctx);
            };
            
            videoWs.onmessage = (e) => {
                try {
                    const data = JSON.parse(e.data);
                    if (data.inference_ms !== undefined) {
                        document.getElementById('vid-inference').textContent = data.inference_ms.toFixed(1) + ' ms';
                        document.getElementById('vid-fps').textContent = (1000/data.inference_ms).toFixed(0);
                        document.getElementById('vid-faces').textContent = data.count || 0;
                        drawVideoFaces(video, canvas, ctx, data.faces || []);
                    }
                    if (videoRunning && !video.paused && !video.ended) {
                        // Wait for next frame with throttling
                        setTimeout(() => sendVideoFrame(video, canvas, ctx), 30);
                    }
                } catch(err) {
                    console.error('Parse error:', err);
                }
            };
            
            videoWs.onerror = () => {
                document.getElementById('stream-status').className = 'status error';
                document.getElementById('stream-status').innerHTML = '<span class="status-dot"></span><span>Connection failed - is webcam-server running?</span>';
            };
            
            video.onended = () => {
                // Loop video if it's the test video
                if (isTestVideo && videoRunning) {
                    video.currentTime = 0;
                    video.play();
                } else {
                    stopVideoStream();
                    document.getElementById('stream-status').className = 'status success';
                    document.getElementById('stream-status').innerHTML = '<span class="status-dot"></span><span>Video finished</span>';
                }
            };
        }
        
        function sendVideoFrame(video, canvas, ctx) {
            if (!videoRunning || video.paused || video.ended) return;
            if (!videoWs || videoWs.readyState !== 1) return;
            
            // Draw current frame
            ctx.drawImage(video, 0, 0, 640, 640);
            
            canvas.toBlob((blob) => {
                if (!blob) return;
                const reader = new FileReader();
                reader.onload = () => {
                    videoSendTime = Date.now();
                    try {
                        videoWs.send(JSON.stringify({ frame: reader.result.split(',')[1] }));
                    } catch(e) {
                        console.error('Send error:', e);
                    }
                };
                reader.readAsDataURL(blob);
            }, 'image/jpeg', 0.8);
        }
        
        function drawVideoFaces(video, canvas, ctx, faces) {
            // Redraw current video frame
            ctx.drawImage(video, 0, 0, 640, 640);
            ctx.lineWidth = 3;
            ctx.font = '16px sans-serif';
            
            for (const f of faces) {
                const isRecognized = f.identity !== 'Unknown';
                const color = isRecognized ? '#00ff66' : '#ff5555';
                
                // Draw box
                ctx.strokeStyle = color;
                ctx.strokeRect(f.x1, f.y1, f.x2-f.x1, f.y2-f.y1);
                
                // Only show label for recognized faces (green)
                if (isRecognized) {
                    const label = f.identity + ': ' + (f.score * 100).toFixed(0) + '%';
                    ctx.fillStyle = color;
                    ctx.fillRect(f.x1, f.y1-20, ctx.measureText(label).width+8, 20);
                    ctx.fillStyle = '#000';
                    ctx.fillText(label, f.x1+4, f.y1-5);
                }
                // Unknown faces: just red box, no text
            }
        }
        
        function stopVideoStream() {
            videoRunning = false;
            if (videoWs) {
                try { videoWs.close(); } catch(e) {}
            }
            const video = document.getElementById('test-video');
            video.pause();
            video.currentTime = 0;
            document.getElementById('test-canvas').style.display = 'none';
            document.getElementById('stream-placeholder').style.display = 'flex';
            document.getElementById('video-stats').style.display = 'none';
            document.getElementById('stop-video-btn').classList.add('hidden');
            document.getElementById('test-video-btn').classList.remove('hidden');
            document.getElementById('upload-btn').parentElement.classList.remove('hidden');
            document.getElementById('stream-status').className = 'status info';
            document.getElementById('stream-status').innerHTML = '<span class="status-dot"></span><span>Select a video source</span>';
        }
        
        async function startWebcam() {
            const video = document.getElementById('webcam-video');
            const canvas = document.getElementById('webcam-canvas');
            const ctx = canvas.getContext('2d');
            
            setWebcamStatus('warning', 'Starting webcam...');
            
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 640 } });
                video.srcObject = stream;
                await video.play();
                
                setWebcamStatus('warning', 'Connecting to server...');
                
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
                        document.getElementById('wc-faces').textContent = data.count;
                        drawFaces(video, canvas, ctx, data.faces);
                        if (webcamRunning) {
                            requestAnimationFrame(() => sendWebcamFrame(video, canvas, ctx));
                        }
                    } catch (err) {
                        console.error('Error:', err);
                    }
                };
                
                webcamWs.onerror = () => setWebcamStatus('warning', 'Connection error');
                webcamWs.onclose = () => { if (webcamRunning) stopWebcam(); };
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
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 640;
            tempCanvas.height = 480;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(video, 0, 0, 640, 640);
            tempCanvas.toBlob((blob) => {
                const reader = new FileReader();
                reader.onload = () => {
                    sendTime = Date.now();
                    webcamWs.send(JSON.stringify({ frame: reader.result.split(',')[1] }));
                };
                reader.readAsDataURL(blob);
            }, 'image/jpeg', 0.7);
        }
        
        function drawFaces(video, canvas, ctx, faces) {
            ctx.drawImage(video, 0, 0, 640, 640);
            ctx.lineWidth = 3;
            ctx.font = '16px sans-serif';
            
            for (const f of faces) {
                const isRecognized = f.identity !== 'Unknown';
                const color = isRecognized ? '#00ff66' : '#ff5555';
                
                // Draw box
                ctx.strokeStyle = color;
                ctx.strokeRect(f.x1, f.y1, f.x2-f.x1, f.y2-f.y1);
                
                // Only show label for recognized faces (green)
                if (isRecognized) {
                    const label = f.identity + ': ' + (f.score * 100).toFixed(0) + '%';
                    ctx.fillStyle = color;
                    ctx.fillRect(f.x1, f.y1-20, ctx.measureText(label).width+8, 20);
                    ctx.fillStyle = '#000';
                    ctx.fillText(label, f.x1+4, f.y1-5);
                }
                // Unknown faces: just red box, no text
            }
        }
        
        async function registerFace() {
            const name = document.getElementById('reg-name').value.trim();
            const photo1 = document.getElementById('reg-photo-1');
            const photo2 = document.getElementById('reg-photo-2');
            const photo3 = document.getElementById('reg-photo-3');
            const resultDiv = document.getElementById('reg-result');
            
            if (!name) {
                resultDiv.style.display = 'block';
                resultDiv.style.background = 'rgba(255,50,50,0.2)';
                resultDiv.style.color = '#ff5555';
                resultDiv.textContent = 'Please enter a name';
                return;
            }
            
            if (!photo1.files[0]) {
                resultDiv.style.display = 'block';
                resultDiv.style.background = 'rgba(255,50,50,0.2)';
                resultDiv.style.color = '#ff5555';
                resultDiv.textContent = 'Please select at least Photo 1';
                return;
            }
            
            // Collect all photos
            const photos = [photo1.files[0]];
            if (photo2.files[0]) photos.push(photo2.files[0]);
            if (photo3.files[0]) photos.push(photo3.files[0]);
            
            resultDiv.style.display = 'block';
            resultDiv.style.background = 'rgba(255,200,0,0.2)';
            resultDiv.style.color = '#ffcc00';
            resultDiv.textContent = 'Registering ' + photos.length + ' photo(s)...';
            
            // Convert all photos to base64
            const frames = [];
            for (const photo of photos) {
                const base64 = await readFileAsBase64(photo);
                frames.push(base64);
            }
            
            // Try WebSocket first, fall back to HTTP
            const wsUrl = 'ws://' + window.location.hostname + ':8765';
            
            async function registerViaHTTP() {
                try {
                    const resp = await fetch('/register', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ name: name, frames: frames })
                    });
                    const data = await resp.json();
                    if (data.success) {
                        resultDiv.style.background = 'rgba(0,255,100,0.2)';
                        resultDiv.style.color = '#00ff66';
                        resultDiv.textContent = data.message;
                        document.getElementById('reg-name').value = '';
                        photo1.value = '';
                        photo2.value = '';
                        photo3.value = '';
                        loadFaces();
                    } else {
                        resultDiv.style.background = 'rgba(255,50,50,0.2)';
                        resultDiv.style.color = '#ff5555';
                        resultDiv.textContent = data.message;
                    }
                } catch (e) {
                    resultDiv.style.background = 'rgba(255,50,50,0.2)';
                    resultDiv.style.color = '#ff5555';
                    resultDiv.textContent = 'Registration failed: ' + e.message;
                }
            }
            
            // Try WebSocket (for webcam mode), fall back to HTTP (for other modes)
            const ws = new WebSocket(wsUrl);
            let wsConnected = false;
            let responseReceived = false;
            
            // Longer timeout for image processing (30 seconds)
            setTimeout(() => {
                if (!responseReceived) {
                    ws.close();
                    if (!wsConnected) {
                        // WebSocket never connected - use HTTP
                        registerViaHTTP();
                    } else {
                        resultDiv.style.background = 'rgba(255,50,50,0.2)';
                        resultDiv.style.color = '#ff5555';
                        resultDiv.textContent = 'Registration timed out. Please try again.';
                    }
                }
            }, 30000);
            
            ws.onopen = () => {
                wsConnected = true;
                ws.send(JSON.stringify({
                    register: { name: name, frames: frames }
                }));
            };
            
            ws.onmessage = (e) => {
                responseReceived = true;
                const data = JSON.parse(e.data);
                if (data.register_result) {
                    if (data.register_result.success) {
                        resultDiv.style.background = 'rgba(0,255,100,0.2)';
                        resultDiv.style.color = '#00ff66';
                        resultDiv.textContent = data.register_result.message;
                        document.getElementById('reg-name').value = '';
                        photo1.value = '';
                        photo2.value = '';
                        photo3.value = '';
                        loadFaces();
                    } else {
                        resultDiv.style.background = 'rgba(255,50,50,0.2)';
                        resultDiv.style.color = '#ff5555';
                        resultDiv.textContent = data.register_result.message;
                    }
                }
                ws.close();
            };
            
            ws.onerror = () => {
                // WebSocket not available, use HTTP
                registerViaHTTP();
            };
        }
        
        function readFileAsBase64(file) {
            return new Promise((resolve) => {
                const img = new Image();
                img.onload = () => {
                    // Resize to max 640px and compress
                    const canvas = document.createElement('canvas');
                    const maxSize = 640;
                    let w = img.width, h = img.height;
                    if (w > h && w > maxSize) { h = h * maxSize / w; w = maxSize; }
                    else if (h > maxSize) { w = w * maxSize / h; h = maxSize; }
                    canvas.width = w;
                    canvas.height = h;
                    canvas.getContext('2d').drawImage(img, 0, 0, w, h);
                    // Compress to JPEG 80% quality
                    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
                    resolve(dataUrl.split(',')[1]);
                };
                img.src = URL.createObjectURL(file);
            });
        }
        
        async function loadFaces() {
            const listDiv = document.getElementById('face-list');
            const countEl = document.getElementById('face-count');
            
            function displayFaces(faces) {
                if (faces && faces.length > 0) {
                    countEl.textContent = faces.length;
                    listDiv.innerHTML = faces.map(name => 
                        '<div class="face-item"><span>' + name + '</span>' +
                        '<button class="btn-danger" onclick="deleteFace(&apos;' + name + '&apos;)">Delete</button></div>'
                    ).join('');
                } else {
                    countEl.textContent = '0';
                    listDiv.innerHTML = '<p style="color:#666;">No faces registered</p>';
                }
            }
            
            // Try HTTP first (always works)
            try {
                const resp = await fetch('/faces');
                const data = await resp.json();
                displayFaces(data.faces);
            } catch(e) {
                listDiv.innerHTML = '<p style="color:#ff5555;">Failed to load faces</p>';
            }
        }
        
        async function deleteFace(name) {
            if (!confirm('Delete face for ' + name + '?')) return;
            
            try {
                const resp = await fetch('/faces/' + encodeURIComponent(name), { method: 'DELETE' });
                const data = await resp.json();
                if (data.deleted) {
                    loadFaces();
                } else {
                    alert(data.error || 'Delete failed');
                }
            } catch(e) {
                alert('Delete failed: ' + e.message);
            }
        }
        
        function copyCmd(id) {
            const text = document.getElementById(id).innerText;
            navigator.clipboard.writeText(text).then(() => {
                event.target.innerText = '✓ Copied!';
                setTimeout(() => { event.target.innerText = '📋 Copy'; }, 2000);
            });
        }
        
        // Initialize page based on mode
        document.addEventListener('DOMContentLoaded', () => {
            if (SERVER_MODE === 'webcam') {
                // Default to Webcam tab in webcam mode
                showTab('webcam');
            } else {
                // Default to Video Stream tab (video won't auto-start)
                showTab('stream');
            }
        });
    </script>
</body>
</html>
'''


# Face management functions (works without TT device - just saves to disk)
def list_registered_faces():
    """List all registered faces from disk."""
    faces = []
    if FACES_DIR.exists():
        for person_dir in FACES_DIR.iterdir():
            if person_dir.is_dir():
                faces.append(person_dir.name)
    return sorted(faces)


def save_face_image(name, image_data):
    """Save a face image for later embedding extraction.
    
    The actual embedding will be extracted when the TT model loads.
    """
    import cv2
    
    # Decode image
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return False, "Invalid image"
    
    # Use OpenCV's Haar cascade for CPU-based face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return False, "No face detected"
    
    # Use the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    margin = int(max(w, h) * 0.1)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img.shape[1], x + w + margin)
    y2 = min(img.shape[0], y + h + margin)
    
    face_crop = img[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, (112, 112))
    
    # Save to disk
    person_dir = FACES_DIR / name
    person_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(person_dir / "face.jpg"), face_resized)
    
    # Mark as needing embedding (will be processed when model loads)
    pending_file = person_dir / "pending_embedding"
    pending_file.touch()
    
    return True, f"Face saved for {name}. Will be activated when model loads."


def delete_face(name):
    """Delete a face from disk."""
    import shutil
    person_dir = FACES_DIR / name
    if person_dir.exists():
        shutil.rmtree(person_dir)
        return True
    return False


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            # Replace mode placeholder with actual mode
            html = UNIFIED_HTML.replace('{{MODE}}', MODE)
            self.wfile.write(html.encode('utf-8'))
            
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
                
        elif self.path == '/faces':
            # Return list of registered faces from disk
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            faces = list_registered_faces()
            self.wfile.write(json.dumps({"faces": faces}).encode())
        
        elif self.path.startswith('/demo/'):
            # Serve demo files (test videos) with Range support
            import mimetypes
            file_path = '/app' + self.path
            if os.path.exists(file_path):
                mime_type, _ = mimetypes.guess_type(file_path)
                file_size = os.path.getsize(file_path)
                
                # Check for Range header (required for video seeking)
                range_header = self.headers.get('Range')
                if range_header:
                    # Parse range
                    range_match = range_header.replace('bytes=', '').split('-')
                    start = int(range_match[0]) if range_match[0] else 0
                    end = int(range_match[1]) if range_match[1] else file_size - 1
                    
                    self.send_response(206)  # Partial content
                    self.send_header('Content-type', mime_type or 'application/octet-stream')
                    self.send_header('Accept-Ranges', 'bytes')
                    self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                    self.send_header('Content-Length', str(end - start + 1))
                    self.end_headers()
                    
                    with open(file_path, 'rb') as f:
                        f.seek(start)
                        self.wfile.write(f.read(end - start + 1))
                else:
                    # Full file
                    self.send_response(200)
                    self.send_header('Content-type', mime_type or 'application/octet-stream')
                    self.send_header('Accept-Ranges', 'bytes')
                    self.send_header('Content-Length', str(file_size))
                    self.end_headers()
                    with open(file_path, 'rb') as f:
                        self.wfile.write(f.read())
            else:
                self.send_error(404)
            
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/register':
            # Handle face registration via HTTP (works in all modes)
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                name = data.get('name', '').strip()
                frames = data.get('frames', [])
                
                # Legacy support for single frame
                if not frames and 'frame' in data:
                    frames = [data['frame']]
                
                if not name:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "message": "Name required"}).encode())
                    return
                
                if not frames:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "message": "At least one photo required"}).encode())
                    return
                
                # Save the first photo (CPU-based face detection)
                success, message = save_face_image(name, frames[0])
                
                # Save additional photos if provided
                if success and len(frames) > 1:
                    import cv2
                    person_dir = FACES_DIR / name
                    for i, frame in enumerate(frames[1:], 2):
                        try:
                            nparr = np.frombuffer(base64.b64decode(frame), np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if img is not None:
                                cv2.imwrite(str(person_dir / f"face_{i}.jpg"), cv2.resize(img, (112, 112)))
                        except:
                            pass
                    message = f"Saved {len(frames)} photos for {name}. Will be activated when model loads."
                
                self.send_response(200 if success else 400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": success, "message": message}).encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "message": str(e)}).encode())
        else:
            self.send_error(404)
    
    def do_DELETE(self):
        if self.path.startswith('/faces/'):
            name = self.path.split('/faces/')[1]
            if delete_face(name):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"deleted": True, "name": name}).encode())
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"deleted": False, "error": "Face not found"}).encode())
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    print(f"=" * 50)
    print(f"  Face Recognition Demo - HTTP Server")
    print(f"=" * 50)
    print(f"  Open: http://localhost:{HTTP_PORT}")
    print(f"  Stream from GStreamer TCP port: {TCP_PORT}")
    print(f"=" * 50)
    
    server = HTTPServer(('0.0.0.0', HTTP_PORT), StreamHandler)
    server.serve_forever()
