#!/usr/bin/env python3
"""
GStreamer YOLOv8s Plugin for Tenstorrent Devices
Uses GstElement with chain function for reliable frame processing.
Supports mesh devices (T3K, QuietBox) with multiple chips.
"""
import os
import sys
import gi
import time
import numpy as np
import cv2
import torch

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject

# Force unbuffered output
sys.stdout = sys.stderr

import ttnn
from models.demos.yolov8s.runner.performant_runner import YOLOv8sPerformantRunner
from models.demos.yolov8s.common import YOLOV8S_L1_SMALL_SIZE
from models.demos.utils.common_demo_utils import load_coco_class_names, postprocess as obj_postprocess, get_mesh_mappers

# Initialize GStreamer
Gst.init(None)


class Yolov8s(Gst.Element):
    """GStreamer element for YOLOv8s inference on Tenstorrent devices."""

    __gtype_name__ = "GstYolov8sTT"

    __gstmetadata__ = (
        "YOLOv8s Tenstorrent",
        "Filter/Effect/Video",
        "YOLOv8s object detection on Tenstorrent hardware",
        "Tenstorrent",
    )

    _sink_template = Gst.PadTemplate.new(
        "sink",
        Gst.PadDirection.SINK,
        Gst.PadPresence.ALWAYS,
        Gst.Caps.from_string("video/x-raw,format=BGRx,width=640,height=640"),
    )
    _src_template = Gst.PadTemplate.new(
        "src",
        Gst.PadDirection.SRC,
        Gst.PadPresence.ALWAYS,
        Gst.Caps.from_string("video/x-raw,format=BGRx,width=640,height=640"),
    )
    __gsttemplates__ = (_sink_template, _src_template)

    __gproperties__ = {
        "device-id": (int, "Device ID", "TT Device ID", 0, 7, 0, GObject.ParamFlags.READWRITE),
    }

    def __init__(self):
        super().__init__()
        print("[YOLOv8s] __init__ called", flush=True)
        
        self.device_id = 0
        self.model = None
        self.device = None
        self.mesh_composer = None
        self.names = None
        self.frame_count = 0
        self.total_inference_time = 0
        
        # Create pads
        self.sinkpad = Gst.Pad.new_from_template(self._sink_template, "sink")
        self.sinkpad.set_chain_function(self._chain)
        self.sinkpad.set_event_function(self._sink_event)
        self.add_pad(self.sinkpad)
        
        self.srcpad = Gst.Pad.new_from_template(self._src_template, "src")
        self.add_pad(self.srcpad)
        
        print("[YOLOv8s] Pads created", flush=True)

    def _initialize_device(self):
        """Initialize TT device and load model.
        
        Uses single device for real-time streaming (batch_size=1).
        Set USE_MESH_DEVICE=1 to use all devices (for batched processing).
        """
        device_ids = ttnn.get_device_ids()
        num_devices = len(device_ids)
        use_mesh = os.environ.get("USE_MESH_DEVICE", "0") == "1"
        
        print(f"[YOLOv8s] Found {num_devices} TT devices: {device_ids}", flush=True)
        
        device_params = {
            "l1_small_size": YOLOV8S_L1_SMALL_SIZE,
            "trace_region_size": 6434816,
            "num_command_queues": 2,
        }
        
        if use_mesh and num_devices > 1:
            # Multi-device mesh mode (for batched processing)
            mesh_shape = ttnn.MeshShape(1, num_devices)
            print(f"[YOLOv8s] Creating mesh device with shape {mesh_shape}...", flush=True)
            self.device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
            self.device.enable_program_cache()
            
            actual_devices = self.device.get_num_devices()
            inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(self.device)
            self.mesh_composer = output_mesh_composer
            
            print(f"[YOLOv8s] Mesh device created with {actual_devices} devices", flush=True)
            batch_size = actual_devices
        else:
            # Single device mode (for real-time streaming)
            print(f"[YOLOv8s] Using single device mode (device 0)...", flush=True)
            self.device = ttnn.CreateDevice(
                self.device_id,
                l1_small_size=device_params["l1_small_size"],
                trace_region_size=device_params["trace_region_size"],
                num_command_queues=device_params["num_command_queues"],
            )
            self.device.enable_program_cache()
            inputs_mesh_mapper = None
            self.mesh_composer = None
            batch_size = 1
            print(f"[YOLOv8s] Single device created", flush=True)
        
        print("[YOLOv8s] Loading model (trace compile ~2 min)...", flush=True)
        self.model = YOLOv8sPerformantRunner(
            self.device,
            device_batch_size=batch_size,
            mesh_mapper=inputs_mesh_mapper,
            mesh_composer=self.mesh_composer,
        )
        
        self.names = load_coco_class_names()
        print(f"[YOLOv8s] Model loaded! Batch size: {batch_size}", flush=True)

    def _sink_event(self, pad, parent, event):
        """Handle sink pad events."""
        if event.type == Gst.EventType.CAPS:
            caps = event.parse_caps()
            print(f"[YOLOv8s] Got caps: {caps.to_string()}", flush=True)
            return self.srcpad.push_event(event)
        return self.srcpad.push_event(event)

    def _chain(self, pad, parent, buf):
        """Process each frame - this is called for every buffer."""
        print(f"[YOLOv8s] _chain called, frame {self.frame_count}", flush=True)
        
        # Lazy init on first frame
        if self.model is None:
            self._initialize_device()

        try:
            # 1. Read input frame
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                print("[YOLOv8s] Failed to map buffer", flush=True)
                return Gst.FlowReturn.ERROR

            frame_bgrx = np.frombuffer(map_info.data, dtype=np.uint8).reshape(640, 640, 4)
            frame_bgr = frame_bgrx[:, :, :3].copy()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            buf.unmap(map_info)

            # 2. Preprocess
            tensor = torch.from_numpy(frame_rgb).float().div(255.0).unsqueeze(0)
            tensor = torch.permute(tensor, (0, 3, 1, 2))

            # 3. TT Inference (runs on all mesh devices)
            t_start = time.time()
            preds = self.model.run(tensor)
            preds = ttnn.to_torch(preds[0], dtype=torch.float32, mesh_composer=self.mesh_composer)
            t_end = time.time()

            inference_ms = (t_end - t_start) * 1000
            self.frame_count += 1
            self.total_inference_time += inference_ms

            # 4. Postprocess + draw
            results = obj_postprocess(preds, tensor, [frame_bgr], [["1"]], self.names)[0]
            out_image = self._draw_detections(results, frame_bgr)

            # Print stats every frame for now
            avg_ms = self.total_inference_time / self.frame_count
            fps = 1000 / avg_ms if avg_ms > 0 else 0
            print(f"[YOLOv8s] Frame {self.frame_count}: {inference_ms:.1f}ms (avg: {avg_ms:.1f}ms, ~{fps:.0f} FPS)", flush=True)

            # 5. Create output buffer
            out_buf = Gst.Buffer.new_allocate(None, out_image.nbytes, None)
            out_buf.fill(0, out_image.tobytes())
            out_buf.pts = buf.pts
            out_buf.dts = buf.dts
            out_buf.duration = buf.duration

            return self.srcpad.push(out_buf)

        except Exception as e:
            print(f"[YOLOv8s] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return Gst.FlowReturn.ERROR

    def _draw_detections(self, result, image):
        """Draw bounding boxes on image with confidence filtering."""
        boxes = result["boxes"]["xyxy"]
        scores = result["boxes"]["conf"]
        classes = result["boxes"]["cls"]
        
        # Higher confidence threshold to reduce false positives
        CONF_THRESHOLD = 0.5
        
        for box, score, cls in zip(boxes, scores, classes):
            conf = score.item() if hasattr(score, 'item') else float(score)
            
            # Skip low confidence detections
            if conf < CONF_THRESHOLD:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            class_name = self.names[class_id] if class_id < len(self.names) else f"class_{class_id}"
            label = f"{class_name} {conf:.2f}"
            
            # Color by class (green for person, blue for vehicle, red for others)
            if class_name == "person":
                color = (0, 255, 0)  # Green
            elif class_name in ["car", "truck", "bus", "motorcycle", "bicycle"]:
                color = (255, 100, 0)  # Blue-ish
            else:
                color = (0, 0, 255)  # Red for other classes
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    def do_get_property(self, prop):
        if prop.name == "device-id":
            return self.device_id
        raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop, value):
        if prop.name == "device-id":
            self.device_id = value
        else:
            raise AttributeError(f"Unknown property {prop.name}")


# Register plugin
GObject.type_register(Yolov8s)
__gstelementfactory__ = ("yolov8s", Gst.Rank.NONE, Yolov8s)
