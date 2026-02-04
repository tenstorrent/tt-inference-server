#!/usr/bin/env python3
"""
GStreamer Pipeline Element Timing Benchmark

Measures timing for EVERY element in the face recognition pipeline:
- filesrc (read file)
- decodebin (video decoder)  
- videoconvert (color conversion)
- videoscale (resize)
- face_recognition (YuNet + SFace on TTNN)
- videoconvert (output color conversion)
- jpegenc (JPEG encoder)

Uses GStreamer pad probes to measure actual element latencies.
"""

import sys
import time
import os
from collections import defaultdict

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)


class ElementTimer:
    """Track timing for GStreamer elements using pad probes."""
    
    def __init__(self):
        self.element_times = defaultdict(list)  # element_name -> list of (enter_time, exit_time)
        self.enter_times = {}  # buffer_ptr -> {element: time}
        self.frame_count = 0
        self.start_time = None
        
    def on_pad_probe(self, pad, info, element_name, is_src):
        """Pad probe callback to measure element timing."""
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
            
        buf_ptr = buf.pts  # Use PTS as buffer identifier
        
        if not is_src:  # Sink pad = entering element
            if buf_ptr not in self.enter_times:
                self.enter_times[buf_ptr] = {}
            self.enter_times[buf_ptr][element_name] = time.time()
        else:  # Src pad = exiting element
            if buf_ptr in self.enter_times and element_name in self.enter_times[buf_ptr]:
                enter_time = self.enter_times[buf_ptr][element_name]
                elapsed_ms = (time.time() - enter_time) * 1000
                self.element_times[element_name].append(elapsed_ms)
                
        return Gst.PadProbeReturn.OK


def run_benchmark(video_path):
    """Run the GStreamer pipeline with timing probes on each element."""
    
    print("\n" + "="*70)
    print("  GSTREAMER PIPELINE - COMPLETE ELEMENT BREAKDOWN")
    print("="*70)
    print(f"\nVideo: {video_path}")
    print("Setting up pipeline with timing probes on each element...\n")
    
    timer = ElementTimer()
    
    # Build pipeline string
    pipeline_str = f"""
        filesrc name=src location={video_path} !
        decodebin name=decoder !
        videoconvert name=convert1 !
        videoscale name=scale1 !
        videorate name=rate !
        video/x-raw,width=640,height=640,framerate=30/1 !
        queue name=queue1 !
        videoconvert name=convert2 !
        videoscale name=scale2 !
        video/x-raw,format=BGRx,width=640,height=640 !
        face_recognition name=face_rec !
        videoconvert name=convert3 !
        jpegenc name=encoder quality=85 !
        fakesink name=sink sync=false
    """
    
    pipeline = Gst.parse_launch(pipeline_str)
    
    # Elements to measure (in order)
    elements_to_measure = [
        ("decoder", "Video Decoder"),
        ("convert1", "videoconvert (input)"),
        ("scale1", "videoscale (to 640x640)"),
        ("rate", "videorate"),
        ("convert2", "videoconvert (to BGRx)"),
        ("scale2", "videoscale (pre-AI)"),
        ("face_rec", "face_recognition (TTNN)"),
        ("convert3", "videoconvert (output)"),
        ("encoder", "jpegenc (JPEG)"),
    ]
    
    # Add pad probes to each element
    for elem_name, desc in elements_to_measure:
        element = pipeline.get_by_name(elem_name)
        if element:
            # Get sink and src pads
            sink_pad = element.get_static_pad("sink")
            src_pad = element.get_static_pad("src")
            
            if sink_pad:
                sink_pad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    lambda pad, info, name=elem_name: timer.on_pad_probe(pad, info, name, False),
                )
            if src_pad:
                src_pad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    lambda pad, info, name=elem_name: timer.on_pad_probe(pad, info, name, True),
                )
            print(f"  ✓ Attached probes to: {elem_name} ({desc})")
        else:
            print(f"  ✗ Element not found: {elem_name}")
    
    print("\nStarting pipeline (warmup may take ~30s)...\n")
    
    # Run pipeline
    pipeline.set_state(Gst.State.PLAYING)
    
    # Wait for EOS or error
    bus = pipeline.get_bus()
    timer.start_time = time.time()
    
    while True:
        msg = bus.timed_pop_filtered(
            Gst.CLOCK_TIME_NONE,
            Gst.MessageType.EOS | Gst.MessageType.ERROR
        )
        
        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print(f"Error: {err.message}")
                break
            elif msg.type == Gst.MessageType.EOS:
                print("Pipeline finished (EOS)")
                break
    
    total_time = time.time() - timer.start_time
    pipeline.set_state(Gst.State.NULL)
    
    # Print results
    print("\n" + "="*70)
    print("  RESULTS: PER-ELEMENT TIMING BREAKDOWN")
    print("="*70)
    
    total_element_time = 0
    results = []
    
    for elem_name, desc in elements_to_measure:
        times = timer.element_times.get(elem_name, [])
        if times:
            # Skip first few (warmup)
            if len(times) > 5:
                times = times[5:]
            
            import numpy as np
            mean_ms = np.mean(times)
            p95_ms = np.percentile(times, 95) if len(times) > 1 else mean_ms
            total_element_time += mean_ms
            results.append((elem_name, desc, mean_ms, p95_ms, len(times)))
    
    print(f"\n{'Element':<30} {'Mean(ms)':<12} {'P95(ms)':<12} {'Frames':<10}")
    print("-" * 70)
    
    for elem_name, desc, mean_ms, p95_ms, count in results:
        pct = (mean_ms / total_element_time * 100) if total_element_time > 0 else 0
        marker = " ⬅ AI" if "face_rec" in elem_name else ""
        marker = " ⬅ ENCODER" if "encoder" in elem_name else marker
        marker = " ⬅ DECODER" if "decoder" in elem_name else marker
        print(f"{desc:<30} {mean_ms:>8.2f} ms  {p95_ms:>8.2f} ms  {count:>6}{marker}")
    
    print("-" * 70)
    print(f"{'TOTAL (all elements)':<30} {total_element_time:>8.2f} ms")
    
    # Summary for manager
    decoder_time = next((r[2] for r in results if "decoder" in r[0].lower()), 0)
    encoder_time = next((r[2] for r in results if "encoder" in r[0].lower()), 0)
    ai_time = next((r[2] for r in results if "face_rec" in r[0].lower()), 0)
    convert_time = sum(r[2] for r in results if "convert" in r[0].lower())
    scale_time = sum(r[2] for r in results if "scale" in r[0].lower())
    
    print("\n" + "="*70)
    print("  SUMMARY FOR MANAGER")
    print("="*70)
    print(f"""
┌────────────────────────────────────────────────────────────────────┐
│                    PIPELINE LATENCY BREAKDOWN                      │
├────────────────────────────────────────────────────────────────────┤
│  VIDEO DECODER (decodebin)    {decoder_time:>6.2f} ms  ({decoder_time/total_element_time*100 if total_element_time else 0:>5.1f}%)  {"✅ NOT bottleneck" if decoder_time < 10 else "⚠️  Check"}
│  VIDEO ENCODER (jpegenc)      {encoder_time:>6.2f} ms  ({encoder_time/total_element_time*100 if total_element_time else 0:>5.1f}%)  {"✅ NOT bottleneck" if encoder_time < 10 else "⚠️  Check"}
│  COLOR CONVERT (videoconvert) {convert_time:>6.2f} ms  ({convert_time/total_element_time*100 if total_element_time else 0:>5.1f}%)  {"✅ NOT bottleneck" if convert_time < 10 else "⚠️  Check"}
│  RESIZE (videoscale)          {scale_time:>6.2f} ms  ({scale_time/total_element_time*100 if total_element_time else 0:>5.1f}%)  {"✅ NOT bottleneck" if scale_time < 10 else "⚠️  Check"}
│  AI INFERENCE (TTNN)         {ai_time:>6.2f} ms  ({ai_time/total_element_time*100 if total_element_time else 0:>5.1f}%)  {"⚠️  MAIN BOTTLENECK" if ai_time > 30 else "✅ OK"}
├────────────────────────────────────────────────────────────────────┤
│  TOTAL PER FRAME             {total_element_time:>6.2f} ms  (100%)   FPS: {1000/total_element_time if total_element_time else 0:>5.1f}
│  Requirement: <500 ms        {"✅ PASS" if total_element_time < 500 else "❌ FAIL"}
└────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "/app/demo/test_video.mp4"
    run_benchmark(video)
