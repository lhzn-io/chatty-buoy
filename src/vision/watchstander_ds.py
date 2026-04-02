#!/usr/bin/env python3

import sys
import gi
import configparser
import logging
import redis
import math
import os
import time
import threading
import json

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst, GLib, GstRtspServer

RESTART_REQUESTED = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("WatchstanderDS")

# Configuration
RTSP_URL = os.environ.get("RTSP_URL", "rtsp://192.168.1.100:8080/video")
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
VISION_MODE = os.environ.get("VISION_MODE", "marine").lower()

# Redis
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None

def redis_control_listener(loop):
    global RESTART_REQUESTED
    if not redis_client:
        return
    pubsub = redis_client.pubsub()
    pubsub.subscribe("vision_control")
    logger.info("Listening for control messages on 'vision_control'...")
    for message in pubsub.listen():
        if message["type"] == "message":
            try:
                data = json.loads(message["data"])
                if data.get("command") == "switch_source":
                    new_url = data.get("url")
                    if new_url:
                        logger.info(f"Switch Source command received. New URL: {new_url}")
                        os.environ["RTSP_URL"] = new_url
                        RESTART_REQUESTED = True
                        GLib.idle_add(loop.quit)
            except Exception as e:
                logger.error(f"Failed to parse control message: {e}")

def bus_call(bus, message, loop, pipeline):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream reached. Restarting video slice...\n")
        # Seek back to 0 to loop
        success = pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0)
        if not success:
            sys.stderr.write("Failed to seek back to 0. Quitting.\n")
            loop.quit()
    elif t==Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error("Pipeline Error: %s: %s" % (err, debug))
        logger.warning("Attempting automatic pipeline recovery in 5 seconds...")
        global RESTART_REQUESTED
        RESTART_REQUESTED = True
        time.sleep(5)
        loop.quit()
    return True

def pgie_src_pad_buffer_probe(pad, info, u_data):
    """
    Probe to extract metadata from the primary inference engine (PGIE).
    This is where we get detection results.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    # Retrieve batch metadata from the gst_buffer
    # Note: access pyds dynamically to avoid import errors in non-DS environments
    try:
        import pyds
    except ImportError:
        # Fallback/Mock for testing
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Frame resolution
        frame_width = frame_meta.source_frame_width
        frame_height = frame_meta.source_frame_height
        
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            # Extract Object Data
            class_id = obj_meta.class_id
            confidence = obj_meta.confidence
            
            # Create a simple mapping if label is not populated or we want to override
            # Assuming config uses standard COCO labels if pre-trained, 
            # or custom labels from the config file.
            # obj_meta.obj_label should contain the string label.
            label = obj_meta.obj_label

            # Detection Logic
            # Rect parameters: top, left, width, height
            # Note: DeepStream rect params are floats
            rect_params = obj_meta.rect_params
            top = rect_params.top
            left = rect_params.left
            width = rect_params.width
            height = rect_params.height
            
            x_center = left + (width / 2)
            y_bottom = top + height
            
            # Calculate Bearing/Range (Mock Logic from Python version)
            # Bearing: -45 to +45
            norm_x = (x_center / frame_width) - 0.5
            bearing = round(norm_x * 90, 1)
            
            # Range: Inverse pixel plane
            norm_y = y_bottom / frame_height
            if norm_y < 0.2:
                rng = 1000
            else:
                rng = round(10 / (norm_y - 0.2), 1)

            # Filter based on mode
            should_publish = False
            
            # Marine Mode: Boat (8), Ship, or Person (0)
            if label in ["person", "boat", "ship", "vessel"] and confidence > 0.5:
                should_publish = True
            
            if should_publish and redis_client:
                 event_data = {
                    "class": label,
                    "bearing": bearing,
                    "range": rng,
                    "heading_rel": 0.0, # No OBB in this version yet
                    "threat": "critical" if rng < 50 else "monitor",
                    "timestamp": time.time()
                }
                 try:
                    # Optional: Rate limit prints or send to redis
                    logger.info(f"SPOTTER HIT: {label} ({confidence:.2f})")
                    redis_client.xadd("vision_events", event_data, maxlen=100)
                 except Exception as e:
                    logger.error(f"Redis Publish Error: {e}")

            try: 
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break
            
    return Gst.PadProbeReturn.OK

def main():
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    logger.info("Initializing Pipeline...")

    # Create Pipeline
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
        return

    # Element Creation
    # Source -> Decode -> Mux -> Inference -> Sink
    
    # Source: uridecodebin (Handles RTSP, File, etc.)
    source = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    source.set_property("uri", RTSP_URL)
    
    # Stream Muxer (Required before nvinfer)
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property('width', 1920) # Mux resolution
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    
    # Input Converter (Buffer between source CPU mem and muxer NVMM mem)
    nvvidconv_in = Gst.ElementFactory.make("nvvideoconvert", "in_convertor")
    
    # Enforce NVMM memory output from the converter
    caps_filter = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
    caps_filter.set_property("caps", caps)
    
    # PGIE (Primary Inference)
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', "config_infer_primary.txt")

    # OSD (On-Screen Display to draw boxes)
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    
    # Post-OSD Converter
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    
    # Capsfilter for H264 Encoder (I420)
    caps_filter_enc = Gst.ElementFactory.make("capsfilter", "filter_enc")
    caps_enc = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
    caps_filter_enc.set_property("caps", caps_enc)
    
    # Hardware H264 Encoder
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    encoder.set_property('bitrate', 4000000)
    # Target preset-level 1 for max speed/low latency on Jetson
    encoder.set_property('preset-level', 1) 
    encoder.set_property('insert-sps-pps', 1)
    
    # RTP Payload
    rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
    rtppay.set_property("config-interval", 1) # Send SPS/PPS with every keyframe
    
    # RTSP Client Sink (Pushes to MediaMTX)
    sink = Gst.ElementFactory.make("rtspclientsink", "sink")
    sink.set_property("location", "rtsp://mediamtx:8554/labeled")
    sink.set_property("protocols", "tcp")
    sink.set_property("latency", 100)
    sink.set_property("async", True)

    if not all([source, nvvidconv_in, caps_filter, streammux, pgie, nvosd, nvvidconv_postosd, caps_filter_enc, encoder, rtppay, sink]):
        sys.stderr.write(" Unable to create elements \n")
        return

    # Add elements to pipeline
    pipeline.add(source)
    pipeline.add(nvvidconv_in)
    pipeline.add(caps_filter)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps_filter_enc)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)

    # Linking
    # uridecodebin has dynamic pads, so we link later via signal
    def decodebin_pad_added(decodebin, pad):
        caps = pad.query_caps(None)
        if not caps: 
            return
        
        name = caps.get_structure(0).get_name()
        logger.info(f"Source Pad Added: {name}")
        
        if name.find("video") != -1:
            sink_pad = nvvidconv_in.get_static_pad("sink")
            if not sink_pad:
                logger.error("Unable to get sink pad of nvvidconv_in")
                return
            
            link_ret = pad.link(sink_pad)
            if link_ret == Gst.PadLinkReturn.OK:
                logger.info("Successfully linked video pad to nvvidconv_in")
            else:
                logger.error(f"Failed to link video pad to nvvidconv_in: {link_ret}")
        else:
            logger.info(f"Ignoring non-video pad: {name}")

    source.connect("pad-added", decodebin_pad_added)
    
    # Static Linking
    # Link nvvidconv_in -> caps_filter -> streammux
    nvvidconv_in.link(caps_filter)
    
    caps_src_pad = caps_filter.get_static_pad("src")
    muxer_sink_pad = streammux.get_request_pad("sink_0")
    caps_src_pad.link(muxer_sink_pad)

    streammux.link(pgie)
    pgie.link(nvosd)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps_filter_enc)
    caps_filter_enc.link(encoder)
    encoder.link(rtppay)
    rtppay.link(sink)

    # Attach probe AFTER nvosd to ensure bounded boxes are drawn before publishing
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")
    else:
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)

    # Start Pipeline
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop, pipeline)

    logger.info("*** DeepStream: Pushing RTSP Stream to rtsp://mediamtx:8554/labeled ***")

    # Start Redis Control Thread
    control_thread = threading.Thread(target=redis_control_listener, args=(loop,), daemon=True)
    control_thread.start()

    logger.info("Starting Pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except BaseException:
        pass
    
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    main()
    
    if RESTART_REQUESTED:
        logger.info("Restart requested. Executing os.execv to reload the pipeline processes...")
        os.execv(sys.executable, ['python3'] + sys.argv)
