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
import cv2
import base64
import requests
import collections
import numpy as np
import ctypes

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib

RESTART_REQUESTED = False
TRACKER_HISTORY = {} # track_id -> state 

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("WatchstanderDS")

RTSP_URL = os.environ.get("RTSP_URL", "rtsp://192.168.1.100:8080/video")
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
VISION_MODE = os.environ.get("VISION_MODE", "marine").lower()
COSMOS_VLLM_URL = os.environ.get("COSMOS_VLLM_URL", "http://cosmos-vision:8010/v1/chat/completions")

FRAME_BUFFER = collections.deque(maxlen=150)
LAST_SENTINEL_TRIGGER = 0
SENTINEL_COOLDOWN = int(os.environ.get("SENTINEL_COOLDOWN", 10))
MAX_CACHED_VIDEOS = int(os.environ.get("MAX_CACHED_VIDEOS", 50))
MANUAL_SENTINEL_TRIGGER = False
MANUAL_PROMPT_OVERRIDE = {}

def process_sentinel_video_delayed(fps, event_class, prompt_overrides=None, delay=2.5):
    import time
    # Give the object time to pass through the frame before copying the buffer
    if delay > 0:
        time.sleep(delay) 
    # Create a thread-safe snapshot of the buffer
    buffer_snapshot = list(FRAME_BUFFER)
    process_sentinel_video(buffer_snapshot, fps, event_class, prompt_overrides)

def process_sentinel_video(buffer_snapshot, fps=20, event_class="scene_summary", prompt_overrides=None):
    if not buffer_snapshot or len(buffer_snapshot) < 10:
        return
        
    logger.info(f"Sentinel triggered ({event_class})! Encoding buffer to clip for Cosmos analysis...")
    try:
        h, w = buffer_snapshot[0].shape[:2]
        tmp_file = "/tmp/sentinel_clip.mp4"
        debug_file = "/app/data/videos/sentinel_debug_clip.mp4" 
        import shutil
        import os
        
        # Use XVID (mp4v) - much more stable for vLLM than MJPG
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(tmp_file, fourcc, fps, (w, h))
        
        if not out.isOpened():
            logger.warning("mp4v codec failed, falling back to MJPG")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(tmp_file, fourcc, fps, (w, h))
        
        for f in buffer_snapshot:
            out.write(f)
        out.release()

        # FIFO Cache for rolling clips
        import glob
        video_dir = "/app/data/videos"
        event_ts = int(time.time())
        historical_file = f"{video_dir}/event_{event_ts}.mp4"

        # Re-encode to H.264 for web compatibility using GStreamer CLI
        web_tmp = "/tmp/sentinel_web.mp4"
        import subprocess
        # Pipeline: Decode -> Convert -> NV Hardware H.264 Encode -> Parse -> MP4 Mux -> File
        # We use a faster hardware path
        gst_cmd = f"gst-launch-1.0 filesrc location={tmp_file} ! decodebin ! videoconvert ! nvv4l2h264enc ! h264parse ! mp4mux ! filesink location={web_tmp}"
        
        try:
            logger.info("Re-encoding clip for web compatibility...")
            subprocess.run(gst_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            shutil.copy(web_tmp, debug_file)
            shutil.copy(web_tmp, historical_file)
            logger.info(f"Successfully encoded web-friendly clip to {historical_file}")
            # Use the high-quality H264 clip for Cosmos too
            cosmos_input_file = web_tmp
        except Exception as e:
            logger.error(f"Failed to re-encode video for web: {e}. Falling back to original.")
            shutil.copy(tmp_file, debug_file)
            shutil.copy(tmp_file, historical_file)
            cosmos_input_file = tmp_file

        # Enforce FIFO limit
        if MAX_CACHED_VIDEOS > 0:
            cached_files = sorted(glob.glob(f"{video_dir}/event_*.mp4"), key=os.path.getmtime)
            while len(cached_files) > MAX_CACHED_VIDEOS:
                oldest_file = cached_files.pop(0)
                try:
                    os.remove(oldest_file)
                    logger.debug(f"Removed old cached clip to maintain FIFO limit: {oldest_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove {oldest_file}: {e}")
        
        with open(cosmos_input_file, "rb") as vf:
            encoded_string = base64.b64encode(vf.read()).decode('utf-8')
            
        data_uri = f"data:video/mp4;base64,{encoded_string}"
        
        logger.info("Submitting clip to Cosmos API...")

        # Import prompts (usually they reside in the same repo structure)
        # Note: In a containerized environment, we might need a relative import or a shared lib
        try:
             # Try to load from centralized prompts if available in the PYTHONPATH
             from orchestrator.prompts import WATCHSTANDER_SYSTEM_PROMPT, WATCHSTANDER_USER_PROMPT_TEMPLATE
             system_prompt = WATCHSTANDER_SYSTEM_PROMPT
             
             prev_state = "No previous context provided."
             user_query = "None."
             
             if prompt_overrides:
                 prev_state = prompt_overrides.get("previous_state", prev_state)
                 user_query = prompt_overrides.get("user_query", user_query)
             
             user_prompt = WATCHSTANDER_USER_PROMPT_TEMPLATE.format(
                 previous_state=prev_state,
                 user_query=user_query
             )
        except ImportError:
            # Fallback for standalone container deployment
            system_prompt = "You are Sentinel, an autonomous AI watchstander. Your duty is to continuously monitor video feeds, detect anomalies, track moving objects (especially people and equipment), and provide clear, structured situation reports. CRITICAL: You do NOT have a heading sensor or compass. DO NOT hallucinate bearings, headings, or coordinates. Instead, describe positions relative to the camera frame using standard terms: 'Left', 'Right', or 'Center'. YOU MUST RESPOND STRICTLY IN ENGLISH."
            user_prompt = "Observe this short video clip. Provide a professional, objective situation report. If no specific user request is present, provide a concise (2-3 sentence) summary of the current scene state, including key objects and their positions (Left/Right/Center). Focus on movement if present. DO NOT OUTPUT CHINESE CHARACTERS."
        
        if redis_client:
            stored_sys = redis_client.get("prompt:cosmos:system")
            stored_user = redis_client.get("prompt:cosmos:user")
            if stored_sys: system_prompt = stored_sys
            if stored_user: user_prompt = stored_user

        payload = {
            "model": os.environ.get("COSMOS_MODEL_NAME", "nvidia/cosmos-reason2-2b-fp8"),
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "video_url", "video_url": {"url": data_uri}}
                    ]
                }
            ],
            "temperature": 0.2,
            "repetition_penalty": 1.15,
            "presence_penalty": 0.1,
            "max_tokens": 512
        }
        
        resp = requests.post(COSMOS_VLLM_URL, json=payload, timeout=60)
        resp.raise_for_status()
        response_data = resp.json()
        result_text = response_data['choices'][0]['message']['content']
        logger.info(f"Cosmos Response: {result_text}")
        
        _, buffer = cv2.imencode('.jpg', buffer_snapshot[-1])
        thumb_base64 = base64.b64encode(buffer).decode('utf-8')
        
        if redis_client:
            event_data = {
                "class": event_class,
                "content": result_text,
                "image_base64": thumb_base64,
                "video_path": historical_file,
                "timestamp": time.time()
            }
            redis_client.xadd("vision_events", event_data, maxlen=10000)
            logger.info(f"Successfully published {event_class} to vision_events stream.")
            
    except Exception as e:
        logger.error(f"Failed to query Cosmos API: {e}")

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None

def redis_control_listener(loop):
    global RESTART_REQUESTED
    global MANUAL_SENTINEL_TRIGGER
    global MANUAL_PROMPT_OVERRIDE
    if not redis_client:
        return
    pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe("vision_control")
    logger.info("Listening for control messages on 'vision_control'...")
    
    while True:
        try:
            message = pubsub.get_message(timeout=1.0)
            if message and message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    if data.get("command") == "analyze_scene":
                        MANUAL_SENTINEL_TRIGGER = True
                        MANUAL_PROMPT_OVERRIDE = {
                            "previous_state": data.get("previous_state"),
                            "user_query": data.get("user_query")
                        }
                        logger.info(f"Manual scene analysis requested via Redis: {MANUAL_PROMPT_OVERRIDE['user_query']}")
                except Exception as e:
                    logger.error(f"Error parsing Redis message: {e}")
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Redis listener error: {e}")
            time.sleep(5)

def bus_call(bus, message, loop, pipeline):
    t = message.type
    if t == Gst.MessageType.EOS:
        pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0)
    elif t==Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error("Pipeline Error: %s: %s" % (err, debug))
        global RESTART_REQUESTED
        RESTART_REQUESTED = True
        time.sleep(5)
        loop.quit()
    return True

def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    try:
        import pyds
    except ImportError:
        return Gst.PadProbeReturn.OK
        
    try:
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        active_contact_detected = False
        events_to_publish = []
        current_time = time.time()
        
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
                
            frame_width = frame_meta.source_frame_width
            frame_height = frame_meta.source_frame_height
            
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                
                class_id = obj_meta.class_id
                confidence = obj_meta.confidence
                label = obj_meta.obj_label if obj_meta.obj_label else {0: "person", 8: "boat"}.get(class_id, str(class_id))
                track_id = obj_meta.object_id if obj_meta.object_id != -1 else id(obj_meta)

                rect_params = obj_meta.rect_params
                x_center = rect_params.left + (rect_params.width / 2)
                y_bottom = rect_params.top + rect_params.height
                
                norm_x = (x_center / frame_width) - 0.5
                
                # Use qualitative positions instead of hallucinated compass bearings
                if norm_x < -0.15:
                    bearing_desc = "port"
                elif norm_x > 0.15:
                    bearing_desc = "starboard"
                else:
                    bearing_desc = "dead ahead"
                
                norm_y = y_bottom / frame_height
                dist_desc = "distant" if norm_y < 0.4 else "close" if norm_y > 0.7 else "mid-range"

                is_stationary = False
                
                if track_id not in TRACKER_HISTORY:
                    TRACKER_HISTORY[track_id] = {'x': x_center, 'y': y_bottom, 'last_publish': 0, 'stationary_score': 0}
                else:
                    hist = TRACKER_HISTORY[track_id]
                    dx = abs(hist['x'] - x_center)
                    dy = abs(hist['y'] - y_bottom)
                    if dx < 10 and dy < 10: 
                        hist['stationary_score'] += 1
                    else:
                        hist['stationary_score'] = max(0, hist['stationary_score'] - 2) 
                        hist['x'] = x_center
                        hist['y'] = y_bottom
                        
                    if hist['stationary_score'] > 30:
                        is_stationary = True

                should_publish = False
                if label in ["person", "boat", "ship", "vessel", "car", "truck", "vehicle"] and confidence > 0.5:
                    if not is_stationary:
                        hist = TRACKER_HISTORY[track_id]
                        active_contact_detected = True
                        if current_time - hist['last_publish'] > 2.0:
                            logger.info(f"Target seen: {label} ID:{track_id} stat:{is_stationary} score:{hist['stationary_score']}")
                            should_publish = True
                            hist['last_publish'] = current_time
                
                if should_publish:
                    event_data = {
                        "class": label,
                        "position": bearing_desc,
                        "distance": dist_desc,
                        "heading_rel": 0.0,
                        "nav_status": "in_range" if dist_desc == "close" else "monitor",
                        "timestamp": current_time,
                        "track_id": track_id,
                        "confidence": confidence
                    }
                    events_to_publish.append(event_data)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
                    
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        if redis_client:
            for ev in events_to_publish:
                try:
                    redis_client.xadd("vision_events", ev, maxlen=10000)
                except Exception:
                    pass

        global MANUAL_SENTINEL_TRIGGER
        global LAST_SENTINEL_TRIGGER
        global MANUAL_PROMPT_OVERRIDE
        if active_contact_detected or MANUAL_SENTINEL_TRIGGER:
            is_cooldown_ok = (current_time - LAST_SENTINEL_TRIGGER > SENTINEL_COOLDOWN)
            if is_cooldown_ok or MANUAL_SENTINEL_TRIGGER:
                if len(FRAME_BUFFER) >= 10:
                    LAST_SENTINEL_TRIGGER = current_time
                    ev_class = "status_request" if MANUAL_SENTINEL_TRIGGER else "contact_report"
                    
                    # Pass overrides if manual trigger
                    overrides = MANUAL_PROMPT_OVERRIDE if MANUAL_SENTINEL_TRIGGER else None
                    
                    delay_val = 0.0 if MANUAL_SENTINEL_TRIGGER else 2.5
                    t = threading.Thread(target=process_sentinel_video_delayed, args=(30, ev_class, overrides, delay_val))
                    t.daemon = True
                    t.start()
                    
                    if MANUAL_SENTINEL_TRIGGER:
                        logger.info("Consumed manual sentinel trigger.")
                        MANUAL_SENTINEL_TRIGGER = False
                        MANUAL_PROMPT_OVERRIDE = {}

    except Exception as e:
        logger.error(f"Error in deepstream loop: {e}")
        
    return Gst.PadProbeReturn.OK

def on_new_sample(sink, data):
    sample = sink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.OK
        
    gst_buffer = sample.get_buffer()
    if not gst_buffer:
        return Gst.FlowReturn.OK
        
    success, map_info = gst_buffer.map(Gst.MapFlags.READ)
    if success:
        try:
            h, w = 1080, 1920
            size = map_info.size
            if size == h * w * 4:
                frame_array = np.ndarray(shape=(h, w, 4), dtype=np.uint8, buffer=map_info.data)
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGR)
                frame_small = cv2.resize(frame_bgr, (1280, 720))
                FRAME_BUFFER.append(frame_small)
            elif size > h * w * 4:
                pitch = size // h
                frame_array = np.ndarray(shape=(h, pitch), dtype=np.uint8, buffer=map_info.data)
                frame_cropped = frame_array[:, :w*4]
                frame_rgb = frame_cropped.reshape((h, w, 4))
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2BGR)
                frame_small = cv2.resize(frame_bgr, (1280, 720))
                FRAME_BUFFER.append(frame_small)
        except Exception as e:
            pass
        finally:
            gst_buffer.unmap(map_info)

    return Gst.FlowReturn.OK

def main():
    GObject.threads_init()
    Gst.init(None)

    logger.info("Initializing Pipeline...")

    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
        return

    is_v4l2 = RTSP_URL.startswith("v4l2://") or RTSP_URL.startswith("/dev/video")
    if is_v4l2:
        device = RTSP_URL.replace("v4l2://", "")
        source = Gst.ElementFactory.make("v4l2src", "v4l2-source")
        source.set_property("device", device)
        v4l2_caps_filter = Gst.ElementFactory.make("capsfilter", "v4l2_caps")
        v4l2_caps_filter.set_property("caps", Gst.Caps.from_string("video/x-raw, width=1920, height=1080, framerate=30/1"))
        vidconv = Gst.ElementFactory.make("videoconvert", "v4l2_vidconv")
    else:
        source = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
        source.set_property("uri", RTSP_URL)
        
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    
    nvvidconv_in = Gst.ElementFactory.make("nvvideoconvert", "in_convertor")
    
    caps_filter = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
    caps_filter.set_property("caps", caps)
    
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', "config_infer_primary.txt")

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 384)
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    
    nvvidconv_preosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_preosd")
    caps_filter_preosd = Gst.ElementFactory.make("capsfilter", "filter_preosd")
    caps_preosd = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    caps_filter_preosd.set_property("caps", caps_preosd)

    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    caps_filter_postosd = Gst.ElementFactory.make("capsfilter", "filter_postosd")
    caps_postosd = Gst.Caps.from_string("video/x-raw, format=RGBA")
    caps_filter_postosd.set_property("caps", caps_postosd)
    
    sink = Gst.ElementFactory.make("appsink", "sink")
    sink.set_property("drop", True)
    sink.set_property("max-buffers", 2)
    sink.set_property("emit-signals", True)
    sink.connect("new-sample", on_new_sample, None)

    if not all([source, nvvidconv_in, caps_filter, streammux, pgie, tracker, nvvidconv_preosd, caps_filter_preosd, nvosd, nvvidconv_postosd, caps_filter_postosd, sink]):
        sys.stderr.write(" Unable to create elements \n")
        return

    pipeline.add(source)
    if is_v4l2:
        pipeline.add(v4l2_caps_filter)
        pipeline.add(vidconv)
    pipeline.add(nvvidconv_in)
    pipeline.add(caps_filter)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(nvvidconv_preosd)
    pipeline.add(caps_filter_preosd)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps_filter_postosd)
    pipeline.add(sink)

    if is_v4l2:
        source.link(v4l2_caps_filter)
        v4l2_caps_filter.link(vidconv)
        vidconv.link(nvvidconv_in)
    else:
        def decodebin_pad_added(decodebin, pad):
            caps = pad.query_caps(None)
            if not caps: return
            name = caps.get_structure(0).get_name()
            if name.find("video") != -1:
                sink_pad = nvvidconv_in.get_static_pad("sink")
                if sink_pad and not sink_pad.is_linked():
                    link_ret = pad.link(sink_pad)
                    if link_ret == Gst.PadLinkReturn.OK:
                        logger.info("Successfully linked video pad")

        source.connect("pad-added", decodebin_pad_added)
        
    nvvidconv_in.link(caps_filter)
    caps_src_pad = caps_filter.get_static_pad("src")
    muxer_sink_pad = streammux.get_request_pad("sink_0")
    caps_src_pad.link(muxer_sink_pad)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv_preosd)
    nvvidconv_preosd.link(caps_filter_preosd)
    caps_filter_preosd.link(nvosd)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps_filter_postosd)
    caps_filter_postosd.link(sink)

    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")
    else:
        # PROBE ON NVMM BOUNDARY
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop, pipeline)

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
        os.execv(sys.executable, ['python3'] + sys.argv)
