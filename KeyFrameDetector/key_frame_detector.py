import os
import numpy as np
import av
import json
from scipy.interpolate import griddata
from KeyFrameDetector.utils import convert_frame_to_grayscale, save_keyframe
from collections import deque

def analyze_video_motion_vectors(video_path, sample_duration=10):
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            start_time = float(stream.start_time or 0)
            
            # Handle cases where duration is not available
            if stream.duration is not None:
                duration = float(stream.duration * stream.time_base)
            else:
                # If duration is not available, we'll analyze the first 'sample_duration' seconds
                duration = sample_duration
            
            container.seek(int(start_time * stream.time_base))
            
            frames_with_mv = 0
            total_frames = 0
            
            for frame in container.decode(video=0):
                if frame.time > start_time + sample_duration:
                    break
                
                total_frames += 1
                if hasattr(frame, 'motion_vectors') and frame.motion_vectors:
                    frames_with_mv += 1
            
            has_motion_vectors = frames_with_mv > 0
            percentage_with_mv = (frames_with_mv / total_frames) * 100 if total_frames > 0 else 0
            
            return has_motion_vectors, percentage_with_mv
    except Exception as e:
        print(f"Error analyzing video motion vectors: {str(e)}")
        return False, 0

def enhance_motion_vectors(mv_array, frame_shape):
    y, x = np.mgrid[0:frame_shape[0], 0:frame_shape[1]]
    src_points = mv_array[:, 0]
    dst_x = griddata(src_points, mv_array[:, 1], (y, x), method='linear', fill_value=0)
    dst_y = griddata(src_points, mv_array[:, 2], (y, x), method='linear', fill_value=0)
    return np.stack([dst_x, dst_y], axis=-1)

def compute_frame_difference(prev_frame, curr_frame):
    return np.mean(np.abs(curr_frame.astype(float) - prev_frame.astype(float)))

def lightweight_optical_flow(video_path, use_motion_vectors=True):
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        previous_frame = None
        previous_flow = None
        
        for frame in container.decode(video=0):
            current_frame = frame.to_ndarray(format='gray')
            
            if use_motion_vectors and hasattr(frame, 'motion_vectors') and frame.motion_vectors:
                mv_array = np.array([(mv.source, mv.dst_x, mv.dst_y) for mv in frame.motion_vectors])
                flow = enhance_motion_vectors(mv_array, current_frame.shape)
                
                if previous_flow is not None:
                    flow = 0.5 * flow + 0.5 * previous_flow
                
                previous_flow = flow
                yield frame.time, np.sqrt(np.sum(flow**2, axis=-1)).mean()
            elif previous_frame is not None:
                diff = compute_frame_difference(previous_frame, current_frame)
                yield frame.time, diff
            else:
                yield frame.time, 0
            
            previous_frame = current_frame

def keyframe_detection(video_path, dest, threshold, chunk_size=1000, start_time=None, end_time=None,
                       min_scene_length=1, max_motion_factor=1.0, content_weight=1.0, use_original_algorithm=False,
                       min_time_constraint=None, max_time_constraint=None, look_ahead_time=1.0,
                       verbose=False, text_mode=False, save_images=True, json_filename=None, debug_mode=False):
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            fps = stream.average_rate
            frame_count = 0
            keyframe_data = []
            magnitude_buffer = deque(maxlen=int(look_ahead_time * fps))
            last_keyframe_time = -float('inf')

            keyframePath = os.path.join(dest, 'keyFrames')
            if save_images:
                os.makedirs(keyframePath, exist_ok=True)

            # Make frame 0 a keyframe
            for frame in container.decode(video=0):
                if save_images:
                    save_keyframe(frame.to_ndarray(format='rgb24'), keyframePath, frame_count)
                keyframe_info = {
                    "time": 0.00,
                    "frame": frame_count,
                    "reason": "first frame"
                }
                if debug_mode:
                    keyframe_info.update({
                        "magnitude": 0,
                        "adjusted_threshold": threshold,
                        "time_since_last_keyframe": 0
                    })
                keyframe_data.append(keyframe_info)
                if verbose:
                    print(f"Keyframe detected at 0.00 seconds (first frame)")
                last_keyframe_time = frame.time
                prev_frame = convert_frame_to_grayscale(frame.to_ndarray(format='rgb24'))
                break

            for frame in container.decode(video=0):
                frame_count += 1
                time = frame.time

                if start_time and time < start_time:
                    continue
                if end_time and time > end_time:
                    break

                curr_frame = convert_frame_to_grayscale(frame.to_ndarray(format='rgb24'))
                magnitude = np.mean(np.abs(curr_frame.astype(float) - prev_frame.astype(float)))
                magnitude_buffer.append((time, magnitude))

                time_since_last_keyframe = time - last_keyframe_time
                motion_penalty = min(1.0, time_since_last_keyframe / min_scene_length)
                adjusted_threshold = threshold * motion_penalty * max_motion_factor * content_weight

                is_keyframe = False
                keyframe_reason = ""

                # Check if current magnitude is a local maximum
                if len(magnitude_buffer) == magnitude_buffer.maxlen:
                    _, center_magnitude = magnitude_buffer[len(magnitude_buffer) // 2]
                    if center_magnitude == max(m for _, m in magnitude_buffer):
                        if center_magnitude > adjusted_threshold:
                            if min_time_constraint is None or time_since_last_keyframe >= min_time_constraint:
                                is_keyframe = True
                                keyframe_reason = "local maximum"

                # Force keyframe if max time constraint is reached
                if not is_keyframe and max_time_constraint is not None and time_since_last_keyframe >= max_time_constraint:
                    is_keyframe = True
                    keyframe_reason = "max time constraint"

                if is_keyframe:
                    if save_images:
                        save_keyframe(frame.to_ndarray(format='rgb24'), keyframePath, frame_count)

                    keyframe_info = {
                        "time": time,
                        "frame": frame_count,
                        "reason": keyframe_reason
                    }
                    if debug_mode:
                        keyframe_info.update({
                            "magnitude": magnitude,
                            "adjusted_threshold": adjusted_threshold,
                            "time_since_last_keyframe": time_since_last_keyframe
                        })
                    keyframe_data.append(keyframe_info)
                    last_keyframe_time = time
                    if verbose:
                        print(f"Keyframe detected at {time:.2f} seconds ({keyframe_reason})")

                prev_frame = curr_frame

                if verbose and frame_count % 100 == 0:
                    print(f"Processed frame {frame_count} at {time:.2f} seconds")

            if verbose:
                print(f"Total frames processed: {frame_count}")

            # Write keyframe metadata to JSON file if text_mode is True
            if text_mode:
                json_path = os.path.join(dest, json_filename)
                with open(json_path, 'w') as f:
                    json.dump(keyframe_data, f, indent=2)
                
                if verbose:
                    print(f"Keyframe metadata written to {json_path}")

    except Exception as e:
        print(f"Error during keyframe detection: {str(e)}")

def tune_keyframe_detection(sensitivity='balanced', content_type='general', min_scene_duration=1.0,
                            min_time_constraint=None, max_time_constraint=None, look_ahead_time=1.0):
    sensitivity_presets = {
        'low': {'threshold': 15, 'max_motion_factor': 1.5, 'content_weight': 0.8},
        'balanced': {'threshold': 10, 'max_motion_factor': 1.2, 'content_weight': 1.0},
        'high': {'threshold': 5, 'max_motion_factor': 1.0, 'content_weight': 1.2}
    }
    
    content_type_adjustments = {
        'action': {'threshold_mult': 0.8, 'min_scene_length_mult': 0.5},
        'documentary': {'threshold_mult': 1.2, 'min_scene_length_mult': 1.5},
        'general': {'threshold_mult': 1.0, 'min_scene_length_mult': 1.0}
    }
    
    base_params = sensitivity_presets[sensitivity]
    content_adj = content_type_adjustments[content_type]
    
    params = {
        'threshold': base_params['threshold'] * content_adj['threshold_mult'],
        'min_scene_length': min_scene_duration * content_adj['min_scene_length_mult'],
        'max_motion_factor': base_params['max_motion_factor'],
        'content_weight': base_params['content_weight'],
        'min_time_constraint': min_time_constraint,
        'max_time_constraint': max_time_constraint,
        'look_ahead_time': look_ahead_time
    }
    
    return params