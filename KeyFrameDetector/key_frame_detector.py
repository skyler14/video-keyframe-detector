import os
import numpy as np
import time
import peakutils
import ffmpeg
from ffprobe import FFProbe
from KeyFrameDetector.utils import convert_frame_to_grayscale, save_keyframe

def create_frame_reader(video_path, chunk_size=1000, start_time=None, end_time=None):
    metadata = FFProbe(video_path)
    video_stream = next(s for s in metadata.streams if s.is_video())
    width = int(video_stream.width)
    height = int(video_stream.height)
    fps = eval(video_stream.r_frame_rate)
    
    input_args = {'ss': start_time} if start_time else {}
    output_args = {'t': end_time} if end_time else {}
    
    stream = ffmpeg.input(video_path, **input_args)
    
    if start_time:
        stream = stream.filter('setpts', 'PTS-STARTPTS')
    
    process = (
        stream
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', **output_args)
        .run_async(pipe_stdout=True)
    )
    
    frame_size = width * height * 3
    
    def read_frames():
        while True:
            in_bytes = process.stdout.read(frame_size * chunk_size)
            if not in_bytes:
                break
            yield np.frombuffer(in_bytes, np.uint8).reshape([-1, height, width, 3])
    
    return read_frames, process, fps

def original_keyframe_detection(video_path, dest, threshold, chunk_size=1000, start_time=None, end_time=None):
    frame_reader, process, fps = create_frame_reader(video_path, chunk_size, start_time, end_time)
    
    keyframePath = os.path.join(dest, 'keyFrames')
    os.makedirs(keyframePath, exist_ok=True)
    
    lstfrm = []
    lstdiffMag = []
    frames = []
    
    try:
        frame_count = 0
        last_frame_gray = None
        for chunk in frame_reader():
            frames.extend(chunk)
            for frame in chunk:
                gray_frame = convert_frame_to_grayscale(frame)
                
                if last_frame_gray is not None:
                    diff = np.mean(np.abs(gray_frame.astype(int) - last_frame_gray.astype(int)))
                    lstdiffMag.append(diff)
                    lstfrm.append(frame_count)
                
                last_frame_gray = gray_frame
                frame_count += 1
        
        y = np.array(lstdiffMag)
        base = peakutils.baseline(y, 2)
        indices = peakutils.indexes(y-base, threshold, min_dist=1)
        
        for idx in indices:
            save_keyframe(frames[idx], keyframePath, idx)
            print(f"Keyframe detected at {lstfrm[idx] / fps:.2f} seconds")
    
    finally:
        process.stdout.close()
        process.wait()

def keyframe_detection(video_path, dest, threshold, chunk_size=1000, start_time=None, end_time=None,
                       min_scene_length=1, max_motion_factor=1.0, content_weight=1.0, use_original_algorithm=False):
    if use_original_algorithm:
        return original_keyframe_detection(video_path, dest, threshold, chunk_size, start_time, end_time)
    frame_reader, process, fps = create_frame_reader(video_path, chunk_size, start_time, end_time)
    
    keyframePath = os.path.join(dest, 'keyFrames')
    os.makedirs(keyframePath, exist_ok=True)
    
    last_keyframe_time = -float('inf')
    last_frame_gray = None
    frame_diffs = []
    
    try:
        frame_count = 0
        for chunk in frame_reader():
            for frame in chunk:
                frame_time = frame_count / fps
                if end_time and frame_time > end_time:
                    return
                
                is_keyframe, last_frame_gray, frame_diff = process_frame(
                    frame, last_frame_gray, threshold, frame_time, last_keyframe_time,
                    min_scene_length, max_motion_factor, content_weight, fps
                )
                
                frame_diffs.append(frame_diff)
                
                if is_keyframe:
                    save_keyframe(frame, keyframePath, frame_count)
                    last_keyframe_time = frame_time
                    print(f"Keyframe detected at {frame_time:.2f} seconds")
                
                frame_count += 1
    finally:
        process.stdout.close()
        process.wait()

def process_frame(frame, last_frame_gray, base_threshold, frame_time, last_keyframe_time,
                  min_scene_length, max_motion_factor, content_weight, fps):
    gray_frame = convert_frame_to_grayscale(frame)
    
    if last_frame_gray is None:
        return True, gray_frame, 0  # First frame is always a keyframe
    
    # Compute frame difference
    frame_diff = np.mean(np.abs(gray_frame.astype(int) - last_frame_gray.astype(int)))
    
    # Apply tunable parameters
    time_since_last_keyframe = frame_time - last_keyframe_time
    motion_penalty = min(1.0, time_since_last_keyframe / min_scene_length)
    content_boost = content_weight * (1 + np.std(gray_frame) / 128)  # Boost for high-contrast frames
    
    # Adjust threshold
    adjusted_threshold = base_threshold * motion_penalty * max_motion_factor * content_boost
    
    is_keyframe = frame_diff > adjusted_threshold and time_since_last_keyframe >= min_scene_length
    
    return is_keyframe, gray_frame, frame_diff

def tune_keyframe_detection(sensitivity='balanced', content_type='general', min_scene_duration=1.0):
    # Sensitivity presets with lower thresholds
    sensitivity_presets = {
        'low': {'threshold': 15, 'max_motion_factor': 1.5, 'content_weight': 0.8},
        'balanced': {'threshold': 10, 'max_motion_factor': 1.2, 'content_weight': 1.0},
        'high': {'threshold': 5, 'max_motion_factor': 1.0, 'content_weight': 1.2}
    }
    
    # Content type adjustments
    content_type_adjustments = {
        'action': {'threshold_mult': 0.8, 'min_scene_length_mult': 0.5},
        'documentary': {'threshold_mult': 1.2, 'min_scene_length_mult': 1.5},
        'general': {'threshold_mult': 1.0, 'min_scene_length_mult': 1.0}
    }
    
    base_params = sensitivity_presets[sensitivity]
    content_adj = content_type_adjustments[content_type]
    
    return {
        'threshold': base_params['threshold'] * content_adj['threshold_mult'],
        'min_scene_length': min_scene_duration * content_adj['min_scene_length_mult'],
        'max_motion_factor': base_params['max_motion_factor'],
        'content_weight': base_params['content_weight']
    }