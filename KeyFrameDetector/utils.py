import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_frame_to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

def prepare_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def plot_metrics(frame_times, frame_diffs, keyframe_times, fps):
    plt.figure(figsize=(12, 6))
    plt.plot(frame_times, frame_diffs, 'r-', label='Frame differences')
    plt.plot(keyframe_times, [frame_diffs[int(t * fps)] for t in keyframe_times], "bx", label='Detected keyframes')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frame difference')
    plt.title("Frame differences and detected keyframes")
    plt.legend()
    plt.show()

def save_keyframe(frame, keyframePath, frame_number):
    cv2.imwrite(os.path.join(keyframePath, f'keyframe{frame_number}.jpg'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))