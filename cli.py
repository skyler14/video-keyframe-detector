import argparse
from KeyFrameDetector.key_frame_detector import keyframe_detection, tune_keyframe_detection

def main():
    parser = argparse.ArgumentParser(description="Key Frame Detector")
    parser.add_argument('-s', '--source', help='Source video file', required=True)
    parser.add_argument('-d', '--dest', help='Destination folder', required=True)
    parser.add_argument('--sensitivity', choices=['low', 'balanced', 'high'], default='balanced',
                        help='Sensitivity of keyframe detection')
    parser.add_argument('--content-type', choices=['action', 'documentary', 'general'], default='general',
                        help='Type of video content')
    parser.add_argument('--min-scene-duration', type=float, default=1.0,
                        help='Minimum duration between keyframes in seconds')
    parser.add_argument('-c', '--chunk-size', help='Number of frames to process at once', type=int, default=1000)
    parser.add_argument('--start-time', help='Start time for processing (format: HH:MM:SS)', default=None)
    parser.add_argument('--end-time', help='End time for processing (format: HH:MM:SS)', default=None)
    parser.add_argument('--use-original', action='store_true', help='Use the original algorithm')

    args = parser.parse_args()

    if args.use_original:
        # Use a fixed threshold for the original algorithm
        threshold = 0.3
        keyframe_detection(args.source, args.dest, threshold,
                           chunk_size=args.chunk_size,
                           start_time=args.start_time,
                           end_time=args.end_time,
                           use_original_algorithm=True)
    else:
        # Tune the detector based on user input
        tuned_params = tune_keyframe_detection(
            sensitivity=args.sensitivity,
            content_type=args.content_type,
            min_scene_duration=args.min_scene_duration
        )

        # Run keyframe detection with tuned parameters
        keyframe_detection(
            args.source, args.dest,
            chunk_size=args.chunk_size,
            start_time=args.start_time,
            end_time=args.end_time,
            use_original_algorithm=False,
            **tuned_params
        )

if __name__ == '__main__':
    main()