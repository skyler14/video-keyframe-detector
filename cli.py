import argparse
import os
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
    parser.add_argument('--min-time-constraint', type=float, help='Minimum time before considering a new keyframe')
    parser.add_argument('--max-time-constraint', type=float, help='Maximum time before forcing a new keyframe')
    parser.add_argument('--look-ahead-time', type=float, default=1.0,
                        help='Time to look ahead for better keyframe when using max time constraint')
    parser.add_argument('-c', '--chunk-size', help='Number of frames to process at once', type=int, default=1000)
    parser.add_argument('--start-time', help='Start time for processing (format: HH:MM:SS)', default=None)
    parser.add_argument('--end-time', help='End time for processing (format: HH:MM:SS)', default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--text', nargs='?', const='', default=None, 
                        choices=['', 'only', 'debug'],
                        help='Generate JSON metadata file. Use --text without argument for both JSON and images, '
                             '--text only for JSON only, --text debug for detailed JSON')

    args = parser.parse_args()

    # Tune the detector based on user input
    tuned_params = tune_keyframe_detection(
        sensitivity=args.sensitivity,
        content_type=args.content_type,
        min_scene_duration=args.min_scene_duration,
        min_time_constraint=args.min_time_constraint,
        max_time_constraint=args.max_time_constraint,
        look_ahead_time=args.look_ahead_time
    )

    # Determine text mode and whether to save images
    text_mode = args.text is not None
    save_images = args.text != 'only' and args.text != 'debug'
    debug_mode = args.text == 'debug'

    # Generate JSON filename based on input video
    video_basename = os.path.splitext(os.path.basename(args.source))[0]
    json_filename = f"{video_basename}_keyframes.json"

    # Run keyframe detection with tuned parameters
    keyframe_detection(
        args.source,
        args.dest,
        chunk_size=args.chunk_size,
        start_time=args.start_time,
        end_time=args.end_time,
        verbose=args.verbose,
        text_mode=text_mode,
        save_images=save_images,
        json_filename=json_filename,
        debug_mode=debug_mode,
        **tuned_params
    )

    if text_mode and not save_images:
        print("Text-only mode: No keyframe images were saved.")

if __name__ == '__main__':
    main()