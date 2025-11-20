#!/usr/bin/env python3
"""
License Plate Detection and Recording System
A comprehensive system for detecting, tracking, and recording license plates
from RTSP streams, video files, or batch directory processing
"""

import sys
import signal
import time
import cv2
import numpy as np
import argparse
import gc
import psutil
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.visualization import Visualizer
from src.utils.detection_zone import DetectionZone
from src.utils.zone_editor import ZoneEditor
from src.detection.yolo_lpd_detector import YOLOLPDDetector
from src.tracking.plate_tracker import LicensePlateTracker
from src.recording.plate_recorder import PlateRecorder
from src.capture.rtsp_capture import RTSPCapture
from src.capture.file_capture import FileCapture
from src.capture.directory_capture import DirectoryCapture


class MemoryMonitor:
    """Monitor memory usage and trigger cleanup when needed"""

    def __init__(self, logger, threshold_percent=80):
        self.logger = logger
        self.threshold_percent = threshold_percent
        self.process = psutil.Process(os.getpid())
        self.last_cleanup = time.time()
        self.cleanup_interval = 3600  # Force cleanup every hour

    def get_memory_info(self):
        """Get current memory usage"""
        mem_info = self.process.memory_info()
        mem_percent = self.process.memory_percent()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,
            'vms_mb': mem_info.vms / 1024 / 1024,
            'percent': mem_percent
        }

    def should_cleanup(self):
        """Check if cleanup is needed"""
        mem_info = self.get_memory_info()
        time_since_cleanup = time.time() - self.last_cleanup

        return (mem_info['percent'] > self.threshold_percent or
                time_since_cleanup > self.cleanup_interval)

    def perform_cleanup(self):
        """Perform memory cleanup"""
        self.logger.info("Performing memory cleanup...")
        gc.collect()
        self.last_cleanup = time.time()

        # Log memory after cleanup
        mem_info = self.get_memory_info()
        self.logger.info(f"Memory after cleanup: {mem_info['rss_mb']:.1f}MB ({mem_info['percent']:.1f}%)")


class LicensePlateRecorder:
    """Main License Plate Detection and Recording System"""

    def __init__(self, config_path: str = "config.yaml", preview: bool = False, input_source: str = None):
        self.config = Config(config_path)
        self.logger = setup_logger(
            "LicensePlateRecorder",
            self.config.logging.get('file'),
            self.config.logging.get('level', 'INFO')
        )

        # Initialize components
        self.capture = None
        self.detector = None
        self.tracker = None
        self.recorder = None
        self.visualizer = None
        self.detection_zone = None
        self.zone_editor = None
        self.memory_monitor = None

        self.running = False
        self.width = 1280
        self.height = 720
        self.actual_fps = self.config.recording.get('default_fps', 30)

        # Input source configuration
        self.input_source = input_source
        self.input_type = self._determine_input_type(input_source)

        # Statistics
        self.recordings_saved = []
        self.total_detections = 0
        self.zone_detections = 0
        self.total_frames = 0
        self.start_time = time.time()

        # Preview settings
        self.preview_enabled = preview
        self.show_debug = False
        self.show_help = True
        self.paused = False
        self.screenshot_count = 0
        self.zone_editing = False

        # Performance tracking
        self.fps_tracker = []
        self.last_fps_time = time.time()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self.logger.info("Received shutdown signal")
        self.stop()

    def _determine_input_type(self, input_source):
        """Determine the type of input source"""
        if not input_source:
            return 'rtsp'  # Default to RTSP from config

        if input_source.startswith(('rtsp://', 'http://', 'https://')):
            return 'rtsp'
        elif os.path.isfile(input_source):
            return 'file'
        elif os.path.isdir(input_source):
            return 'directory'
        else:
            # Assume it's a file path even if it doesn't exist yet
            return 'file'

    def initialize(self):
        self.logger.info(f"Initializing License Plate Recorder (Preview: {self.preview_enabled})...")
        self.logger.info(f"Input type: {self.input_type}, Source: {self.input_source or 'config default'}")

        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(self.logger)

        # Initialize detector
        detector_config = self.config.detection
        engine_path = detector_config.get(
            'engine_path',
            'src/models/dev/car_lpd_11nv6.engine'
        )
        self.detector = YOLOLPDDetector(
            model_path=engine_path,
            confidence_threshold=detector_config.get('confidence_threshold', 0.3),
            iou_threshold=detector_config.get('iou_threshold', 0.45),
            input_size=detector_config.get('detection_input_size', 640),
            max_det=detector_config.get('max_det', 300),
            device='cuda:0',
            enable_ocr=detector_config.get('enable_ocr', True),
            ocr_engine_path=detector_config.get('ocr_engine_path'),
            ocr_config_path=detector_config.get('ocr_config_path')
        )

        # Initialize detection zone
        self.detection_zone = DetectionZone()
        if self.detection_zone.is_enabled():
            self.logger.info(f"Detection zone enabled with {self.detection_zone.get_zone_info(1920, 1080)['coverage']:.1f}% coverage")
        else:
            self.logger.info("Detection zone disabled (using full frame)")

        # Initialize tracker
        tracking_config = self.config.tracking
        self.tracker = LicensePlateTracker(
            frame_rate=self.actual_fps,
            track_thresh=tracking_config.get('track_thresh', 0.3),
            track_buffer=tracking_config.get('track_buffer', 30),
            match_thresh=tracking_config.get('match_thresh', 0.8),
            confidence_aggregation=tracking_config.get('confidence_aggregation', True),
            min_detections_for_recording=tracking_config.get('min_detections_for_recording', 3)
        )

        # Initialize recorder
        recording_config = self.config.recording
        output_dir = Path(recording_config['output_dir'])
        self.recorder = PlateRecorder(
            output_dir=str(output_dir),
            pre_plate_duration=recording_config.get('pre_plate_buffer', 3.0),
            post_plate_duration=recording_config.get('post_plate_buffer', 5.0),
            max_recording_duration=recording_config.get('max_duration', 30.0),
            fps=self.actual_fps,
            video_codec=recording_config.get('video_codec', 'mp4v')
        )

        # Initialize visualizer if preview is enabled
        if self.preview_enabled:
            self.visualizer = Visualizer()
            cv2.namedWindow('License Plate Recorder', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('License Plate Recorder', self._mouse_callback)
            self.logger.info("Preview window created. Press 'q' to quit, 'h' for help")

        # Initialize input capture
        if not self._initialize_capture():
            raise RuntimeError(f"Failed to initialize input capture: {self.input_source}")

        # Initialize zone editor with actual frame dimensions
        if self.preview_enabled:
            self.zone_editor = ZoneEditor(self.width, self.height)
            cv2.setMouseCallback('License Plate Recorder', self._mouse_callback)
            self.logger.info(f"Zone editor initialized ({self.width}x{self.height}). Press 'e' for zone editing")

        # Log initial memory usage
        mem_info = self.memory_monitor.get_memory_info()
        self.logger.info(f"Initial memory: {mem_info['rss_mb']:.1f}MB ({mem_info['percent']:.1f}%)")

        self.logger.info("Initialization complete")

    def _initialize_capture(self) -> bool:
        """Initialize the appropriate capture based on input type"""
        try:
            if self.input_type == 'rtsp':
                # RTSP stream
                if self.input_source:
                    # Override config with command line URL
                    self.config.config['stream']['rtsp_url'] = self.input_source
                self.capture = RTSPCapture(self.config)
                if not self.capture.connect():
                    self.logger.error("Failed to connect to RTSP stream")
                    return False
                self.actual_fps = float(self.capture.get_fps())
                self.width, self.height = self.capture.get_resolution()

            elif self.input_type == 'file':
                # Single video file
                self.capture = FileCapture(self.input_source, maintain_fps=not self.preview_enabled)
                if not self.capture.connect():
                    self.logger.error("Failed to open video file")
                    return False
                self.actual_fps = self.capture.get_fps()
                self.width, self.height = self.capture.get_resolution()

            elif self.input_type == 'directory':
                # Directory batch processing
                self.capture = DirectoryCapture(self.input_source, recursive=True, maintain_fps=not self.preview_enabled)
                if not self.capture.initialize():
                    self.logger.error("Failed to initialize directory capture")
                    return False
                # Connect to first file to get dimensions
                if not self.capture.connect_next_file():
                    self.logger.error("No valid video files found in directory")
                    return False
                first_file_info = self.capture.get_current_file_info()
                if first_file_info:
                    self.actual_fps = first_file_info['fps']
                    self.width, self.height = first_file_info['resolution']

            else:
                self.logger.error(f"Unknown input type: {self.input_type}")
                return False

            # Update tracker and recorder with actual FPS
            if hasattr(self.tracker, 'frame_rate'):
                self.tracker.frame_rate = self.actual_fps
            if hasattr(self.recorder, 'fps'):
                self.recorder.fps = self.actual_fps

            self.logger.info(f"Capture initialized: {self.width}x{self.height} @ {self.actual_fps}fps")

            # Set optimal crop position based on detection zone location
            if self.detection_zone.is_enabled():
                optimal_crop = self.detection_zone.get_optimal_crop_position(self.width, self.height)
                self.detector.set_crop_position(optimal_crop)
                self.logger.info(f"Set crop position to '{optimal_crop}' based on detection zone location")

            return True

        except Exception as e:
            self.logger.error(f"Error initializing capture: {e}")
            return False

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param):
        """Handle mouse events for zone editing"""
        if self.zone_editor and self.zone_editing:
            self.zone_editor.handle_mouse_event(event, x, y, flags)

    def process_frame(self, frame, frame_number: int):
        """Process a single frame"""
        if frame is None or frame.size == 0:
            self.logger.warning("Received invalid frame")
            return [], []

        self.total_frames += 1

        # Always add frame to recorder for buffering
        self.recorder.add_frame(frame)

        # Run license plate detection
        try:
            detections = self.detector.detect(frame)
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            detections = []

        # Apply detection zone filtering
        if self.detection_zone.is_enabled():
            zone_detections = self.detection_zone.filter_detections(detections, self.width, self.height)
        else:
            zone_detections = detections

        # Update tracker
        tracked_detections = self.tracker.update(zone_detections, frame_number)

        # Check for recording triggers
        ready_tracks = self.tracker.get_tracks_ready_for_recording()
        if ready_tracks and not self.recorder.is_recording():
            # Start recording for the first ready track
            track_info = self.tracker.get_track_info(ready_tracks[0])
            if track_info:
                plate_text = track_info['best_plate']
                success = self.recorder.trigger_recording(plate_text)
                if success:
                    self.tracker.mark_recording_triggered([ready_tracks[0]])
                    self.zone_detections += 1

        # Check if recording should stop
        if self.recorder.is_recording():
            # Recording will auto-stop based on time limits
            pass

        self.total_detections += len(detections)

        return detections, tracked_detections

    def handle_keyboard(self):
        """Handle keyboard input in preview mode"""
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.logger.info("Quit requested via keyboard")
            return False
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            if hasattr(self, 'last_preview_frame'):
                cv2.imwrite(filename, self.last_preview_frame)
                self.logger.info(f"Screenshot saved: {filename}")
                self.screenshot_count += 1
        elif key == ord('d'):
            self.show_debug = not self.show_debug
            self.logger.info(f"Debug display: {'ON' if self.show_debug else 'OFF'}")
        elif key == ord('h'):
            self.show_help = not self.show_help
        elif key == ord(' '):
            self.paused = not self.paused
            self.logger.info(f"{'Paused' if self.paused else 'Resumed'}")
        elif key == ord('e'):
            # Toggle zone editing mode
            if self.zone_editor:
                self.zone_editing = not self.zone_editing
                self.zone_editor.set_editing_mode(self.zone_editing)

                if self.zone_editing:
                    # Initialize zone editor with current detection zone
                    zone_rect = self.detection_zone.calculate_zone(self.width, self.height)
                    if zone_rect:
                        self.zone_editor.left_x, self.zone_editor.top_y, self.zone_editor.right_x, self.zone_editor.bottom_y = zone_rect
                    self.zone_editor.on_zone_changed = self._update_zone_from_editor
                    self.logger.info("Zone editing mode ENABLED")
                else:
                    # Save final zone configuration
                    if self.zone_editor and self.detection_zone:
                        left_x, top_y, right_x, bottom_y = self.zone_editor.get_zone_rect()
                        self.detection_zone.update_zone_from_pixels(
                            top_y, bottom_y, left_x, right_x, self.width, self.height, save=True
                        )
                        top_p, bottom_p, left_p, right_p = self.zone_editor.get_zone_percentages()
                        self.logger.info(f"Zone saved: T:{top_p:.1f}% B:{bottom_p:.1f}% L:{left_p:.1f}% R:{right_p:.1f}%")

                        # Update crop position based on new zone location
                        optimal_crop = self.detection_zone.get_optimal_crop_position(self.width, self.height)
                        self.detector.set_crop_position(optimal_crop)
                        self.logger.info(f"Updated crop position to '{optimal_crop}' based on new zone location")
                    self.zone_editor.on_zone_changed = None
                    self.logger.info("Zone editing mode DISABLED - zone saved")
        elif key == ord('z'):
            # Toggle detection zone on/off
            if self.detection_zone.is_enabled():
                self.detection_zone.disable_zone(save=True)
                self.detector.set_crop_position('full')  # Default to full frame when zone disabled
                self.logger.info("Detection zone disabled - crop position set to full frame")
            else:
                self.detection_zone.reset_to_defaults(save=True)
                optimal_crop = self.detection_zone.get_optimal_crop_position(self.width, self.height)
                self.detector.set_crop_position(optimal_crop)
                self.logger.info(f"Detection zone enabled - crop position set to '{optimal_crop}'")

        return True

    def _update_zone_from_editor(self):
        """Update detection zone from zone editor"""
        if self.zone_editor and self.detection_zone:
            left_x, top_y, right_x, bottom_y = self.zone_editor.get_zone_rect()
            self.detection_zone.update_zone_from_pixels(
                top_y, bottom_y, left_x, right_x, self.width, self.height, 
                save=False
            )

    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        self.fps_tracker.append(current_time)
        # Keep only recent entries
        self.fps_tracker = [t for t in self.fps_tracker if current_time - t <= 1.0]
        return len(self.fps_tracker)

    def run(self):
        self.running = True
        frame_count = 0
        last_status_time = time.time()

        try:
            while self.running:
                # Check memory and cleanup if needed
                if self.memory_monitor.should_cleanup():
                    self.memory_monitor.perform_cleanup()

                if not self.paused:
                    # Read frame based on input type
                    frame = None
                    file_info = None

                    if self.input_type == 'rtsp':
                        frame = self.capture.read_frame()
                        if frame is None:
                            self.logger.error("Failed to read RTSP frame")
                            break
                    elif self.input_type == 'file':
                        self.logger.debug(f"Reading frame {frame_count}")
                        frame = self.capture.read_frame()
                        if frame is None:
                            self.logger.info("End of video file reached")
                            break
                        file_info = self.capture.get_file_info()
                        self.logger.debug(f"Frame read successfully: {frame.shape if frame is not None else None}")
                    elif self.input_type == 'directory':
                        result = self.capture.read_frame()
                        if result is None:
                            self.logger.info("All files processed")
                            break
                        frame, file_info = result

                    if frame is not None:
                        frame_count += 1

                        # Process frame (with detection)
                        try:
                            detections, tracked_detections = self.process_frame(frame, frame_count)
                        except Exception as e:
                            self.logger.error(f"Error processing frame: {e}")
                            continue

                        # Calculate FPS
                        current_fps = self.calculate_fps()

                        # Create preview if enabled
                        if self.preview_enabled:
                            zone_rect = None
                            zone_info = None
                            if self.detection_zone.is_enabled():
                                zone_rect = self.detection_zone.calculate_zone(self.width, self.height)
                                zone_info = self.detection_zone.get_zone_info(self.width, self.height)

                            # Get crop info from detector
                            crop_info = self.detector.get_crop_info(frame)

                            # Get tracking info
                            tracks_info = self.tracker.get_all_tracks_summary()

                            # Get buffer info
                            buffer_info = self.recorder.get_buffer_info()

                            preview_frame = self.visualizer.create_preview_frame(
                                frame=frame,
                                state=self.recorder.get_state().value,
                                detections=tracked_detections,
                                frame_count=frame_count,
                                detection_count=self.zone_detections,
                                recordings_saved=len(self.recordings_saved),
                                is_recording=self.recorder.is_recording(),
                                buffer_size=buffer_info['buffer_frames'],
                                max_buffer_size=buffer_info['max_buffer_frames'],
                                fps=current_fps,
                                zone_rect=zone_rect if not self.zone_editing else None,
                                zone_info=zone_info,
                                crop_info=crop_info,
                                tracks_info=tracks_info[:5],  # Show top 5 tracks
                                show_debug=self.show_debug,
                                show_help=self.show_help
                            )

                            # Add zone editing overlay if in editing mode
                            if self.zone_editing and self.zone_editor:
                                preview_frame = self.zone_editor.draw_edit_overlay(preview_frame)

                            self.last_preview_frame = preview_frame
                            cv2.imshow('License Plate Recorder', preview_frame)

                        # Log status periodically
                        current_time = time.time()
                        log_interval = 10 if self.preview_enabled else 30
                        if current_time - last_status_time >= log_interval:
                            last_status_time = current_time

                            recording_info = self.recorder.get_recording_info()
                            tracker_stats = self.tracker.get_statistics()

                            status_msg = (
                                f"Frame: {frame_count} | "
                                f"Plates: {self.zone_detections} | "
                                f"Tracks: {tracker_stats['active_tracks']} | "
                                f"State: {recording_info['state']}"
                            )

                            if file_info:
                                progress = file_info.get('progress_percent', 0)
                                status_msg += f" | Progress: {progress:.1f}%"

                            self.logger.info(status_msg)

                        # Handle finished recording
                        if not self.recorder.is_recording() and hasattr(self.recorder, '_last_output_file'):
                            # Check if we have a new recording
                            # This would need to be implemented in the recorder
                            pass

                else:
                    time.sleep(0.01)

                # Handle keyboard input if preview is enabled
                if self.preview_enabled:
                    if not self.handle_keyboard():
                        break

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.stop()

    def stop(self):
        self.logger.info("Stopping License Plate Recorder...")
        self.running = False

        # Stop any active recording
        if self.recorder and self.recorder.is_recording():
            output_file = self.recorder.stop_recording()
            if output_file:
                self.recordings_saved.append(output_file)

        # Clean up components
        if self.recorder:
            self.recorder.cleanup()

        if self.capture:
            self.capture.disconnect()

        if self.preview_enabled:
            cv2.destroyAllWindows()

        # Force garbage collection
        gc.collect()

        # Print final summary
        self.log_statistics()

    def log_statistics(self):
        """Log runtime statistics"""
        runtime = time.time() - self.start_time
        hours = runtime / 3600

        mem_info = self.memory_monitor.get_memory_info()

        self.logger.info("="*60)
        self.logger.info("LICENSE PLATE RECORDER STATISTICS")
        self.logger.info(f"Runtime: {hours:.1f} hours")
        self.logger.info(f"Memory: {mem_info['rss_mb']:.1f}MB ({mem_info['percent']:.1f}%)")
        self.logger.info(f"Total frames: {self.total_frames}")
        self.logger.info(f"Total detections: {self.total_detections}")
        self.logger.info(f"Zone detections: {self.zone_detections}")
        self.logger.info(f"Recordings saved: {len(self.recordings_saved)}")

        # Detector stats
        detector_stats = self.detector.get_performance_stats()
        if 'avg_inference_time' in detector_stats and detector_stats['avg_inference_time'] > 0:
            inference_fps = 1.0 / detector_stats['avg_inference_time']
            self.logger.info(f"Detector performance: {inference_fps:.1f} fps")
        if 'avg_ocr_time' in detector_stats and detector_stats['avg_ocr_time'] > 0:
            self.logger.info(f"OCR avg time: {detector_stats['avg_ocr_time']:.3f}s")

        # Tracker stats
        tracker_stats = self.tracker.get_statistics()
        self.logger.info(f"Tracks created: {tracker_stats['total_tracks_created']}")

        if self.recordings_saved:
            self.logger.info("\nRecorded files:")
            for i, file in enumerate(self.recordings_saved[-10:], 1):
                self.logger.info(f"  {i}. {file}")
            if len(self.recordings_saved) > 10:
                self.logger.info(f"  ... and {len(self.recordings_saved)-10} more")

        self.logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="License Plate Detection and Recording System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video file with preview
  python main.py --input video.mp4 --preview

  # RTSP stream in background
  python main.py --input rtsp://camera/stream

  # Batch process directory
  python main.py --input /path/to/videos/ --batch

  # Edit detection zone
  python main.py --input video.mp4 --preview --zone-edit
        """
    )

    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "-i", "--input",
        help="Input source: RTSP URL, video file, or directory"
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Override output directory from config"
    )
    parser.add_argument(
        "-p", "--preview",
        action="store_true",
        help="Enable preview window with visualizations"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch processing mode (for directories)"
    )
    parser.add_argument(
        "--crop-position",
        choices=['left', 'center', 'right', 'full'],
        help="Override crop position for detection"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        help="Override confidence threshold"
    )
    parser.add_argument(
        "--no-zone",
        action="store_true",
        help="Disable detection zone (use full frame)"
    )

    args = parser.parse_args()

    # Load and modify config based on arguments
    try:
        config = Config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        return 1

    if args.output_dir:
        config.config['recording']['output_dir'] = args.output_dir

    if args.debug:
        config.config['logging']['level'] = 'DEBUG'

    if args.crop_position:
        config.config['detection']['crop_position'] = args.crop_position

    if args.confidence:
        config.config['detection']['confidence_threshold'] = args.confidence

    if args.no_zone:
        # This will be handled by the detection zone system
        pass

    # Save modified config temporarily
    import tempfile
    import yaml

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config.config, f)
        temp_config_path = f.name

    # Create and run recorder
    recorder = LicensePlateRecorder(
        config_path=temp_config_path,
        preview=args.preview,
        input_source=args.input
    )

    # Clean up temp file after
    try:
        recorder.initialize()
        recorder.run()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        recorder.stop()
        os.unlink(temp_config_path)


if __name__ == "__main__":
    sys.exit(main())