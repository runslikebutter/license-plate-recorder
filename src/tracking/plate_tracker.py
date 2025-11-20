"""
Advanced license plate tracking with OCR aggregation and confidence scoring
Integrates the proven tracking logic from zalpr.py with ByteTracker
"""

import time
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import supervision as sv

from detection.detector import Detection
from utils.logger import setup_logger
from utils.image_processing import calculate_iou


@dataclass
class PlateTrackData:
    """Stores comprehensive data for a tracked license plate"""
    track_id: int
    plate_readings: Counter = field(default_factory=Counter)
    confidence_scores: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    best_plate_text: str = 'Unknown'
    best_confidence: float = 0.0
    total_detections: int = 0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    first_seen_time: float = field(default_factory=time.time)
    last_seen_time: float = field(default_factory=time.time)
    has_triggered_recording: bool = False
    zone_entry_frame: Optional[int] = None
    best_detection_bbox: Optional[List[float]] = None
    ocr_history: List[Dict[str, Any]] = field(default_factory=list)

    def update_ocr(self, ocr_text: str, confidence: float, frame_number: int, bbox: List[float]):
        """Update track with new OCR reading"""
        self.total_detections += 1
        self.last_seen_frame = frame_number
        self.last_seen_time = time.time()

        # Store OCR history entry
        ocr_entry = {
            'frame': frame_number,
            'text': ocr_text,
            'confidence': confidence,
            'bbox': bbox.copy(),
            'timestamp': time.time()
        }
        self.ocr_history.append(ocr_entry)

        # Keep only recent OCR history (last 50 entries)
        if len(self.ocr_history) > 50:
            self.ocr_history.pop(0)

        # Process OCR result if available
        if ocr_text and ocr_text.strip():
            plate_text = ocr_text.strip().upper()

            # Keep all readings regardless of confidence level
            self.plate_readings[plate_text] += 1
            self.confidence_scores[plate_text].append(confidence)

            # Update best plate based on frequency and confidence
            self.best_plate_text, self.best_confidence = self._calculate_best_plate()

            # Update best detection bbox if this has higher confidence
            if confidence > self.best_confidence * 0.8:  # Within 80% of best confidence
                self.best_detection_bbox = bbox.copy()

    def _calculate_best_plate(self) -> Tuple[str, float]:
        """Calculate best plate text using frequency and confidence weighting"""
        if not self.plate_readings:
            return 'Unknown', 0.0

        best_text = 'Unknown'
        best_confidence = 0.0
        highest_score = -1

        for plate_text, occurrence_count in self.plate_readings.items():
            if plate_text in self.confidence_scores:
                avg_confidence = np.mean(self.confidence_scores[plate_text])
                # Weight frequency and confidence equally
                combined_score = occurrence_count * avg_confidence

                if combined_score > highest_score:
                    highest_score = combined_score
                    best_text = plate_text
                    best_confidence = avg_confidence

        # Fallback to most frequent if no confidence scores
        if best_text == 'Unknown' and self.plate_readings:
            best_text = self.plate_readings.most_common(1)[0][0]

        return best_text, best_confidence

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive track summary"""
        return {
            'track_id': self.track_id,
            'best_plate': self.best_plate_text,
            'best_confidence': self.best_confidence,
            'detection_count': self.total_detections,
            'all_readings': dict(self.plate_readings),
            'frames_tracked': self.last_seen_frame - self.first_seen_frame + 1,
            'duration_seconds': self.last_seen_time - self.first_seen_time,
            'has_triggered_recording': self.has_triggered_recording,
            'zone_entry_frame': self.zone_entry_frame,
            'best_bbox': self.best_detection_bbox
        }


class LicensePlateTracker:
    """License plate tracking with OCR aggregation and confidence scoring"""

    def __init__(self,
                 frame_rate: float = 30,
                 track_thresh: float = 0.3,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 confidence_aggregation: bool = True,
                 min_detections_for_recording: int = 3):
        """
        Initialize license plate tracker

        Args:
            frame_rate: Video frame rate for ByteTracker
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: Threshold for matching detections to tracks
            confidence_aggregation: Enable OCR confidence aggregation
            min_detections_for_recording: Minimum detections before triggering recording
        """
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.confidence_aggregation = confidence_aggregation
        self.min_detections_for_recording = min_detections_for_recording

        self.logger = setup_logger(self.__class__.__name__)

        # Initialize ByteTracker (compatible with different supervision versions)
        try:
            self.byte_tracker = sv.ByteTrack(
                track_activation_threshold=track_thresh,
                lost_track_buffer=track_buffer,
                minimum_matching_threshold=match_thresh,
                frame_rate=int(frame_rate)
            )
        except TypeError:
            # Fallback for older supervision versions
            self.byte_tracker = sv.ByteTrack(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                frame_rate=int(frame_rate)
            )

        # Tracking state
        self.active_tracks: Dict[int, PlateTrackData] = {}
        self.total_tracks_created = 0
        self.current_frame = 0

        self.logger.info(
            f"LicensePlateTracker initialized: thresh={track_thresh}, "
            f"buffer={track_buffer}, match={match_thresh}"
        )

    @staticmethod
    def _calculate_text_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two OCR text strings"""
        if not text1 or not text2:
            return 0.0
        
        text1 = text1.upper().strip()
        text2 = text2.upper().strip()
        
        if text1 == text2:
            return 1.0
        
        # Use sequence matcher for fuzzy matching
        return SequenceMatcher(None, text1, text2).ratio()

    def _find_track_by_ocr(self, detection: Detection, max_frame_gap: int = 5) -> Optional[int]:
        """Find an existing track that matches this detection based on OCR similarity"""
        if not detection.ocr_text or detection.ocr_text == 'Unknown':
            return None
        
        best_track_id = None
        best_similarity = 0.7  # Minimum similarity threshold
        
        for track_id, track_data in self.active_tracks.items():
            # Only consider tracks seen recently
            frame_gap = self.current_frame - track_data.last_seen_frame
            if frame_gap > max_frame_gap:
                continue
            
            # Skip tracks that already triggered recording to avoid duplicates
            if track_data.has_triggered_recording:
                continue
            
            # Compare with best plate text
            if track_data.best_plate_text and track_data.best_plate_text != 'Unknown':
                similarity = self._calculate_text_similarity(detection.ocr_text, track_data.best_plate_text)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_track_id = track_id
        
        return best_track_id

    def update(self, detections: List[Detection], frame_number: int) -> List[Detection]:
        """
        Update tracking with new detections

        Args:
            detections: List of Detection objects with license plate info
            frame_number: Current frame number

        Returns:
            List of detections with tracking information added
        """
        self.current_frame = frame_number
        current_time = time.time()

        if not detections:
            # No detections - run empty update to maintain tracker state
            empty_detections = sv.Detections.empty()
            self.byte_tracker.update_with_detections(empty_detections)
            return []

        # Convert detections to supervision format
        xyxy = np.array([det.bbox for det in detections], dtype=np.float32)
        confidence = np.array([det.confidence for det in detections], dtype=np.float32)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence
        )

        # Update ByteTracker
        tracked_detections = self.byte_tracker.update_with_detections(sv_detections)

        # Process tracking results
        tracked_plates = []
        current_track_ids = set()

        if len(tracked_detections) > 0 and hasattr(tracked_detections, 'tracker_id'):
            for i, (bbox, track_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
                track_id = int(track_id)
                current_track_ids.add(track_id)

                # Find matching original detection
                matched_detection = self._find_matching_detection(detections, bbox.tolist())
                if matched_detection is None:
                    continue

                # Update or create track data
                if track_id not in self.active_tracks:
                    # New track
                    self.active_tracks[track_id] = PlateTrackData(
                        track_id=track_id,
                        first_seen_frame=frame_number,
                        last_seen_frame=frame_number,
                        first_seen_time=current_time,
                        last_seen_time=current_time
                    )
                    self.total_tracks_created += 1
                    self.logger.info(f"Created new track {track_id} for license plate")

                track_data = self.active_tracks[track_id]

                # Update track with OCR information
                ocr_text = matched_detection.ocr_text or "Unknown"
                ocr_confidence = matched_detection.ocr_confidence or 0.0

                track_data.update_ocr(ocr_text, ocr_confidence, frame_number, bbox.tolist())
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        "Track %s update: detections=%s, best='%s' (%.2f)",
                        track_id,
                        track_data.total_detections,
                        track_data.best_plate_text or 'Unknown',
                        track_data.best_confidence or 0.0,
                    )

                # Add tracking information to detection
                matched_detection.track_id = track_id
                matched_detection.ocr_text = track_data.best_plate_text
                matched_detection.ocr_confidence = track_data.best_confidence

                # Determine if this detection should trigger recording
                should_trigger = self._should_trigger_recording(track_data, log_reason=True)
                matched_detection.should_trigger_recording = should_trigger

                if should_trigger and not track_data.has_triggered_recording:
                    self.logger.info(
                        f"Track {track_id} ready for recording: '{track_data.best_plate_text}' "
                        f"(conf: {track_data.best_confidence:.2f}, detections: {track_data.total_detections})"
                    )

                tracked_plates.append(matched_detection)

                # Log tracking progress
                if track_data.total_detections % 10 == 0:  # Log every 10 detections
                    self.logger.debug(
                        f"Track {track_id}: '{track_data.best_plate_text}' "
                        f"(conf: {track_data.best_confidence:.2f}, {track_data.total_detections} detections)"
                    )

        # Try OCR-based matching for detections that weren't matched by ByteTracker
        matched_detection_ids = {id(det) for det in tracked_plates}
        unmatched_detections = [det for det in detections if id(det) not in matched_detection_ids]
        
        for detection in unmatched_detections:
            # Only try OCR matching for high-confidence detections with readable text
            if detection.confidence < 0.7 or not detection.ocr_text or detection.ocr_text == 'Unknown':
                continue
            
            # Try to find an existing track by OCR similarity
            track_id = self._find_track_by_ocr(detection)
            
            if track_id is not None:
                # Match found! Add this detection to the existing track
                track_data = self.active_tracks[track_id]
                current_track_ids.add(track_id)
                
                # Update track with OCR information
                track_data.update_ocr(
                    detection.ocr_text,
                    detection.ocr_confidence or 0.0,
                    frame_number,
                    detection.bbox
                )
                
                self.logger.debug(
                    f"OCR-matched detection to track {track_id}: '{detection.ocr_text}' "
                    f"(similarity with '{track_data.best_plate_text}')"
                )
                
                # Add tracking information to detection
                detection.track_id = track_id
                detection.ocr_text = track_data.best_plate_text
                detection.ocr_confidence = track_data.best_confidence
                
                # Determine if this detection should trigger recording
                should_trigger = self._should_trigger_recording(track_data, log_reason=True)
                detection.should_trigger_recording = should_trigger
                
                if should_trigger and not track_data.has_triggered_recording:
                    self.logger.info(
                        f"Track {track_id} ready for recording: '{track_data.best_plate_text}' "
                        f"(conf: {track_data.best_confidence:.2f}, detections: {track_data.total_detections})"
                    )
                
                tracked_plates.append(detection)

        # Clean up old tracks
        self._cleanup_old_tracks(current_track_ids, current_time)

        return tracked_plates

    def _find_matching_detection(self, detections: List[Detection], tracked_bbox: List[float]) -> Optional[Detection]:
        """Find the detection that best matches the ByteTracker result"""
        best_match = None
        best_iou = 0.0

        for detection in detections:
            iou = calculate_iou(detection.bbox, tracked_bbox)

            # Lower threshold for fast-moving objects
            # OCR-based matching handles cases where IoU is too low
            if iou > best_iou and iou > 0.15:
                best_iou = iou
                best_match = detection

        return best_match

    def _should_trigger_recording(self, track_data: PlateTrackData, log_reason: bool = False) -> bool:
        """Determine if a track should trigger a recording"""
        # Don't trigger if already triggered
        if track_data.has_triggered_recording:
            return False

        # Must have minimum number of detections
        if track_data.total_detections < self.min_detections_for_recording:
            if log_reason:
                self.logger.debug(
                    "Track %s not ready: detections %s/%s",
                    track_data.track_id,
                    track_data.total_detections,
                    self.min_detections_for_recording,
                )
            return False

        # Must have readable plate text
        if track_data.best_plate_text == 'Unknown' or not track_data.best_plate_text:
            if log_reason:
                self.logger.debug(
                    "Track %s not ready: OCR text unavailable (detections=%s)",
                    track_data.track_id,
                    track_data.total_detections,
                )
            return False

        # Must have reasonable confidence if aggregation is enabled
        if self.confidence_aggregation and track_data.best_confidence < 0.3:
            if log_reason:
                self.logger.debug(
                    "Track %s not ready: OCR confidence %.2f < %.2f",
                    track_data.track_id,
                    track_data.best_confidence,
                    0.3,
                )
            return False

        return True

    def mark_recording_triggered(self, track_ids: List[int]):
        """Mark tracks as having triggered a recording"""
        for track_id in track_ids:
            if track_id in self.active_tracks:
                track_data = self.active_tracks[track_id]
                track_data.has_triggered_recording = True
                self.logger.info(f"Marked track {track_id} as recorded: '{track_data.best_plate_text}'")

    def mark_zone_entry(self, track_id: int, frame_number: int):
        """Mark when a track enters the detection zone"""
        if track_id in self.active_tracks:
            track_data = self.active_tracks[track_id]
            if track_data.zone_entry_frame is None:
                track_data.zone_entry_frame = frame_number
                self.logger.debug(f"Track {track_id} entered detection zone at frame {frame_number}")

    def _cleanup_old_tracks(self, current_track_ids: set, current_time: float):
        """Remove tracks that haven't been seen for a while"""
        tracks_to_remove = []
        track_timeout = 300.0  # 5 minutes

        for track_id, track_data in self.active_tracks.items():
            time_since_seen = current_time - track_data.last_seen_time

            if time_since_seen > track_timeout:
                tracks_to_remove.append(track_id)

        if tracks_to_remove:
            removed_info = []
            for track_id in tracks_to_remove:
                track_data = self.active_tracks[track_id]
                removed_info.append(f"{track_id}:{track_data.best_plate_text}")
                del self.active_tracks[track_id]

            self.logger.info(f"Removed old tracks: {', '.join(removed_info)}")

    def get_track_info(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a specific track"""
        if track_id not in self.active_tracks:
            return None

        return self.active_tracks[track_id].get_summary()

    def get_all_tracks_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all active tracks"""
        return [track_data.get_summary() for track_data in self.active_tracks.values()]

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        active_count = len(self.active_tracks)
        recorded_count = sum(1 for track in self.active_tracks.values() if track.has_triggered_recording)

        return {
            'total_tracks_created': self.total_tracks_created,
            'active_tracks': active_count,
            'tracks_with_recordings': recorded_count,
            'current_frame': self.current_frame,
            'tracker_settings': {
                'track_thresh': self.track_thresh,
                'track_buffer': self.track_buffer,
                'match_thresh': self.match_thresh,
                'min_detections_for_recording': self.min_detections_for_recording
            }
        }

    def reset(self):
        """Reset tracker state"""
        self.active_tracks.clear()
        self.total_tracks_created = 0
        self.current_frame = 0

        # Reset ByteTracker (compatible with different supervision versions)
        try:
            self.byte_tracker = sv.ByteTrack(
                track_activation_threshold=self.track_thresh,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
                frame_rate=int(self.frame_rate)
            )
        except TypeError:
            # Fallback for older supervision versions
            self.byte_tracker = sv.ByteTrack(
                track_thresh=self.track_thresh,
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh,
                frame_rate=int(self.frame_rate)
            )

        self.logger.info("LicensePlateTracker reset")

    def get_tracks_ready_for_recording(self) -> List[int]:
        """Get list of track IDs that are ready to trigger recording"""
        ready_tracks = []
        for track_id, track_data in self.active_tracks.items():
            if self._should_trigger_recording(track_data):
                ready_tracks.append(track_id)
        return ready_tracks