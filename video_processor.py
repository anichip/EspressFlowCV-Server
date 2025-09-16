#!/usr/bin/env python3
"""
Complete Video Processing Pipeline for EspressFlowCV Railway Server
Handles frame extraction, feature analysis, and ML classification
"""

import os
import tempfile
import shutil
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Complete video processing pipeline for espresso shot analysis"""

    def __init__(self):
        """Initialize video processor with ML components"""
        self.model = None
        self.model_metadata = None
        self._load_dependencies()

    def _load_dependencies(self):
        """Load heavy dependencies only when needed"""
        try:
            # Import ML and CV libraries
            import cv2
            import joblib
            import pandas as pd
            import numpy as np

            # Import feature extraction
            from espresso_flow_features import process_frames_folder

            # Store references
            self.cv2 = cv2
            self.pd = pd
            self.np = np
            self.extract_features = process_frames_folder

            # Load model components
            self.model = joblib.load('espresso_model.joblib')
            self.model_metadata = joblib.load('model_metadata.joblib')

            logger.info("✅ VideoProcessor dependencies loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load VideoProcessor dependencies: {str(e)}")
            raise

    def extract_frames_from_video(self, video_path: str,
                                 clip_duration_sec: int = 7,  # Keep 7 seconds
                                 target_fps: int = 30,  # 30 fps = 210 frames (middle ground)
                                 skip_initial_sec: int = 1) -> Optional[str]:
        """Extract frames from video for analysis"""
        cap = self.cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"⚠️ Couldn't open video: {video_path}")
            return None

        try:
            # Get video properties
            fps = cap.get(self.cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))

            logger.info(f"📹 Processing video | FPS: {fps} | Total Frames: {total_frames}")

            # Calculate frame limits for efficient extraction
            max_frames = int(clip_duration_sec * target_fps)  # Only frames we need
            start_time_ms = skip_initial_sec * 1000  # Convert to milliseconds

            # Seek directly to start time (much faster than reading every frame)
            cap.set(self.cv2.CAP_PROP_POS_MSEC, start_time_ms)

            # Create temporary directory for frames
            temp_dir = tempfile.mkdtemp(prefix="espresso_frames_")

            saved_frame_count = 0
            last_valid_frame = None

            while cap.isOpened() and saved_frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Save frame
                frame_filename = f"frame_{saved_frame_count:04d}.jpg"
                frame_path = os.path.join(temp_dir, frame_filename)

                success = self.cv2.imwrite(frame_path, frame)
                if success:
                    last_valid_frame = frame_path
                    saved_frame_count += 1

            cap.release()

            logger.info(f"✅ Extracted {saved_frame_count} frames to {temp_dir}")
            return temp_dir

        except Exception as e:
            logger.error(f"❌ Error extracting frames: {e}")
            cap.release()
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return None

    def analyze_features(self, frames_dir: str) -> Dict[str, Any]:
        """Extract features from frame directory"""
        try:
            # Extract features using the espresso_flow_features module
            features = self.extract_features(frames_dir)

            if features is None:
                return {
                    'success': False,
                    'error': 'Feature extraction failed',
                    'features': {}
                }

            return {
                'success': True,
                'features': features,
                'frames_processed': len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
            }

        except Exception as e:
            logger.error(f"❌ Error analyzing features: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'features': {}
            }

    def classify_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Classify features using ML model"""
        try:
            # Prepare feature vector
            feature_columns = self.model_metadata.get('feature_names', [])
            feature_vector = []

            for col in feature_columns:
                value = features.get(col, 0.0)
                feature_vector.append(value)

            # Convert to numpy array and reshape
            X = self.np.array(feature_vector).reshape(1, -1)

            # Get prediction probability
            prob = self.model.predict_proba(X)[0]
            good_prob = prob[1]  # Probability of "good"

            # Apply optimal threshold
            optimal_threshold = self.model_metadata.get('optimal_threshold', 0.5)
            classification = 'good' if good_prob >= optimal_threshold else 'under-extracted'

            return {
                'classification': classification,
                'confidence': float(good_prob),
                'probabilities': {
                    'under-extracted': float(prob[0]),
                    'good': float(prob[1])
                },
                'threshold_used': optimal_threshold
            }

        except Exception as e:
            logger.error(f"❌ Error classifying features: {str(e)}")
            return {
                'classification': 'error',
                'confidence': 0.0,
                'error': str(e)
            }

    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Complete video processing pipeline"""
        logger.info(f"🎬 Starting video processing: {video_path}")

        # Extract frames
        frames_dir = self.extract_frames_from_video(video_path)
        if not frames_dir:
            return {
                'success': False,
                'error': 'Frame extraction failed',
                'classification': 'error',
                'confidence': 0.0
            }

        try:
            # Analyze features
            feature_result = self.analyze_features(frames_dir)
            if not feature_result['success']:
                return {
                    'success': False,
                    'error': feature_result['error'],
                    'classification': 'error',
                    'confidence': 0.0
                }

            # Classify features
            classification_result = self.classify_features(feature_result['features'])

            # Combine results
            result = {
                'success': True,
                'classification': classification_result['classification'],
                'confidence': classification_result['confidence'],
                'features': feature_result['features'],
                'processing_info': {
                    'frames_processed': feature_result['frames_processed'],
                    'probabilities': classification_result.get('probabilities', {}),
                    'threshold_used': classification_result.get('threshold_used', 0.5)
                }
            }

            logger.info(f"✅ Video processing complete: {result['classification']} ({result['confidence']:.3f})")
            return result

        except Exception as e:
            logger.error(f"❌ Error in video processing pipeline: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'classification': 'error',
                'confidence': 0.0
            }

        finally:
            # Clean up temporary frames
            if frames_dir and os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
                logger.info(f"🧹 Cleaned up frames: {frames_dir}")

def cleanup_temp_files(file_path: str) -> None:
    """Clean up temporary files"""
    if file_path and os.path.exists(file_path):
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
        else:
            os.remove(file_path)
        logger.info(f"🧹 Cleaned up: {file_path}")