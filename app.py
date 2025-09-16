#!/usr/bin/env python3
"""
EspressFlowCV Server - Production Railway Version
Full ML-powered Flask API for Railway deployment with conditional imports
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import tempfile
import uuid
from datetime import datetime
import logging
import json
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for iOS app

# Configuration
API_VERSION = "2.0-production"
MAX_VIDEO_SIZE_MB = 100
UPLOAD_FOLDER = tempfile.mkdtemp(prefix='espresso_uploads_')

# Global variables for ML components (loaded on demand)
model = None
model_metadata = None
video_processor = None

def load_ml_components():
    """Load ML components on demand to avoid Railway startup issues"""
    global model, model_metadata, video_processor

    if model is not None:
        return True  # Already loaded

    try:
        logger.info("🤖 Loading ML components...")

        # Import heavy dependencies only when needed
        import joblib
        import pandas as pd
        import numpy as np

        # Load model and metadata
        model = joblib.load('espresso_model.joblib')
        model_metadata = joblib.load('model_metadata.joblib')

        # Import video processor
        from video_processor import VideoProcessor
        video_processor = VideoProcessor()

        logger.info("✅ ML components loaded successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to load ML components: {str(e)}")
        return False

@app.route('/')
def home():
    return "☕ EspressFlowCV API Server - Full Production on Railway! 🚀"

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Test ML loading
    ml_available = load_ml_components()

    return jsonify({
        'status': 'healthy',
        'message': 'EspressFlowCV server is running',
        'version': API_VERSION,
        'ml_available': ml_available,
        'timestamp': datetime.now().isoformat(),
        'upload_folder': UPLOAD_FOLDER
    }), 200

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check what's available"""
    import sys
    import os

    debug_info = {
        'python_version': sys.version,
        'current_dir': os.getcwd(),
        'files_present': os.listdir('.'),
        'model_files_exist': {
            'espresso_model.joblib': os.path.exists('espresso_model.joblib'),
            'model_metadata.joblib': os.path.exists('model_metadata.joblib'),
            'espresso_flow_features.py': os.path.exists('espresso_flow_features.py')
        }
    }

    # Test individual imports
    try:
        import cv2
        debug_info['cv2_version'] = cv2.__version__
    except Exception as e:
        debug_info['cv2_error'] = str(e)

    try:
        import sklearn
        debug_info['sklearn_version'] = sklearn.__version__
    except Exception as e:
        debug_info['sklearn_error'] = str(e)

    try:
        import joblib
        debug_info['joblib_available'] = True
    except Exception as e:
        debug_info['joblib_error'] = str(e)

    return jsonify(debug_info), 200

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Analyze uploaded video with ML model"""
    try:
        # Check if video file is provided
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400

        # Load ML components
        if not load_ml_components():
            return jsonify({'error': 'ML components not available'}), 500

        # Save video temporarily
        video_id = str(uuid.uuid4())
        video_filename = f"{video_id}.mov"
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        video_file.save(video_path)

        logger.info(f"📹 Processing video: {video_filename}")

        # Process video
        result = video_processor.process_video(video_path)

        # Clean up temporary file
        os.remove(video_path)

        # Generate simple integer ID for iOS compatibility
        shot_id_int = hash(video_id) % 1000000  # Convert UUID to simple int

        # Format response
        response = {
            'shot_id': shot_id_int,
            'filename': video_file.filename,
            'analysis_result': result.get('classification', 'unknown'),
            'confidence': round(result.get('confidence', 0.0), 3),
            'features': result.get('features', {}),
            'processing_info': result.get('processing_info', {}),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✅ Analysis complete: {response['analysis_result']} ({response['confidence']})")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ Error analyzing video: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/shots', methods=['GET'])
def get_shots():
    """Get all shots - placeholder for now"""
    return jsonify({
        'shots': [],
        'count': 0,
        'message': 'Shot history not implemented in this version',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get basic stats - placeholder for now"""
    return jsonify({
        'summary': {
            'total_shots': 0,
            'good_shots': 0,
            'under_shots': 0,
            'good_percentage': 0,
            'under_percentage': 0
        },
        'message': 'Stats not implemented in this version',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if not load_ml_components():
        return jsonify({'error': 'ML components not available'}), 500

    return jsonify({
        'model_type': 'Random Forest',
        'version': '2.0',
        'features_count': len(model_metadata.get('feature_names', [])),
        'optimal_threshold': model_metadata.get('optimal_threshold', 0.5),
        'roc_auc_score': model_metadata.get('roc_auc_score', 0.0),
        'timestamp': datetime.now().isoformat()
    }), 200

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': f'File too large. Maximum size: {MAX_VIDEO_SIZE_MB}MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))

    # Create upload directory
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"📁 Upload folder: {UPLOAD_FOLDER}")

    # Test ML loading at startup (but don't fail if it doesn't work)
    logger.info("🧪 Testing ML components at startup...")
    ml_ready = load_ml_components()
    logger.info(f"   ML Ready: {ml_ready}")

    logger.info(f"🚀 Starting EspressFlowCV server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)