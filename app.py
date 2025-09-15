#!/usr/bin/env python3
"""
EspressFlowCV Server - Minimal Version
Ultra-lightweight Flask API for Railway deployment
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for iOS app

# Simple in-memory storage for testing
shots_data = []

@app.route('/')
def home():
    return "☕ EspressFlowCV API Server - Running on Railway! 🚀"

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'EspressFlowCV server is running',
        'version': '1.0-minimal',
        'timestamp': datetime.now().isoformat(),
        'shots_count': len(shots_data)
    }), 200

@app.route('/api/shots', methods=['GET'])
def get_shots():
    """Get all shots (minimal version)"""
    return jsonify({
        'shots': shots_data,
        'count': len(shots_data),
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get basic stats"""
    total = len(shots_data)
    good_shots = len([s for s in shots_data if s.get('analysis_result') == 'good'])

    return jsonify({
        'summary': {
            'total_shots': total,
            'good_shots': good_shots,
            'under_shots': total - good_shots,
            'good_percentage': round((good_shots / total * 100) if total > 0 else 0, 1),
            'under_percentage': round(((total - good_shots) / total * 100) if total > 0 else 0, 1)
        },
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Minimal video analysis - just returns mock data for now"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    # Mock analysis result
    shot_id = len(shots_data) + 1
    result = {
        'shot_id': shot_id,
        'filename': f'shot_{shot_id}.mp4',
        'analysis_result': 'good',
        'confidence': 0.85,
        'features': {},
        'timestamp': datetime.now().isoformat()
    }

    # Store in memory
    shots_data.append({
        'id': shot_id,
        'filename': result['filename'],
        'recorded_at': datetime.now().isoformat(),
        'analysis_result': result['analysis_result'],
        'confidence': result['confidence'],
        'notes': ''
    })

    return jsonify(result), 200

@app.route('/api/shots/<int:shot_id>', methods=['DELETE'])
def delete_shot(shot_id):
    """Delete shot by ID"""
    global shots_data
    shots_data = [s for s in shots_data if s.get('id') != shot_id]

    return jsonify({
        'message': f'Shot {shot_id} deleted',
        'timestamp': datetime.now().isoformat()
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting EspressFlowCV server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)