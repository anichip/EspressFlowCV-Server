# EspressFlowCV Server

Minimal Flask API server for EspressFlowCV iOS app.

## Features

- Health check endpoint
- Mock video analysis (returns fake results)
- Basic shot storage (in-memory for now)
- Full CORS support for iOS app

## Deployment

Designed for Railway deployment with minimal dependencies.

## Endpoints

- `GET /` - Welcome message
- `GET /api/health` - Health check
- `GET /api/shots` - Get all shots
- `GET /api/stats` - Get statistics
- `POST /api/analyze` - Analyze video (mock for now)
- `DELETE /api/shots/{id}` - Delete shot