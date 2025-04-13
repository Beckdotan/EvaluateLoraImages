# EvaluateLoraImages

## Overview
This application helps evaluate and compare AI-generated images against reference images using various quality metrics and visual analysis techniques.

## Installation
1. Clone this repository
2. Install dependencies:
```bash
npm install
cd server
pip install -r requirements.txt
```

## Usage
1. Start the backend server:
```bash
cd server
python server.py
```
2. Start the frontend development server:
```bash
npm run dev
```
3. Open http://localhost:3000 in your browser

## Architecture
- **Frontend**: React-based UI with three main pages (Home, Gallery, Analysis)
- **Backend**: Python Flask server providing analysis endpoints
- **Services**:
  - CLIP for semantic similarity
  - Gemini for visual analysis
  - MediaPipe for face/hand detection
  - PIQ for image quality metrics

## Third-party Services
1. **CLIP**: Used for measuring semantic similarity between images. Alternative considered: VGG16 (less accurate for cross-modal tasks).
2. **Gemini**: Provides detailed visual analysis and suggestions. Alternative: GPT-4 Vision (more expensive).
3. **MediaPipe**: Real-time face and hand detection. Alternative: OpenCV Haar cascades (less accurate).
4. **PIQ**: Image quality assessment metrics. Alternative: No-reference IQA (limited metrics).

## Configuration
- Backend configuration in `server/config/prompts.py`
- Frontend routes in `src/App.jsx`

## Troubleshooting
- Ensure Python 3.10+ and Node.js 18+ are installed
- Check server logs for backend errors
- Clear browser cache if UI issues occur