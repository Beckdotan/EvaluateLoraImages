# Photo Evaluator

Photo Evaluator is a web application for evaluating AI-generated photos by comparing them with reference images of the original person. It uses face, head, and body detection to analyze similarities.

## Features

- Upload multiple reference images (1-15) of the original person
- Upload a single AI-generated image for comparison
- Interactive drag-and-drop interface for image uploads
- Support for various image formats including JPEG, PNG, and HEIC (automatically converted)
- Visualization of detection results with toggleable overlays
- Cross-platform compatibility

## Project Structure

The project consists of two main components:

1. **Frontend**: React-based web interface for uploading images and displaying results
2. **Backend**: Python FastAPI server for image processing and detection

## Prerequisites

Before running the application, make sure you have the following installed:

### Frontend Requirements
- Node.js (version 14.0.0 or higher)
- npm (usually comes with Node.js)

### Backend Requirements
- Python 3.8 or higher
- pip (Python package manager)

## Installation

### Frontend Setup

1. Clone the repository or download the source code
2. Open a terminal in the project directory
3. Install the frontend dependencies:

```bash
npm install
```

### Backend Setup

1. Navigate to the server directory:

```bash
cd server
```

2. Create a Python virtual environment (recommended):

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install the backend dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

### Starting the Backend Server

1. Make sure your Python virtual environment is activated
2. Navigate to the server directory if you're not already there
3. Start the FastAPI server:

```bash
python server.py
```

The server will start on http://localhost:8000

### Starting the Frontend Development Server

1. Open a new terminal window/tab
2. Navigate to the project root directory
3. Start the React development server:

```bash
npm run dev
```

The frontend will be available at http://localhost:5173

## Using the Application

1. Open your web browser and navigate to http://localhost:5173
2. On the home page:
   - Upload 1-15 reference images of the real person using the first dropzone
   - Upload 1 AI-generated image using the second dropzone
   - HEIC images will be automatically converted to JPEG in the browser before uploading
   - Click the "Evaluate Images" button
3. View the results on the Gallery page:
   - The reference images are displayed at the top
   - The generated image with detection overlays is shown below
   - Use the toggle buttons to show/hide face, head, and body detections

## Troubleshooting

### Image Upload Issues

If you encounter problems with image uploads:
- Make sure images are valid and not corrupted
- For large images, try reducing their size before uploading
- If HEIC conversion is slow, consider converting files to JPEG manually before uploading

### Server Connection Issues

If the frontend cannot connect to the backend:
- Verify the server is running at http://localhost:8000
- Check server logs in the terminal for error messages
- Ensure there are no firewall restrictions blocking the connection

### Detection Problems

If face/head/body detection doesn't work properly:
- Try using clearer images with better lighting
- Ensure the face is visible and not obscured in the images
- Check the server.log file for detailed error information

## Production Deployment

### Building the Frontend

To create a production build of the frontend:

```bash
npm run build
```

The build files will be generated in the `dist` directory.

### Running the Backend in Production

For production deployment, it's recommended to use a proper ASGI server like Uvicorn with Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app
```

## Technologies Used

- **Frontend**:
  - React.js
  - Vite
  - React Router
  - React Dropzone
  - heic2any (for HEIC image conversion)

- **Backend**:
  - FastAPI
  - Uvicorn
  - MediaPipe (for face/head/body detection)
  - OpenCV
  - Pillow/PIL
  - Python-multipart