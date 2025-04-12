from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import logging
import traceback
import os
import cv2
import numpy as np
from datetime import datetime
import base64
from utils.image_processing import process_uploaded_image
from services.face_detection import FaceDetector
from services.background_removal import BackgroundRemover

# Define output directories
OUTPUT_DIR = 'output'
FACES_DIR = os.path.join(OUTPUT_DIR, 'faces')
NO_BG_DIR = os.path.join(OUTPUT_DIR, 'no_background')
THUMBNAILS_DIR = os.path.join(OUTPUT_DIR, 'thumbnails')

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(NO_BG_DIR, exist_ok=True)
os.makedirs(THUMBNAILS_DIR, exist_ok=True)

logging.basicConfig(
    filename='server.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Initialize services
face_detector = FaceDetector(min_detection_confidence=0.9)
background_remover = BackgroundRemover(foreground_rect_scale=0.95)

def save_image(image, prefix, directory=NO_BG_DIR):
    """Save the image and return the file path"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(directory, filename)
    cv2.imwrite(filepath, image)
    return filepath

def image_to_base64(image_path):
    """Convert an image file to base64 string"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.post('/process')
async def process_images(files: List[UploadFile] = File(...)):
    try:
        if len(files) < 1:
            raise HTTPException(
                status_code=400, 
                detail={"message": "At least one image is required"}
            )
        
        logging.info(f"Processing {len(files)} files")
        
        results = []
        for i, file in enumerate(files):
            try:
                logging.info(f"Processing image {i+1}: {file.filename}")
                
                # Process the uploaded image - this now returns a resized version for processing
                # Max dimension is set to 1024px by default in process_uploaded_image
                processed_image = await process_uploaded_image(file)
                
                # Get the dimensions of the processed image to store with face coordinates
                img_height, img_width = processed_image.shape[:2]
                
                # Save processed image as thumbnail (it's already resized)
                file_prefix = "reference" if i < len(files) - 1 else "generated"
                thumb_path = save_image(processed_image, f"thumb_{file_prefix}", THUMBNAILS_DIR)
                
                # Remove the background
                no_bg_image = background_remover.remove_background(processed_image)
                
                # Save the processed image (no background)
                image_path = save_image(no_bg_image, file_prefix, NO_BG_DIR)
                
                # Detect faces and save them
                faces = face_detector.detect_faces(processed_image, padding=0.1)
                
                # Convert images to base64 for direct embedding in response
                image_base64 = image_to_base64(image_path)
                thumb_base64 = image_to_base64(thumb_path)
                
                # Add results with image dimensions
                results.append({
                    'id': i,
                    'original_filename': file.filename,
                    'type': file_prefix,
                    'image_path': image_path,
                    'image_base64': f"data:image/png;base64,{image_base64}",
                    'thumbnail_path': thumb_path,
                    'thumbnail_base64': f"data:image/png;base64,{thumb_base64}",
                    'width': img_width,
                    'height': img_height,
                    'faces': faces
                })
                
                logging.info(f"Image {i+1} processed successfully")
            except Exception as e:
                logging.error(f"Failed to process image {file.filename}: {str(e)}")
                logging.error(traceback.format_exc())
                raise HTTPException(
                    status_code=400, 
                    detail={"message": f"Error processing image {i+1}: {str(e)}"}
                )

        return {
            'message': 'Processing completed successfully',
            'results': results
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail={"message": f"Server error: {str(e)}"}
        )

@app.get('/image/{image_id}')
async def get_image(image_id: str):
    """Endpoint to serve no-background images"""
    try:
        # Security check to prevent directory traversal
        image_id = os.path.basename(image_id)
        image_path = os.path.join(NO_BG_DIR, image_id)
        
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(image_path)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error serving image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/thumbnail/{image_id}')
async def get_thumbnail(image_id: str):
    """Endpoint to serve thumbnails"""
    try:
        # Security check to prevent directory traversal
        image_id = os.path.basename(image_id)
        thumb_path = os.path.join(THUMBNAILS_DIR, image_id)
        
        if not os.path.exists(thumb_path):
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        
        return FileResponse(thumb_path)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error serving thumbnail: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/face/{face_id}')
async def get_face(face_id: str):
    """Endpoint to serve face crops"""
    try:
        # Security check to prevent directory traversal
        face_id = os.path.basename(face_id)
        face_path = os.path.join(FACES_DIR, face_id)
        
        if not os.path.exists(face_path):
            raise HTTPException(status_code=404, detail="Face image not found")
        
        return FileResponse(face_path)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error serving face crop: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)