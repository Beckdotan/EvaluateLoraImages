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

# Create output directories if they don't exist
os.makedirs('output/processed_images', exist_ok=True)
os.makedirs('output/thumbnails', exist_ok=True)

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

# Initialize detectors
face_detector = FaceDetector()

def remove_background(image):
    """
    Remove background from an image using a simple segmentation approach.
    Returns the image with transparent background (RGBA).
    """
    try:
        # Convert to RGB if it's not
        if image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image.copy()
        
        # Use GrabCut algorithm for background removal
        # Create a mask initialized with obvious background (0) and foreground (1)
        mask = np.zeros(image_rgb.shape[:2], np.uint8)
        
        # Set rectangle for foreground - assume subject is somewhat centered
        rect = (int(image_rgb.shape[1] * 0.05), 
                int(image_rgb.shape[0] * 0.05), 
                int(image_rgb.shape[1] * 0.9), 
                int(image_rgb.shape[0] * 0.9))
        
        # Initialize background and foreground models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut
        cv2.grabCut(image_rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Modify the mask for the output
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Multiply with the input image
        image_rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
        image_rgba[:, :, 3] = mask2 * 255
        
        return image_rgba
    except Exception as e:
        logging.error(f"Error in background removal: {str(e)}")
        logging.error(traceback.format_exc())
        # If the background removal fails, return the original image
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA) if image.shape[2] == 3 else image

def save_image(image, prefix):
    """Save the image and return the file path"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join('output/processed_images', filename)
    cv2.imwrite(filepath, image)
    return filepath

def create_thumbnail(image, filepath):
    """Create a thumbnail version of the image"""
    try:
        # Create a smaller version (thumbnail)
        max_size = 300
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = max_size, int(w * max_size / h)
        else:
            new_h, new_w = int(h * max_size / w), max_size
        
        thumbnail = cv2.resize(image, (new_w, new_h))
        
        # Save thumbnail
        thumb_filename = os.path.basename(filepath).replace('.png', '_thumb.png')
        thumb_filepath = os.path.join('output/thumbnails', thumb_filename)
        cv2.imwrite(thumb_filepath, thumbnail)
        
        return thumb_filepath
    except Exception as e:
        logging.error(f"Error creating thumbnail: {str(e)}")
        return None

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
                
                # Process the uploaded image
                original_image = await process_uploaded_image(file)
                
                # Remove the background
                processed_image = remove_background(original_image)
                
                # Save the processed image
                file_prefix = "reference" if i < len(files) - 1 else "generated"
                image_path = save_image(processed_image, file_prefix)
                
                # Create thumbnail
                thumb_path = create_thumbnail(processed_image, image_path)
                
                # Detect faces
                faces = face_detector.detect_faces(original_image, padding=0)  # No padding
                
                # Convert images to base64 for direct embedding in response
                image_base64 = image_to_base64(image_path)
                thumb_base64 = image_to_base64(thumb_path) if thumb_path else None
                
                # Add results
                results.append({
                    'id': i,
                    'original_filename': file.filename,
                    'type': file_prefix,
                    'image_path': image_path,
                    'image_base64': f"data:image/png;base64,{image_base64}",
                    'thumbnail_path': thumb_path,
                    'thumbnail_base64': f"data:image/png;base64,{thumb_base64}" if thumb_base64 else None,
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
    """Endpoint to serve saved images"""
    try:
        # Security check to prevent directory traversal
        image_id = os.path.basename(image_id)
        image_path = os.path.join('output/processed_images', image_id)
        
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
        thumb_path = os.path.join('output/thumbnails', image_id)
        
        if not os.path.exists(thumb_path):
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        
        return FileResponse(thumb_path)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error serving thumbnail: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)