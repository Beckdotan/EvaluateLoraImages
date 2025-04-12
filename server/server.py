from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
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
from services.visual_analysis import VisualAnalysisService
from services.gemini_visual_analysis import GeminiVisualAnalysisService



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
face_detector = FaceDetector(min_detection_confidence=0.8)
background_remover = BackgroundRemover(foreground_rect_scale=0.95)
try:
    visual_analysis_service = VisualAnalysisService()
    # Optional: Add a check here if initialization logged an error due to missing token
    # if visual_analysis_service.api_token is None:
    #     print("WARNING: VisualAnalysisService initialized without API token. Analysis will fail.")
except Exception as e:
    logging.error(f"CRITICAL: Failed to initialize VisualAnalysisService: {e}")
    visual_analysis_service = None # Or raise the exception
    
try:
    gemini_analysis_service = GeminiVisualAnalysisService()
except Exception as e:
    log.error(f"FATAL: Failed to initialize GeminiVisualAnalysisService: {e}", exc_info=True)
    gemini_analysis_service = None # Ensure it's None if init fails

def save_image(image, prefix, directory=NO_BG_DIR):
    """Save image (including alpha channel if present) to PNG file"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(directory, filename)

    # Save PNG, preserving all 4 channels
    success = cv2.imwrite(filepath, image)
    if not success:
        logging.error("Failed to write image to disk")

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

@app.post('/analyze')
async def analyze_images(request_data: Dict[Any, Any] = Body(...)):
    logging.info("Entered analyze_images endpoint")

    # --- Check if the service is available ---
    if gemini_analysis_service is None:
         logging.error("Request received but analysis service is unavailable.")
         raise HTTPException(status_code=503, detail="Visual analysis service is not available due to initialization error.")

    try:

        reference_ids = request_data.get('reference_ids', [])
        generated_id = request_data.get('generated_id')

        # --- Validation of IDs ---
        if not isinstance(reference_ids, list) or not reference_ids:
             logging.warning("Validation failed: Missing or invalid reference_ids")
             raise HTTPException(status_code=400, detail={"message": "A non-empty list of reference_ids (filenames) is required"})
        if not generated_id or not isinstance(generated_id, str):
            logging.warning("Validation failed: Missing or invalid generated_id")
            raise HTTPException(status_code=400, detail={"message": "A valid generated_id (string filename) is required"})
        logging.info(f"Extracted Reference IDs: {reference_ids}, Generated ID: {generated_id}")

        # --- Get Full Image Paths ---
        reference_image_paths: List[str] = []
        for ref_id in reference_ids:
            # Basic security: prevent path traversal, use only filename part
            filename = os.path.basename(ref_id)
            if not filename: # Handle empty string case
                 log.warning(f"Invalid empty reference ID provided.")
                 raise HTTPException(status_code=400, detail=f"Invalid reference filename provided: {ref_id}")

            path = os.path.join(NO_BG_DIR, filename)

            if not os.path.exists(path) or not os.path.isfile(path): # Check if it's actually a file
                log.error(f"Reference image file not found or is not a file: {path} (from ID: {ref_id})")
                raise HTTPException(status_code=404, detail=f"Reference image not found on server: {filename}")
            reference_image_paths.append(path)

        # Construct and validate generated image path
        generated_filename = os.path.basename(generated_id)
        if not generated_filename: # Handle empty string case
            logging.warning(f"Invalid empty generated ID provided.")
            raise HTTPException(status_code=400, detail=f"Invalid generated filename provided: {generated_id}")

        generated_image_path = os.path.join(NO_BG_DIR, generated_filename)

        if not os.path.exists(generated_image_path) or not os.path.isfile(generated_image_path):
            logging.error(f"Generated image file not found or is not a file: {generated_image_path} (from ID: {generated_id})")
            raise HTTPException(status_code=404, detail=f"Generated image not found on server: {generated_filename}")

        logging.info(f"Constructed reference paths: {reference_image_paths}")
        logging.info(f"Constructed generated path: {generated_image_path}")

        # --- Perform Analysis using the NEW Service ---
        logging.info("Starting face analysis call using Gemini service...")
        face_analysis = gemini_analysis_service.analyze_face_features(
            reference_image_paths, generated_image_path
        )
        logging.info("Face analysis call completed.")
        logging.info(f"Face Analysis Results: {face_analysis}")

        logging.info("Starting body analysis call using Gemini service...")
        body_analysis = gemini_analysis_service.analyze_body_features(
            reference_image_paths, generated_image_path
        )
        logging.info("Body analysis call completed.")
        logging.info(f"Face Analysis Results: {body_analysis}")


        # --- Check for Errors from Analysis Service ---
        # The service methods should return strings starting with "Error:" on failure
        if face_analysis.startswith("Error:") or body_analysis.startswith("Error:"):
            logging.error(f"Analysis service returned an error. Face: '{face_analysis}', Body: '{body_analysis}'")
            # Return a generic server error, specific details are in logs
            raise HTTPException(status_code=502, detail="Analysis service failed to process the request.")

        # --- Return Successful Response ---
        return {
            "reference_ids": reference_ids,
            "generated_id": generated_id,
            "face_analysis": face_analysis,
            "body_analysis": body_analysis
        }

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions (like 400, 404, 503, 502) directly
        logging.warning(f"HTTPException caught: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        # Catch any unexpected errors during endpoint execution
        logging.error(f"Unexpected error in analysis endpoint: {str(e)}", exc_info=True) # Log traceback
        raise HTTPException(
            status_code=500,
            detail={"message": "An unexpected server error occurred during analysis."}
        )

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)