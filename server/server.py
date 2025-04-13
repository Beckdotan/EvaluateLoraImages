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
import shutil
from utils.image_processing import process_uploaded_image
from services.face_detection import FaceDetector
from services.background_removal import BackgroundRemover
from services.gemini_visual_analysis import GeminiVisualAnalysisService
from services.piq_image_quality_detector import PIQImageQualityDetector
from services.CLIPSimilarityService import CLIPSimilarityService

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
face_detector = FaceDetector(min_detection_confidence=0.7)
background_remover = BackgroundRemover(foreground_rect_scale=0.95)
# Initialize the Gemini Visual Analysis service
try:
    gemini_analysis_service = GeminiVisualAnalysisService()
except Exception as e:
    logging.error(f"FATAL: Failed to initialize GeminiVisualAnalysisService: {e}", exc_info=True)
    gemini_analysis_service = None # Ensure it's None if init fails


# Initialize the Image quality detector
try:
    image_quality_detector = PIQImageQualityDetector()
    logging.info("Successfully initialized PIQImageQualityDetector")
except Exception as e:
    logging.error(f"Failed to initialize PIQImageQualityDetector: {e}", exc_info=True)
    image_quality_detector = None
    
    # Initialize the Image quality detector
try:
    clip_service = CLIPSimilarityService()
    logging.info("Successfully initialized CLIPSimilarityService")
except Exception as e:
    logging.error(f"Failed to initialize CLIPSimilarityService: {e}", exc_info=True)
    image_quality_detector = None
    
def clean_output_directories():
    """Clean all files from output directories before processing new images"""
    directories = [FACES_DIR, NO_BG_DIR, THUMBNAILS_DIR]
    for directory in directories:
        logging.info(f"Cleaning directory: {directory}")
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                logging.info(f"Deleted: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting {file_path}: {e}")

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
        
        # Clean all output directories before processing new images
        clean_output_directories()
        
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

    # --- Check if the services are available ---
    if gemini_analysis_service is None:
         logging.error("Request received but Gemini analysis service is unavailable.")
         raise HTTPException(status_code=503, detail="Visual analysis service is not available due to initialization error.")
         
    if image_quality_detector is None:
         logging.error("Request received but image quality detector is unavailable.")
         raise HTTPException(status_code=503, detail="Image quality detector is not available due to initialization error.")

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
        #logging.info(f"Extracted Reference IDs: {reference_ids}, Generated ID: {generated_id}")

        # --- Get Full Image Paths for Analysis ---
        # For background-removed images (for body analysis)
        reference_image_paths: List[str] = []
        for ref_id in reference_ids:
            # Basic security: prevent path traversal, use only filename part
            filename = os.path.basename(ref_id)
            if not filename: # Handle empty string case
                 logging.warning(f"Invalid empty reference ID provided.")
                 raise HTTPException(status_code=400, detail=f"Invalid reference filename provided: {ref_id}")

            path = os.path.join(NO_BG_DIR, filename)

            if not os.path.exists(path) or not os.path.isfile(path): # Check if it's actually a file
                logging.error(f"Reference image file not found or is not a file: {path} (from ID: {ref_id})")
                raise HTTPException(status_code=404, detail=f"Reference image not found on server: {filename}")
            reference_image_paths.append(path)

        # Construct and validate generated image path (no background)
        generated_filename = os.path.basename(generated_id)
        if not generated_filename: # Handle empty string case
            logging.warning(f"Invalid empty generated ID provided.")
            raise HTTPException(status_code=400, detail=f"Invalid generated filename provided: {generated_id}")

        generated_image_path = os.path.join(NO_BG_DIR, generated_filename)

        if not os.path.exists(generated_image_path) or not os.path.isfile(generated_image_path):
            logging.error(f"Generated image file not found or is not a file: {generated_image_path} (from ID: {generated_id})")
            raise HTTPException(status_code=404, detail=f"Generated image not found on server: {generated_filename}")

        #logging.info(f"Constructed reference paths: {reference_image_paths}")
        #logging.info(f"Constructed generated path: {generated_image_path}")

        # --- Get the full image (with background) for quality analysis ---
        # Find the corresponding thumbnail which has the full image with background
        thumbnail_prefix = "thumb_generated_"
        thumbnail_suffix = generated_filename.replace("generated_", "")
        thumbnail_path = None
        
        # Look for the matching thumbnail in the THUMBNAILS_DIR
        if os.path.exists(THUMBNAILS_DIR):
            thumbnail_files = [f for f in os.listdir(THUMBNAILS_DIR) if f.startswith(thumbnail_prefix)]
            for thumb_file in thumbnail_files:
                if thumbnail_suffix in thumb_file:
                    thumbnail_path = os.path.join(THUMBNAILS_DIR, thumb_file)
                    break
        
        # If we can't find the exact match, try to find any thumbnail that starts with "thumb_generated_"
        if not thumbnail_path and thumbnail_files:
            thumbnail_path = os.path.join(THUMBNAILS_DIR, thumbnail_files[0])
            logging.warning(f"Exact thumbnail match not found, using: {thumbnail_path}")
        
        # If no thumbnails at all, fall back to the no-bg image
        if not thumbnail_path:
            logging.warning(f"No thumbnails found for quality analysis, falling back to no-bg image: {generated_image_path}")
            thumbnail_path = generated_image_path
        else:
            logging.info(f"Using thumbnail for quality analysis: {thumbnail_path}")

        # --- Perform Image Quality Analysis using PIQ on the full image ---
        logging.info("Starting image quality analysis using PIQ metrics on full image...")
        try:
            quality_analysis = image_quality_detector.analyze_image(thumbnail_path)
            
            quality_score = quality_analysis.get("overall_score", 0)
            is_acceptable = quality_analysis.get("is_acceptable", False)
            quality_issues = quality_analysis.get("issues_summary", [])
            quality_metrics = quality_analysis.get("metrics", {})
            normalized_metrics = quality_analysis.get("normalized_metrics", {})
            quality_level = quality_analysis.get("quality_level", "unknown")
            
            logging.info(f"Image quality analysis completed. Score: {quality_score}, Quality level: {quality_level}, Acceptable: {is_acceptable}")
            if quality_issues:
                logging.info(f"Quality issues detected: {quality_issues}")
                
            # Log available metrics dynamically
            metrics_log = []
            for metric_name, value in quality_metrics.items():
                metrics_log.append(f"{metric_name.upper()}: {value}")
            
            if metrics_log:
                logging.info(f"Quality metrics: {', '.join(metrics_log)}")
                
        except Exception as e:
            logging.error(f"Error in image quality analysis: {e}", exc_info=True)
            quality_score = 0.5  # Neutral score
            is_acceptable = True  # Don't block the process due to analysis error
            quality_issues = [f"Error analyzing image quality: {str(e)}"]
            quality_metrics = {}
            normalized_metrics = {}
            quality_level = "unknown"
            # Continue with the rest of the analysis
        
        # --- Get All Face Images ---
        # Since we're cleaning the directories before each processing run,
        # we can simply get all face images from the FACES_DIR
        reference_face_paths = []
        generated_face_path = None
        
        if os.path.exists(FACES_DIR):
            face_files = sorted(os.listdir(FACES_DIR))
            face_count = len(face_files)
            
            if face_count > 0:
                # Assuming the faces are saved in the same order as the images are processed
                # And the generated image is always the last one processed
                if face_count > 1:  # At least one reference and one generated
                    reference_face_paths = [os.path.join(FACES_DIR, f) for f in face_files[:-1]]
                    generated_face_path = os.path.join(FACES_DIR, face_files[-1])
                else:  # Only one face detected (the generated one)
                    generated_face_path = os.path.join(FACES_DIR, face_files[0])
                
                logging.info(f"Found {len(reference_face_paths)} reference faces and 1 generated face")
            else:
                logging.warning("No face images found in faces directory")
        else:
            logging.warning(f"Faces directory does not exist: {FACES_DIR}")
            
        # --- Perform Face Analysis ---
        if not reference_face_paths:
            logging.warning("No reference face paths found, using error message for face analysis")
            face_analysis = "Error: No faces detected in reference images."
        elif not generated_face_path:
            logging.warning("No generated face path found, using error message for face analysis")
            face_analysis = "Error: No face detected in generated image."
        else:
            logging.info("Starting face analysis call using Gemini service...")
            # Note that we're using the regular analyze_face_features method
            # But we're providing face crops instead of full body images
            face_analysis = gemini_analysis_service.analyze_face_features(
                reference_face_paths, generated_face_path
            )
            logging.info("Face analysis call completed.")
            #logging.info(f"Face Analysis Results: {face_analysis}")

        # --- Perform Body Analysis using the full images ---
        logging.info("Starting body analysis call using Gemini service...")
        body_analysis = gemini_analysis_service.analyze_body_features(
            reference_image_paths, generated_image_path
        )
        logging.info("Body analysis call completed.")
        #logging.info(f"Body Analysis Results: {body_analysis}")

        # --- Check for Errors from Analysis Service ---
        # Ensure we don't have None values before checking startswith
        if face_analysis is None:
            logging.error("Face analysis returned None")
            face_analysis = "Error: Face analysis failed to return a result."
            
        if body_analysis is None:
            logging.error("Body analysis returned None")
            body_analysis = "Error: Body analysis failed to return a result."
        
        # Now check if both analyses failed
        if face_analysis.startswith("Error:") and body_analysis.startswith("Error:"):
            logging.error(f"Both analysis services returned errors. Face: '{face_analysis}', Body: '{body_analysis}'")
            # Return a generic server error, specific details are in logs
            raise HTTPException(status_code=502, detail="Analysis services failed to process the request.")

        # --- Generate Improvement Suggestions ---
        # Create a quality info string with detailed metrics
        # Create dynamic metrics list for the quality info
        metrics_info = []
        for metric_name, value in quality_metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.2f}" if value >= 0.01 else f"{value:.6f}"
            else:
                formatted_value = str(value)
            metrics_info.append(f"{metric_name.upper()} Score: {formatted_value}")
        
        metrics_str = "\n".join(metrics_info) if metrics_info else "No metrics available"
        
        quality_info = (
            f"Image Quality Analysis:\n"
            f"Overall Score: {quality_score:.2f}/1.00 ({['Not Acceptable', 'Acceptable'][is_acceptable]})\n"
            f"Quality Level: {quality_level}\n"
            f"{metrics_str}\n"
            f"Quality Issues: {', '.join(quality_issues) if quality_issues else 'None detected'}"
        )
        
        # Modify the analyze_improvements call to include quality analysis
        improvement_suggestions = gemini_analysis_service.analyze_improvements(
            f"{face_analysis}\n\n{quality_info}", body_analysis
        )
        logging.info("Improvement suggestions generated.")
        #logging.info(f"Service Improvement Suggestions: {improvement_suggestions}")

        clip_score = clip_service.calculate_similarity(reference_face_paths, generated_face_path)
        
        

        # --- Return Successful Response with Quality Analysis ---
        return {
            "reference_ids": reference_ids,
            "generated_id": generated_id,
            "face_analysis": face_analysis,
            "body_analysis": body_analysis,
            "clip_score": clip_score,
            "overall_score": 0.7* clip_score + 0.3*quality_score,
            "quality_analysis": {
                "score": quality_score,
                "is_acceptable": 0.7* clip_score + 0.3*quality_score > 0.75,
                "quality_level": quality_level,
                "issues": quality_issues,
                "metrics": quality_metrics,
                "normalized_metrics": normalized_metrics,
                "hand_analysis": quality_analysis.get("detailed_results", {}).get("hand_analysis", {})
            },
            "improvement_suggestions": improvement_suggestions
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