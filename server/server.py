from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import traceback
from utils.image_processing import process_uploaded_image
from services.face_detection import FaceDetector
from services.head_detection import HeadDetector
from services.body_detection import BodyDetector

logging.basicConfig(
    filename='server.log',
    level=logging.INFO,  # Changed to INFO for more detailed logging
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
head_detector = HeadDetector()
body_detector = BodyDetector()

@app.post('/detect')
async def detect_regions(files: List[UploadFile] = File(...)):
    try:
        if len(files) < 2:
            raise HTTPException(
                status_code=400, 
                detail="At least one reference image and one generated image are required"
            )
        
        logging.info(f"Processing {len(files)} files")
            
        # Process all reference images first
        reference_results = []
        for i, file in enumerate(files[:-1]):  # All except last file are references
            try:
                logging.info(f"Processing reference image {i+1}: {file.filename}")
                image = await process_uploaded_image(file)
                
                face_results = face_detector.detect_faces(image)
                head_results = head_detector.detect_head(image)
                body_results = body_detector.detect_body(image)
                
                reference_results.append({
                    'face': face_results,
                    'head': head_results,
                    'body': body_results
                })
                
                logging.info(f"Reference image {i+1} processed successfully")
            except Exception as e:
                logging.error(f"Image processing failed for {file.filename}: {str(e)}")
                logging.error(traceback.format_exc())
                raise HTTPException(status_code=400, detail=f"Error processing reference image {i+1}: {str(e)}")
        
        # Process generated image (last file)
        try:
            logging.info(f"Processing generated image: {files[-1].filename}")
            generated_image = await process_uploaded_image(files[-1])
            
            generated_result = {
                'face': face_detector.detect_faces(generated_image),
                'head': head_detector.detect_head(generated_image),
                'body': body_detector.detect_body(generated_image)
            }
            
            logging.info("Generated image processed successfully")
        except Exception as e:
            logging.error(f"Failed to process generated image: {str(e)}")
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"Error processing generated image: {str(e)}")

        return {
            'reference_results': reference_results,
            'generated_result': generated_result
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logging.error(f"Unexpected error processing upload: {str(e)}")
        logging.error(traceback.format_exc())
        return {'error': 'Failed to process images. Please check server.log for details.'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)