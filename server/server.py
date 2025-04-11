from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import traceback
from utils.image_processing import process_uploaded_image
from services import face_detection, head_detection, body_detection

logging.basicConfig(
    filename='server.log',
    level=logging.ERROR,
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

@app.post('/detect')
async def detect_regions(files: List[UploadFile] = File(..., min_items=2)):
    try:
        # Validate input files
        def _validate_detection(results):
            if any(isinstance(item, str) and 'error' in item for item in results):
                raise ValueError('Invalid detection results from service')
            return results

        # Process all reference images first
        reference_results = []
        for file in files[:-1]:  # All except last file are references
            try:
                image = await process_uploaded_image(file)
                try:
                    reference_results.append({
                        'face': _validate_detection(face_detection.detect_faces(image)),
                        'head': head_detection.detect_head(image),
                        'body': body_detection.detect_body(image)
                    })
                except Exception as e:
                    logging.error(f"Detection failed: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")
            except Exception as e:
                logging.error(f"Image processing failed: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))
        
        # Process generated image (last file)
        generated_image = await process_uploaded_image(files[-1])
        generated_result = {
            'face': face_detection.detect_faces(generated_image),
            'head': head_detection.detect_head(generated_image),
            'body': body_detection.detect_body(generated_image)
        }
        
        def _validate_detection(results):
            if any(isinstance(item, str) and 'error' in item for item in results):
                raise ValueError('Invalid detection results from service')
            return results

        return {
            'reference_results': reference_results,
            'generated_result': generated_result
        }
    except Exception as e:
        logging.error(f"Error processing upload: {str(e)}\n{traceback.format_exc()}")
        def _validate_detection(results):
            if any(isinstance(item, str) and 'error' in item for item in results):
                raise ValueError('Invalid detection results from service')
            return results

        return {'error': 'Failed to process images. Please check server.log for details.'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)