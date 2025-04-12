import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import logging
import traceback

async def process_uploaded_image(file):
    """
    Process an uploaded image file and convert it to OpenCV format.
    
    Args:
        file: The uploaded file object
        
    Returns:
        OpenCV image in BGR format
    """
    try:
        contents = await file.read()
        
        # Standard image processing - HEIC files should be converted client-side
        try:
            image = Image.open(BytesIO(contents))
            # Convert to OpenCV format
            image = image.convert('RGB')
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return opencv_image
        except Exception as e:
            logging.error(f"Image open error: {str(e)}")
            logging.error(traceback.format_exc())
            raise ValueError(f"Could not open image file: {str(e)}. If this is a HEIC file, please convert it to JPEG or PNG before uploading.")
            
    except Exception as e:
        logging.error(f"Image processing failed: {str(e)}")
        logging.error(traceback.format_exc())
        raise