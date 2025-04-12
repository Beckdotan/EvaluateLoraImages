import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import logging
import traceback

async def process_uploaded_image(file, max_dimension=1024):
    """
    Process an uploaded image file, resize it to improve processing time, 
    and convert it to OpenCV format.
    
    Args:
        file: The uploaded file object
        max_dimension: Maximum width or height for the processed image (default: 1024px)
        
    Returns:
        OpenCV image in BGR format, resized if necessary
    """
    try:
        contents = await file.read()
        
        # Standard image processing - HEIC files should be converted client-side
        try:
            # Open with PIL first
            image = Image.open(BytesIO(contents))
            
            # Resize large images to improve processing time
            orig_width, orig_height = image.size
            scale_factor = 1.0
            
            # Check if image needs resizing (only downscale, never upscale)
            if orig_width > max_dimension or orig_height > max_dimension:
                # Calculate scale factor to maintain aspect ratio
                scale_factor = min(max_dimension / orig_width, max_dimension / orig_height)
                new_width = int(orig_width * scale_factor)
                new_height = int(orig_height * scale_factor)
                
                logging.info(f"Resizing image from {orig_width}x{orig_height} to {new_width}x{new_height}")
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to OpenCV format
            image = image.convert('RGB')
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Log image dimensions
            logging.info(f"Processed image: {opencv_image.shape[1]}x{opencv_image.shape[0]} (scale: {scale_factor:.2f})")
            
            return opencv_image
        except Exception as e:
            logging.error(f"Image open error: {str(e)}")
            logging.error(traceback.format_exc())
            raise ValueError(f"Could not open image file: {str(e)}. If this is a HEIC file, please convert it to JPEG or PNG before uploading.")
            
    except Exception as e:
        logging.error(f"Image processing failed: {str(e)}")
        logging.error(traceback.format_exc())
        raise