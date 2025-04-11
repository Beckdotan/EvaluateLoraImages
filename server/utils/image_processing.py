import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import pillow_heif

async def process_uploaded_image(file):
    contents = await file.read()
    
    # Handle HEIC conversion
    if file.filename.lower().endswith('.heic') or file.content_type == 'image/heic':
        try:
            heif_file = pillow_heif.read_heif(contents)
            image = Image.frombytes(
                heif_file.mode, 
                heif_file.size, 
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
        except Exception as e:
            raise ValueError("HEIC processing requires pillow-heif with libheif binaries installed")
    else:
        image = Image.open(BytesIO(contents))
    
    # Convert to OpenCV format
    image = image.convert('RGB')
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    return opencv_image

# Note: On Windows systems, users need to install:
# 1. pip install pyheif
# 2. Download libheif DLLs from https://github.com/strukturag/libheif/releases
# 3. Add DLLs to system PATH or same directory as Python executable