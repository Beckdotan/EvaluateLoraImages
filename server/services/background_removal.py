import cv2
import numpy as np
import logging
import traceback

class BackgroundRemover:
    def __init__(self, foreground_rect_scale=0.9):
        """
        Initialize the background remover with configuration settings.
        
        Args:
            foreground_rect_scale: Scale factor for the initial foreground rectangle (0-1)
                                 Higher values include more of the image as potential foreground
        """
        self.foreground_rect_scale = foreground_rect_scale
        
    def remove_background(self, image):
        """
        Remove background from an image using GrabCut algorithm.
        
        Args:
            image: Input OpenCV image in BGR format
            
        Returns:
            Image with transparent background (RGBA)
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
            margin = (1.0 - self.foreground_rect_scale) / 2.0
            rect = (
                int(image_rgb.shape[1] * margin), 
                int(image_rgb.shape[0] * margin), 
                int(image_rgb.shape[1] * self.foreground_rect_scale), 
                int(image_rgb.shape[0] * self.foreground_rect_scale)
            )
            
            # Initialize background and foreground models
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut
            cv2.grabCut(image_rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Modify the mask for the output
            # 0 and 2 are background, 1 and 3 are foreground
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Multiply with the input image
            image_rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
            image_rgba[:, :, 3] = mask2 * 255
            
            return image_rgba
        except Exception as e:
            logging.error(f"Error in background removal: {str(e)}")
            logging.error(traceback.format_exc())
            # If the background removal fails, return the original image with alpha channel
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA) if image.shape[2] == 3 else image
            
    def remove_background_with_mask(self, image, mask):
        """
        Remove background using a pre-computed binary mask.
        
        Args:
            image: Input OpenCV image
            mask: Binary mask where 1 is foreground and 0 is background
            
        Returns:
            Image with transparent background
        """
        try:
            # Ensure mask is binary and has correct dimensions
            mask = mask.astype('uint8')
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Convert to RGBA and apply mask
            if image.shape[2] == 3:
                image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            else:
                image_rgba = image.copy()
                
            # Set alpha channel from mask
            image_rgba[:, :, 3] = mask * 255
            
            return image_rgba
        except Exception as e:
            logging.error(f"Error applying mask in background removal: {str(e)}")
            # Return original with alpha
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA) if image.shape[2] == 3 else image