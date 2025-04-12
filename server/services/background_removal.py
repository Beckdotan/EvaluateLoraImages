# services/background_removal.py

import cv2
import numpy as np
import logging
import traceback
import mediapipe as mp

class BackgroundRemover:
    def __init__(self, foreground_rect_scale=0.95):
        """
        Initialize the background remover using MediaPipe Selfie Segmentation.
        """
        self.foreground_rect_scale = foreground_rect_scale
        self.selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def remove_background(self, image):
        """
        Remove background from an image using MediaPipe Selfie Segmentation.

        Args:
            image: Input OpenCV image in BGR format

        Returns:
            Image with transparent background (RGBA)
        """
        try:
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Segment
            results = self.selfie_segmentation.process(image_rgb)

            # Create mask
            mask = results.segmentation_mask
            binary_mask = (mask > 0.5).astype(np.uint8)
            binary_mask = cv2.medianBlur(binary_mask, 5)

            # Convert BGR to RGBA correctly (OpenCV loads in BGR order)
            b, g, r = cv2.split(image)
            alpha = binary_mask * 255
            image_rgba = cv2.merge((b, g, r, alpha))  # Keeps original colors

            return image_rgba

        except Exception as e:
            logging.error(f"Error in background removal: {str(e)}")
            logging.error(traceback.format_exc())
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA) if image.shape[2] == 3 else image
