import mediapipe as mp
import cv2
import numpy as np
import logging
from datetime import datetime
import os

# Define directory for face crops
FACES_DIR = os.path.join('output', 'faces')

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        # Create output directory if it doesn't exist
        os.makedirs(FACES_DIR, exist_ok=True)

    def detect_faces(self, image, padding=0.0):
        """
        Detect faces in an image.
        
        Args:
            image: The input image in BGR format
            padding: Optional padding around the face as a percentage of face size (0.0 means no padding)
        
        Returns:
            List of detected faces with their coordinates and confidence scores
        """
        try:
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb_image)
            detections = []

            if results.detections:
                for detection in results.detections:
                    box = detection.location_data.relative_bounding_box
                    ih, iw = image.shape[:2]
                    
                    # Calculate coordinates
                    x = max(0, int(box.xmin * iw))
                    y = max(0, int(box.ymin * ih))
                    w = min(iw - x, int(box.width * iw))
                    h = min(ih - y, int(box.height * ih))
                    
                    # Apply padding if requested
                    if padding > 0:
                        x_pad = int(w * padding)
                        y_pad = int(h * padding)
                        x = max(0, x - x_pad)
                        y = max(0, y - y_pad)
                        w = min(iw - x, w + 2*x_pad)
                        h = min(ih - y, h + 2*y_pad)

                    if w > 0 and h > 0:
                        # Crop the face
                        face_img = image[y:y+h, x:x+w]
                        face_path = self._save_face(face_img)
                        
                        detections.append({
                            'confidence': float(detection.score[0]),
                            'coordinates': [x, y, x+w, y+h],
                            'face_path': face_path
                        })
                    else:
                        logging.warning(f"Invalid face dimensions detected: {w}x{h}")
            
            return detections
        except Exception as e:
            logging.error(f"Error in face detection: {str(e)}")
            return []

    def _save_face(self, face_img):
        """Save the face crop and return the file path"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            filename = f"face_{timestamp}.png"
            filepath = os.path.join(FACES_DIR, filename)
            cv2.imwrite(filepath, face_img)
            return filepath
        except Exception as e:
            logging.error(f"Error saving face: {str(e)}")
            return None