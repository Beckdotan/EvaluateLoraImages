import mediapipe as mp
import cv2
import os
import logging
from datetime import datetime

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )

    def detect_faces(self, image, padding=0.2):
        results = self.detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detections = []

        if results.detections:
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                
                # Calculate coordinates with padding
                x = max(0, int(box.xmin * iw))
                y = max(0, int(box.ymin * ih))
                w = min(iw - x, int(box.width * iw))
                h = min(ih - y, int(box.height * ih))
                
                # Apply padding
                x_pad = int(w * padding)
                y_pad = int(h * padding)
                x = max(0, x - x_pad)
                y = max(0, y - y_pad)
                w = min(iw - x, w + 2*x_pad)
                h = min(ih - y, h + 2*y_pad)

                if w > 0 and h > 0:
                    detections.append({
                        'confidence': detection.score[0],
                        'coordinates': (x, y, x+w, y+h),
                        'image_path': self._save_face(image, (x, y, w, h))
                    })
                else:
                    logging.warning(f"Invalid face dimensions detected: {w}x{h}")
        
        return detections

    def _save_face(self, image, coords):
        x, y, w, h = coords
        try:
            # Validate coordinates before cropping
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid dimensions: {w}x{h}")
            
            face_img = image[y:y+h, x:x+w]
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            os.makedirs('detections/faces', exist_ok=True)
            path = f'detections/faces/face_{timestamp}.jpg'
            cv2.imwrite(path, face_img)
            return path
        except Exception as save_error:
            logging.error(f"Face save failed: {str(save_error)}")
            return "error_saving_image"