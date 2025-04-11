import mediapipe as mp
import cv2
import os
from datetime import datetime

class BodyDetector:
    def __init__(self, min_detection_confidence=0.5, padding_ratio=0.15):
        self.mp_pose = mp.solutions.pose
        self.detector = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence
        )
        self.padding_ratio = padding_ratio

    def detect_body(self, image):
        results = self.detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detections = []

        if results.pose_landmarks:
            # Get all landmark coordinates
            ih, iw, _ = image.shape
            landmarks = results.pose_landmarks.landmark
            
            # Calculate bounding box
            x_coords = [lm.x * iw for lm in landmarks]
            y_coords = [lm.y * ih for lm in landmarks]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Apply padding
            width = x_max - x_min
            height = y_max - y_min
            padding = int(max(width, height) * self.padding_ratio)
            
            x_min = max(0, x_min - padding)
            x_max = min(iw, x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(ih, y_max + padding)

            detections.append({
                'coordinates': (int(x_min), int(y_min), int(x_max), int(y_max)),
                'image_path': self._save_body(image, (int(x_min), int(y_min), 
                                                    int(x_max-x_min), int(y_max-y_min)))
            })
        
        return detections

    def _save_body(self, image, coords):
        x, y, w, h = coords
        body_img = image[y:y+h, x:x+w]
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        os.makedirs('detections/bodies', exist_ok=True)
        path = f'detections/bodies/body_{timestamp}.jpg'
        cv2.imwrite(path, body_img)
        return path