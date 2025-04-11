import mediapipe as mp
import cv2
import os
from datetime import datetime

class HeadDetector:
    def __init__(self, min_detection_confidence=0.5, expansion_ratios={'top': 0.3, 'sides': 0.2, 'jaw': 0.15}):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=min_detection_confidence
        )
        self.expansion = expansion_ratios

    def detect_head(self, image):
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detections = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get boundary landmarks for hairline and jaw
                scalp_points = [face_landmarks.landmark[i] for i in [10, 338, 297, 332]]  # Forehead and temple points
                jaw_points = [face_landmarks.landmark[i] for i in [152, 148, 176, 149]]  # Jawline points
                
                # Calculate bounding box
                ih, iw, _ = image.shape
                all_points = scalp_points + jaw_points
                
                # Get min/max coordinates
                x_min = min(p.x * iw for p in all_points)
                x_max = max(p.x * iw for p in all_points)
                y_min = min(p.y * ih for p in all_points)
                y_max = max(p.y * ih for p in all_points)

                # Apply expansion ratios
                width = x_max - x_min
                height = y_max - y_min
                
                x_min = max(0, x_min - width * self.expansion['sides'])
                x_max = min(iw, x_max + width * self.expansion['sides'])
                y_min = max(0, y_min - height * self.expansion['top'])
                y_max = min(ih, y_max + height * self.expansion['jaw'])

                detections.append({
                    'coordinates': (int(x_min), int(y_min), int(x_max), int(y_max)),
                    'image_path': self._save_head(image, (int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)))
                })
        
        return detections

    def _save_head(self, image, coords):
        x, y, w, h = coords
        head_img = image[y:y+h, x:x+w]
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        os.makedirs('detections/heads', exist_ok=True)
        path = f'detections/heads/head_{timestamp}.jpg'
        cv2.imwrite(path, head_img)
        return path