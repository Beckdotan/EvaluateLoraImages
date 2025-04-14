import os
import logging
import traceback
from typing import List
import numpy as np
import cv2
from PIL import Image
import torch
from services.interfaces.ISimilarityService import ISimilarityService

# Try to import the necessary libraries
try:
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    logging.warning("facenet_pytorch not installed. Installing it with: pip install facenet-pytorch")

class ArcFaceSimilarityService(ISimilarityService):
    def __init__(self):
        """Initialize the ArcFace similarity service"""
        if not FACENET_AVAILABLE:
            raise ImportError("facenet_pytorch is required but not installed. Install with: pip install facenet-pytorch")
        
        try:
            # Initialize face recognition model (skip MTCNN since we have face crops already)
            self.model = InceptionResnetV1(pretrained='vggface2').eval()
            
            # Move to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
            logging.info(f"ArcFace similarity service initialized successfully using device: {self.device}")
        except Exception as e:
            logging.error(f"Error initializing ArcFace similarity service: {str(e)}")
            logging.error(traceback.format_exc())
            raise RuntimeError("Failed to initialize ArcFace models")

    def extract_face_embedding(self, image_path):
        """Extract face embedding from a pre-cropped face image"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            # Resize to expected input size
            img = img.resize((160, 160))
            
            # Convert to tensor and normalize
            face_tensor = torch.tensor(np.array(img)).float() / 255.0
            # Convert from [H, W, C] to [C, H, W]
            face_tensor = face_tensor.permute(2, 0, 1)
            # Add batch dimension
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.model(face_tensor).detach().cpu().numpy()[0]
                
            return embedding
            
        except Exception as e:
            logging.error(f"Error extracting face embedding from {image_path}: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    def calculate_similarity(self, reference_image_paths: List[str], generated_image_path: str) -> float:
        """
        Calculate face similarity score between reference face crops and generated face crop
        Returns a score between 0.0 and 1.0 as a standard Python float
        """
        try:
            # Extract embedding from the generated face image
            generated_embedding = self.extract_face_embedding(generated_image_path)
            
            if generated_embedding is None:
                logging.warning(f"Could not extract embedding from generated image {generated_image_path}")
                return 0.0  # Return standard Python float
            
            # Process all reference face images
            similarities = []
            valid_refs = 0
            
            for ref_path in reference_image_paths:
                ref_embedding = self.extract_face_embedding(ref_path)
                if ref_embedding is not None:
                    valid_refs += 1
                    
                    # Calculate cosine similarity
                    similarity = np.dot(generated_embedding, ref_embedding) / (
                        np.linalg.norm(generated_embedding) * np.linalg.norm(ref_embedding)
                    )
                    
                    # Convert to a score in [0, 1] as standard Python float
                    score = float((similarity + 1) / 2)  # Explicitly convert to Python float
                    similarities.append(score)
                    logging.info(f"Similarity between {os.path.basename(generated_image_path)} and {os.path.basename(ref_path)}: {score:.4f}")
            
            # Return average similarity if we have any valid comparisons
            if similarities:
                avg_similarity = float(sum(similarities) / len(similarities))  # Explicitly convert to Python float
                logging.info(f"Average similarity score: {avg_similarity:.4f} (from {valid_refs} reference faces)")
                return avg_similarity
            else:
                logging.warning("No valid reference embeddings could be extracted")
                return 0.0  # Return standard Python float
                
        except Exception as e:
            logging.error(f"Error in ArcFace similarity calculation: {str(e)}")
            logging.error(traceback.format_exc())
            return 0.0  # Return standard Python float