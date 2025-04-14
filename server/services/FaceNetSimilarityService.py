import os
import logging
import traceback
from typing import List
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from services.interfaces.ISimilarityService import ISimilarityService

class FaceNetSimilarityService(ISimilarityService):
    def __init__(self):
        """Initialize the ArcFace similarity service with MTCNN for face detection and alignment"""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Load face detector and aligner
            self.mtcnn = MTCNN(image_size=160, margin=0, device=self.device, post_process=True)

            # Load face embedding model
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

            logging.info(f"ArcFace similarity service initialized successfully using device: {self.device}")
        except Exception as e:
            logging.error(f"Error initializing ArcFace similarity service: {str(e)}")
            logging.error(traceback.format_exc())
            raise RuntimeError("Failed to initialize ArcFace components")

    def extract_face_embedding(self, image_path):
        """Detect, align, and extract face embedding from an image"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')

            # Detect and align face
            aligned_face = self.mtcnn(img)

            if aligned_face is None:
                logging.warning(f"No face detected in image: {image_path}")
                return None

            # Add batch dimension
            face_tensor = aligned_face.unsqueeze(0).to(self.device)

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
        Calculate face similarity score between reference face images and a generated face image
        Returns a score between 0.0 and 1.0
        """
        try:
            generated_embedding = self.extract_face_embedding(generated_image_path)

            if generated_embedding is None:
                logging.warning(f"Could not extract embedding from generated image {generated_image_path}")
                return 0.0

            similarities = []
            valid_refs = 0

            for ref_path in reference_image_paths:
                ref_embedding = self.extract_face_embedding(ref_path)
                if ref_embedding is not None:
                    valid_refs += 1

                    # Cosine similarity
                    similarity = np.dot(generated_embedding, ref_embedding) / (
                        np.linalg.norm(generated_embedding) * np.linalg.norm(ref_embedding)
                    )
                    score = float((similarity + 1) / 2)  # Normalize to [0, 1]
                    similarities.append(score)
                    logging.info(f"Similarity between {os.path.basename(generated_image_path)} and {os.path.basename(ref_path)}: {score:.4f}")

            if similarities:
                avg_similarity = float(sum(similarities) / len(similarities))
                logging.info(f"Average similarity score: {avg_similarity:.4f} (from {valid_refs} reference faces)")
                return avg_similarity
            else:
                logging.warning("No valid reference embeddings could be extracted")
                return 0.0

        except Exception as e:
            logging.error(f"Error in ArcFace similarity calculation: {str(e)}")
            logging.error(traceback.format_exc())
            return 0.0
