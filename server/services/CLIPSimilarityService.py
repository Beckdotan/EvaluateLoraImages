import open_clip
import torch
import logging
import traceback
from PIL import Image

class CLIPSimilarityService:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        try:
            self.model_name = model_name
            self.pretrained = pretrained
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model.to(self.device)
            logging.info(f"OpenCLIP model '{model_name}' (pretrained: {pretrained}) successfully loaded.")
        except Exception as e:
            logging.error(f"Error initializing OpenCLIP model: {str(e)}")
            logging.error(traceback.format_exc())
            raise RuntimeError("Failed to initialize OpenCLIP model.")



    def calculate_similarity(self, reference_image_paths, generated_image_path):
        """
        Calculate average similarity score using OpenCLIP.
        """
        try:
            # Load and preprocess generated image
            generated_image = Image.open(generated_image_path).convert("RGB")
            generated_inputs = self.clip_preprocess(generated_image).unsqueeze(0).to(self.device)

            # Extract features for generated image
            with torch.no_grad():
                generated_features = self.clip_model.encode_image(generated_inputs)
                generated_features /= generated_features.norm(dim=-1, keepdim=True)

            # Initialize list to store similarity scores
            similarity_scores = []

            # Compare against each reference image
            for ref_path in reference_image_paths:
                try:
                    ref_image = Image.open(ref_path).convert("RGB")
                    ref_inputs = self.clip_preprocess(ref_image).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        ref_features = self.clip_model.encode_image(ref_inputs)
                        ref_features /= ref_features.norm(dim=-1, keepdim=True)

                    # Calculate cosine similarity
                    similarity = torch.cosine_similarity(generated_features, ref_features).item()
                    similarity_scores.append(similarity)
                    logging.info(f"Similarity score for {generated_image_path} against {ref_path}: {similarity}")
                except Exception as e:
                    logging.error(f"Error processing reference image {ref_path}: {str(e)}")
                    logging.error(traceback.format_exc())

            return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0

        except Exception as e:
            logging.error(f"Error in OpenCLIP similarity calculation: {str(e)}")
            logging.error(traceback.format_exc())
            return 0.0