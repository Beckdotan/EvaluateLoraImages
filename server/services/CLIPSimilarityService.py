import open_clip
import torch
import logging
import traceback
from PIL import Image
from services.interfaces.ISimilarityService import ISimilarityService

class CLIPSimilarityService(ISimilarityService):
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
        Checks multiple orientations of the generated image to find the best match.
        """
        try:
            # Load generated image
            generated_image = Image.open(generated_image_path).convert("RGB")
            
            # Create rotated versions (0°, 90°, 180°, 270°)
            orientations = [
                generated_image,  # Original
                generated_image.transpose(Image.ROTATE_90),  # 90° clockwise
                generated_image.transpose(Image.ROTATE_180),  # 180°
                generated_image.transpose(Image.ROTATE_270)   # 270° clockwise
            ]
            
            # Preprocess all orientations
            generated_inputs = torch.stack([
                self.clip_preprocess(img).unsqueeze(0) for img in orientations
            ]).squeeze(1).to(self.device)
            
            # Extract features for all orientations
            with torch.no_grad():
                generated_features_all = self.clip_model.encode_image(generated_inputs)
                generated_features_all /= generated_features_all.norm(dim=-1, keepdim=True)
            
            # Initialize list to store similarity scores
            max_similarity_scores = []
            
            # Compare against each reference image
            for ref_path in reference_image_paths:
                try:
                    ref_image = Image.open(ref_path).convert("RGB")
                    ref_inputs = self.clip_preprocess(ref_image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        ref_features = self.clip_model.encode_image(ref_inputs)
                        ref_features /= ref_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate cosine similarity for each orientation and take the maximum
                    similarities = torch.cosine_similarity(
                        generated_features_all, 
                        ref_features.expand(generated_features_all.shape[0], -1)
                    )
                    max_similarity = similarities.max().item()
                    max_similarity_scores.append(max_similarity)
                    
                    logging.info(f"Max similarity score for {generated_image_path} against {ref_path}: {max_similarity}")
                except Exception as e:
                    logging.error(f"Error processing reference image {ref_path}: {str(e)}")
                    logging.error(traceback.format_exc())
            
            return sum(max_similarity_scores) / len(max_similarity_scores) if max_similarity_scores else 0.0
            
        except Exception as e:
            logging.error(f"Error in OpenCLIP similarity calculation: {str(e)}")
            logging.error(traceback.format_exc())
            return 0.0