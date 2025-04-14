from abc import ABC, abstractmethod
from typing import Dict, Any

class SimilarityService(ABC):
    @abstractmethod
    def calculate_similarity(self, image_path1: str, image_path2: str) -> Dict[str, Any]:
        """
        Calculate similarity between two images.
        
        Args:
            image_path1: Path to the first image
            image_path2: Path to the second image
            
        Returns:
            Dict containing similarity scores
        """
        pass