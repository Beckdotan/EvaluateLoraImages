from abc import ABC, abstractmethod
from typing import Dict, Any

class IImageQualityDetector(ABC):
    @abstractmethod
    def assess_image_quality(self, image_path: str) -> Dict[str, Any]:
        """
        Assess image quality metrics.
        
        Args:
            image_path: Path to the image to assess
            
        Returns:
            Dict containing quality metrics and scores
        """
        pass