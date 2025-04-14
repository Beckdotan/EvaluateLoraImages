from abc import ABC, abstractmethod
from typing import Dict, Any, List

class TextualVisualAnalysisService(ABC):
           
    @abstractmethod
    def analyze_face_features(self, reference_paths: List[str], generated_path: str) -> Dict[str, Any]:
        """
        Analyze and compare facial features between reference and generated images.
        
        Args:
            reference_paths: List of paths to reference images
            generated_path: Path to generated image to analyze
            
        Returns:
            Dict containing face analysis results and comparisons
        """
        pass
        
    @abstractmethod
    def analyze_body_features(self, reference_paths: List[str], generated_path: str) -> Dict[str, Any]:
        """
        Analyze and compare body features between reference and generated images.
        
        Args:
            reference_paths: List of paths to reference images
            generated_path: Path to generated image to analyze
            
        Returns:
            Dict containing body analysis results and comparisons
        """
        pass
        
    @abstractmethod
    def analyze_improvements(self, quality_metrics: Dict[str, Any], face_analysis: Dict[str, Any], 
                           body_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate improvement suggestions based on quality metrics and analysis results.
        
        Args:
            quality_metrics: Image quality metrics from quality analysis
            face_analysis: Results from face feature analysis
            body_analysis: Results from body feature analysis
            
        Returns:
            Dict containing improvement suggestions
        """
        pass