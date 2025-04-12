import os
import logging
import numpy as np
import cv2
from PIL import Image
import requests
from io import BytesIO
import json
import mediapipe as mp
import img2score
from img2score.models import NIQE, BRISQUE, CLIPIQA
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Any, Union
import google.generativeai as genai
from dotenv import load_dotenv
from img2score.models import NIQE, BRISQUE, CLIPIQA

# Load environment variables (for Google API key)
load_dotenv()

class SimpleAIImageQualityDetector:
    """
    A simplified service that detects flaws in AI-generated images using img2score for 
    general quality assessment, MediaPipe for hand analysis, and Google Gemini for 
    visual analysis.
    """
    
    def __init__(self):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("SimpleAIImageQualityDetector")
        self.logger.info("Initializing AI Image Quality Detector")
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=10,  # Set high to detect multiple hands
            min_detection_confidence=0.5
        )
        
        # Initialize img2score models
        self.logger.info("Loading img2score quality assessment models")
        try:
            self.niqe = NIQE()
            self.brisque = BRISQUE()
            self.clipiqa = CLIPIQA()
            self.logger.info("img2score models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading img2score models: {e}")
            raise
                
        self.logger.info("AI Image Quality Detector initialized successfully")
    
    def load_image(self, image_path_or_url):
        """Load an image from a file path or URL."""
        try:
            if image_path_or_url.startswith(('http://', 'https://')):
                response = requests.get(image_path_or_url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                return cv_image, image
            else:
                # Load from file path
                cv_image = cv2.imread(image_path_or_url)
                if cv_image is None:
                    raise ValueError(f"Failed to load image from {image_path_or_url}")
                pil_image = Image.open(image_path_or_url).convert('RGB')
                return cv_image, pil_image
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            raise
    
    def assess_image_quality(self, pil_image):
        """
        Assess overall image quality using img2score.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Dictionary with quality scores and analysis
        """
        try:
            # Calculate various quality scores
            niqe_score = self.niqe(pil_image)
            brisque_score = self.brisque(pil_image)
            clipiqa_score = self.clipiqa(pil_image)
            
            # NIQE: Lower is better (typically 0-100, with good images below 15)
            # BRISQUE: Lower is better (typically 0-100, with good images below 30)
            # CLIPIQA: Higher is better (typically 0-1)
            
            # Normalize NIQE and BRISQUE to 0-1 range (inverted, so higher is better)
            norm_niqe = max(0, min(1, 1 - (niqe_score / 50)))
            norm_brisque = max(0, min(1, 1 - (brisque_score / 100)))
            
            # Calculate weighted average (quality score)
            weights = {'niqe': 0.3, 'brisque': 0.3, 'clipiqa': 0.4}
            overall_quality = (
                norm_niqe * weights['niqe'] + 
                norm_brisque * weights['brisque'] + 
                clipiqa_score * weights['clipiqa']
            )
            
            # Determine quality level
            if overall_quality >= 0.8:
                quality_level = "excellent"
            elif overall_quality >= 0.6:
                quality_level = "good"
            elif overall_quality >= 0.4:
                quality_level = "average"
            elif overall_quality >= 0.2:
                quality_level = "poor"
            else:
                quality_level = "very poor"
            
            # Check for specific issues
            issues = []
            
            # Check for blur
            if norm_niqe < 0.4:
                issues.append("Image appears blurry or has low detail")
            
            # Check for compression artifacts
            if norm_brisque < 0.4:
                issues.append("Image likely contains compression artifacts")
            
            # Check for overall aesthetic quality
            if clipiqa_score < 0.3:
                issues.append("Image has poor aesthetic quality")
            
            return {
                "overall_quality_score": overall_quality,
                "quality_level": quality_level,
                "raw_scores": {
                    "niqe": float(niqe_score),
                    "brisque": float(brisque_score),
                    "clipiqa": float(clipiqa_score)
                },
                "normalized_scores": {
                    "niqe": float(norm_niqe),
                    "brisque": float(norm_brisque),
                    "clipiqa": float(clipiqa_score)
                },
                "quality_issues": issues
            }
        except Exception as e:
            self.logger.error(f"Error in image quality assessment: {e}")
            return {
                "error": f"Failed to assess image quality: {str(e)}",
                "overall_quality_score": 0
            }
    
    def analyze_hands(self, cv_image):
        """
        Analyze hands in the image using MediaPipe Hands.
        If no hands are detected (e.g., in a selfie showing only a face),
        returns a perfect score since no hand issues can exist.
        
        Args:
            cv_image: OpenCV image in BGR format
            
        Returns:
            Dictionary with hand analysis results
        """
        try:
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb_image.shape
            
            # Process image with MediaPipe
            results = self.hands.process(rgb_image)
            
            hand_analysis = {
                "num_hands_detected": 0,
                "hands": [],
                "hand_issues": [],
                "hand_score": 1.0  # Start with perfect score
            }
            
            # If no hands are detected, return perfect score with appropriate message
            if not results.multi_hand_landmarks:
                hand_analysis["message"] = "No hands detected in the image"
                return hand_analysis
            
            # Hands are detected, analyze them
            hand_analysis["num_hands_detected"] = len(results.multi_hand_landmarks)
            
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Determine hand side (left/right)
                hand_side = handedness.classification[0].label
                
                # Collect all landmarks for this hand
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append({
                        "x": lm.x * w,
                        "y": lm.y * h,
                        "z": lm.z * w,  # z is relative to image width for MediaPipe
                        "visibility": 1.0  # MediaPipe doesn't provide visibility
                    })
                
                # Identify fingers (using MediaPipe hand landmarks indices)
                # MediaPipe numbers: thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20
                # Tip indices are: thumb=4, index=8, middle=12, ring=16, pinky=20
                finger_tips = [4, 8, 12, 16, 20]
                mcp_joints = [1, 5, 9, 13, 17]  # Base knuckles of each finger
                
                # Get wrist point as reference
                wrist = hand_landmarks.landmark[0]
                wrist_point = (wrist.x * w, wrist.y * h)
                
                # Calculate hand size (distance from wrist to middle finger MCP)
                middle_mcp = hand_landmarks.landmark[9]
                middle_mcp_point = (middle_mcp.x * w, middle_mcp.y * h)
                hand_size = self._distance(wrist_point, middle_mcp_point)
                
                # Check finger proportions
                finger_lengths = []
                finger_names = ["thumb", "index", "middle", "ring", "pinky"]
                
                finger_issues = []
                finger_anomaly_score = 0.0
                
                for i, (tip_idx, mcp_idx) in enumerate(zip(finger_tips, mcp_joints)):
                    # Get tip and base points
                    tip = hand_landmarks.landmark[tip_idx]
                    base = hand_landmarks.landmark[mcp_idx]
                    
                    # Calculate length
                    length = self._distance(
                        (tip.x * w, tip.y * h),
                        (base.x * w, base.y * h)
                    )
                    finger_lengths.append(length)
                    
                    # Check if finger is unusually long or short
                    if i > 0:  # Skip thumb as it's naturally different
                        if length > hand_size * 1.5:
                            finger_issues.append(f"{finger_names[i]} finger appears too long")
                            finger_anomaly_score += 0.2
                        elif length < hand_size * 0.2:
                            finger_issues.append(f"{finger_names[i]} finger appears too short")
                            finger_anomaly_score += 0.2
                
                # Check expected finger length relationships
                # Typically: middle > ring > index > pinky > thumb
                # But we're mostly concerned with extreme deviations
                
                # Middle should be longest (index 2)
                if len(finger_lengths) >= 5:
                    if finger_lengths[2] < finger_lengths[1] or finger_lengths[2] < finger_lengths[3]:
                        finger_issues.append("Unusual finger proportions - middle finger should be longest")
                        finger_anomaly_score += 0.15
                    
                    # Pinky should be shorter than index
                    if finger_lengths[4] > finger_lengths[1]:
                        finger_issues.append("Unusual finger proportions - pinky longer than index finger")
                        finger_anomaly_score += 0.15
                
                # Check for extra fingers by analyzing distances between fingertips
                if len(finger_lengths) >= 5:
                    tip_positions = []
                    for tip_idx in finger_tips:
                        tip = hand_landmarks.landmark[tip_idx]
                        tip_positions.append((tip.x * w, tip.y * h))
                    
                    # Check for fingertips too close together (might indicate merged fingers)
                    for i in range(len(tip_positions)):
                        for j in range(i+1, len(tip_positions)):
                            dist = self._distance(tip_positions[i], tip_positions[j])
                            # If two fingertips are very close together relative to hand size
                            if dist < hand_size * 0.1 and i != j:
                                finger_issues.append(f"Possible merged fingers detected between {finger_names[i]} and {finger_names[j]}")
                                finger_anomaly_score += 0.3
                                break
                
                # Cap anomaly score at 1.0
                finger_anomaly_score = min(1.0, finger_anomaly_score)
                
                # Add this hand's information
                hand_analysis["hands"].append({
                    "hand_idx": idx,
                    "hand_side": hand_side,
                    "finger_issues": finger_issues,
                    "finger_anomaly_score": finger_anomaly_score,
                    "landmarks": landmarks  # Include all landmarks for potential further analysis
                })
                
                # Update overall hand issues and score
                hand_analysis["hand_issues"].extend([f"Hand {idx+1} ({hand_side}): {issue}" for issue in finger_issues])
                hand_analysis["hand_score"] = min(hand_analysis["hand_score"], 1.0 - finger_anomaly_score)
            
            return hand_analysis
        except Exception as e:
            self.logger.error(f"Error in hand analysis: {e}")
            return {
                "error": f"Failed to analyze hands: {str(e)}",
                "hand_score": 0.5,  # Neutral score on error
                "num_hands_detected": 0
            }
    
    # Removed the analyze_with_gemini method
    
    def _distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def analyze_image(self, image_path_or_url):
        """
        Main method to analyze the quality of an AI-generated image.
        
        Args:
            image_path_or_url: Path to an image file or a URL
            
        Returns:
            Dictionary with comprehensive quality analysis results
        """
        start_time = time.time()
        
        try:
            # Load image
            cv_image, pil_image = self.load_image(image_path_or_url)
            
            # Run different analyses in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                quality_future = executor.submit(self.assess_image_quality, pil_image)
                hands_future = executor.submit(self.analyze_hands, cv_image)
                
                # Get results
                quality_results = quality_future.result()
                hand_results = hands_future.result()
            
            # Calculate overall score as weighted average
            # If no hands are detected, rely more on image quality scores
            if hand_results.get("num_hands_detected", 0) == 0 and "message" in hand_results:
                # No hands in the image, weight quality score higher
                overall_score = quality_results.get("overall_quality_score", 0)
                self.logger.info("No hands detected in image, relying solely on image quality score")
            else:
                # Hands detected, use weighted average
                overall_score = (
                    quality_results.get("overall_quality_score", 0) * 0.6 +
                    hand_results.get("hand_score", 0.8) * 0.4
                )
            
            # Determine if image is acceptable
            is_acceptable = overall_score >= 0.7
            
            # Compile all issues
            all_issues = quality_results.get("quality_issues", []) + hand_results.get("hand_issues", [])
            
            # Prepare final response
            response = {
                "overall_score": overall_score,
                "is_acceptable": is_acceptable,
                "processing_time_seconds": time.time() - start_time,
                "issues_summary": all_issues,
                "detailed_results": {
                    "image_quality": quality_results,
                    "hand_analysis": hand_results
                }
            }
            
            return response
        except Exception as e:
            self.logger.error(f"Error analyzing image: {e}")
            return {
                "error": f"Failed to analyze image: {str(e)}",
                "is_acceptable": False,
                "processing_time_seconds": time.time() - start_time
            }


# Missing import for regex
import re

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Image Quality Detector")
    parser.add_argument("--image", type=str, required=True, help="Path to image file or URL")
    parser.add_argument("--output", type=str, help="Optional path to save results as JSON")
    parser.add_argument("--no-gemini", action="store_true", help="Disable Google Gemini analysis")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = SimpleAIImageQualityDetector(use_gemini=not args.no_gemini)
    
    # Analyze image
    results = detector.analyze_image(args.image)
    
    # Print summary results
    print("\n==== AI Image Quality Analysis Results ====")
    print(f"Overall Score: {results['overall_score']:.2f}/1.00")
    print(f"Acceptable: {'Yes' if results['is_acceptable'] else 'No'}")
    print(f"Processing Time: {results['processing_time_seconds']:.2f} seconds")
    
    if results.get('issues_summary'):
        print("\nIssues Found:")
        for issue in results['issues_summary']:
            print(f"- {issue}")
    else:
        print("\nNo significant issues detected")
    
    # Save detailed results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")