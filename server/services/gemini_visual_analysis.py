# services/gemini_visual_analysis.py
import os
import logging
from typing import List, Optional, Dict, Any
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types  
from config.prompts import GeminiPrompts  # Import the prompts class

# Load environment variables (especially GOOGLE_API_KEY)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Configuration ---
# Make sure to set GOOGLE_API_KEY in your .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Choose the appropriate Gemini model with vision capabilities
MODEL_NAME = "gemini-1.5-flash-latest" # Using Flash for speed/cost balance

# Define directory for faces
FACES_DIR = os.path.join('output', 'faces')
NO_BG_DIR = os.path.join('output', 'no_background')

class GeminiVisualAnalysisService:
    def __init__(self):
        """
        Initializes the Visual Analysis Service using the Google Gemini API.
        """
        logging.info(f"Initializing GeminiVisualAnalysisService with model: {MODEL_NAME}")
        self.model: Optional[genai.GenerativeModel] = None
        self.prompts = GeminiPrompts()  # Initialize the prompts class
        
        if not GOOGLE_API_KEY:
            logging.error("Google API Key (GOOGLE_API_KEY) not found in environment variables.")
            logging.warning("Service initialized without API key. Analysis calls will fail.")
            # You might want to raise an error here depending on requirements
            # raise ValueError("GOOGLE_API_KEY environment variable not set.")
        else:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                self.model = genai.GenerativeModel(MODEL_NAME)
                logging.info("Google Generative AI client configured successfully.")
            except Exception as e:
                logging.error(f"Failed to configure Google Generative AI: {e}")
                logging.warning("Service initialized without a configured model. Analysis calls will fail.")
                # Depending on requirements, could raise error here too

        # Optional: Configure generation parameters if needed
        self.generation_config = types.GenerationConfig(
            max_output_tokens=2048, # Adjust as needed for detailed analysis
        )

        # Optional: Configure safety settings (adjust based on your content)
        self.safety_settings = {
            # Add safety settings if needed
            
        }


    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Loads an image file using PIL."""
        try:
            img = Image.open(image_path)
            return img
        except FileNotFoundError:
            log.error(f"Image file not found at path: {image_path}")
            return None
        except Exception as e:
            log.error(f"Error loading image {image_path}: {e}")
            return None

    def _generate_response_gemini(self, prompt: str, image_paths: List[str]) -> str:
        """
        Generates a response by calling the Google Gemini API.

        Args:
            prompt (str): The textual prompt for the model.
            image_paths (List[str]): A list of paths to the images to be analyzed.

        Returns:
            str: The generated textual response from the model API or an error message.
        """
        if not self.model:
            log.error("Cannot make API call: Gemini model is not initialized.")
            return "Error: Analysis service is not properly configured (Model not loaded)."
        if not image_paths:
             log.error("No image paths provided for API call.")
             return "Error: No images provided for analysis."

        # --- Load Images ---
        loaded_images = []
        for img_path in image_paths:
            img = self._load_image(img_path)
            if img is None:
                # Error already logged in _load_image
                return f"Error: Failed to load image {os.path.basename(img_path)}."
            loaded_images.append(img)

        # --- Construct Prompt for Gemini API ---
        # The API expects a list containing text and image parts.
        # Often best to put text first, then images in the order mentioned in text.
        prompt_parts = [prompt] + loaded_images

        log.info(f"Sending request to Gemini API ({MODEL_NAME}) for {len(loaded_images)} images.")

        try:
            # --- Make the API Call ---
            response = self.model.generate_content(
                prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                stream=False # Get the full response at once
            )

            # --- Process Response ---
            # Detailed check for blocked content or other issues
            if not response.candidates:
                 # This case might happen if the prompt itself is immediately blocked
                 log.warning("Gemini API returned no candidates. Prompt possibly blocked.")
                 # Try to get feedback if available
                 block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
                 block_details = getattr(response.prompt_feedback, 'block_reason_message', 'No details')
                 return f"Error: Analysis request blocked by safety filters (Reason: {block_reason}, Details: {block_details}). Please revise the prompt or images."

            # Check the finish reason of the first candidate
            first_candidate = response.candidates[0]
            finish_reason = getattr(first_candidate, 'finish_reason', None) # Keep getattr for safety

            # Use the imported 'types' module for comparison
            if finish_reason == 1: # 
                 log.info("Received successful response from Gemini API.")
                 # Accessing text content safely
                 # Check candidate.text directly first for simple non-streaming cases
                 analysis_text = ""
                 try:
                     # For non-streaming, response.text is usually the easiest way
                     analysis_text = response.text
                     
                 except ValueError as ve:
                     # Handle cases where response.text might raise ValueError (e.g., blocked)
                     log.warning(f"Could not directly access response.text: {ve}. Falling back to parts.")
                     if first_candidate.content and first_candidate.content.parts:
                         analysis_text = "".join(part.text for part in first_candidate.content.parts if hasattr(part, 'text'))
                     else:
                          log.warning("Gemini response candidate has no content or parts after failing response.text.")
                          # Re-check safety based on finish_reason, as response.text fails for SAFETY block too
                          if finish_reason == types.FinishReason.SAFETY: # <-- CHANGE HERE
                              safety_ratings = getattr(first_candidate, 'safety_ratings', [])
                              log.warning(f"Response likely blocked by safety filter. Ratings: {safety_ratings}")
                              return "Error: Analysis blocked by safety filters. The content may violate safety policies."
                          else:
                              return "Error: Received an empty or invalid analysis response."
                 except Exception as e:
                     log.error(f"Unexpected error accessing response content: {e}")
                     return "Error: Failed to extract text from analysis response."

                 return analysis_text.strip()
            else:
                 # Other reasons: RECITATION, OTHER, UNSPECIFIED
                 log.error(f"Gemini API call failed with finish reason: {finish_reason}")
                 # Try getting response text even for errors, might contain info
                 error_text = ""
                 try:
                     error_text = response.text
                 except Exception:
                     pass # Ignore if text cannot be retrieved
                 return f"Error: Analysis failed with reason: {finish_reason}. {error_text}".strip()

        except Exception as e:
            # Catch other potential API errors (network, config, etc.)
            log.error(f"Error calling Gemini API: {e}", exc_info=True) # Add exc_info for traceback in logs
            # Check if the exception is one of the specific API exceptions if you didn't catch them above
            if isinstance(e, types.BlockedPromptException):
                 return "Error: Analysis request blocked by safety filters before generation started. Please revise the prompt or images."
            if isinstance(e, types.StopCandidateException):
                 return "Error: Analysis generation stopped unexpectedly by the API."
            # Add other specific google.api_core exceptions if needed
            return f"Error: Failed to communicate with analysis service ({e.__class__.__name__})."

    def _get_face_paths_from_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Extract face image paths from the results of image processing.
        
        Args:
            results: A list of result dictionaries from image processing
            
        Returns:
            A list of paths to face crop images
        """
        face_paths = []
        for result in results:
            for face in result.get('faces', []):
                face_path = face.get('face_path')
                if face_path and os.path.exists(face_path):
                    face_paths.append(face_path)
        return face_paths
    
    def _get_best_face_for_each_image(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Get the highest confidence face for each image from the results.
        
        Args:
            results: A list of result dictionaries from image processing
            
        Returns:
            A list of paths to the best face crop image for each input image
        """
        best_faces = []
        
        for result in results:
            best_face = None
            best_confidence = -1
            
            for face in result.get('faces', []):
                confidence = face.get('confidence', 0)
                face_path = face.get('face_path')
                
                if face_path and os.path.exists(face_path) and confidence > best_confidence:
                    best_confidence = confidence
                    best_face = face_path
            
            if best_face:
                best_faces.append(best_face)
                
        return best_faces

    def analyze_face_features(self, reference_image_paths: List[str], generated_image_path: str) -> str:
        """
        Analyzes and compares facial features using the Gemini API.
        
        This method will work with either face crops or full body images,
        but it's intended to be used with face crops.
        
        Args:
            reference_image_paths: List of paths to reference face images
            generated_image_path: Path to the generated face image
            
        Returns:
            Analysis results as a string
        """
        log.info(f"Analyzing face features via Gemini. References: {len(reference_image_paths)}, Generated: {generated_image_path}")
        
        # Validate inputs
        if not reference_image_paths:
            log.error("No reference image paths provided")
            return "Error: No reference faces provided for analysis."
        
        if not generated_image_path or not os.path.exists(generated_image_path):
            log.error(f"Generated image path invalid or not found: {generated_image_path}")
            return "Error: Generated face not found."
            
        # Validate that all reference paths exist
        valid_reference_paths = []
        for path in reference_image_paths:
            if os.path.exists(path):
                valid_reference_paths.append(path)
            else:
                log.warning(f"Reference image path not found: {path}")
                
        if not valid_reference_paths:
            log.error("None of the reference image paths exist")
            return "Error: No valid reference face images found."
            
        # All inputs are valid, analyze the images
        all_image_paths = valid_reference_paths + [generated_image_path]
        
        # Construct the prompt, explaining the image order
        prompt = self.prompts.face_analysis_prompt.format(count=len(valid_reference_paths))

        # IMPORTANT: Return the result of the API call
        return self._generate_response_gemini(prompt, all_image_paths)

    def analyze_body_features(self, reference_image_paths: List[str], generated_image_path: str) -> str:
        """
        Analyzes and compares body features using the Gemini API.
        This method still uses the full-body images with background removed.
        """
        log.info(f"Analyzing body features via Gemini. References: {len(reference_image_paths)}, Generated: {generated_image_path}")
        all_image_paths = reference_image_paths + [generated_image_path]

        # Construct the prompt, explaining the image order
        prompt = self.prompts.body_analysis_prompt.format(count=len(reference_image_paths))

        return self._generate_response_gemini(prompt, all_image_paths)

    def analyze_improvements(self, face_analysis: str, body_analysis: str) -> str:
        """
        Analyzes the face and body analysis results to generate specific improvement suggestions.

        Args:
            face_analysis: The result from face analysis
            body_analysis: The result from body analysis

        Returns:
            A string containing improvement suggestions
        """
        log.info("Generating improvement suggestions based on analysis results")

        # Skip if either analysis contains an error
        if face_analysis.startswith("Error:") or body_analysis.startswith("Error:"):
            return "Error: Cannot generate improvements due to analysis errors."

        # Construct the prompt for improvements
        prompt = self.prompts.improvement_prompt

        # Combine analyses for context
        combined_analysis = f"Face Analysis:\n{face_analysis}\n\nBody Analysis:\n{body_analysis}"

        # Use new function to send the prompt and combined analysis to Gemini
        return self._generate_text_only_response(prompt, combined_analysis)

    def _generate_text_only_response(self, system_prompt: str, user_message: str) -> str:
        """
        Generates a text-only response by calling the Google Gemini API without images.

        Args:
            system_prompt (str): The system prompt/instruction for the model.
            user_message (str): The user message/content to analyze.

        Returns:
            str: The generated textual response from the model API or an error message.
        """
        if not self.model:
            log.error("Cannot make API call: Gemini model is not initialized.")
            return "Error: Analysis service is not properly configured (Model not loaded)."

        # Construct the complete prompt for Gemini API
        complete_prompt = f"{system_prompt}\n\nContent to analyze:\n{user_message}"

        log.info(f"Sending text-only request to Gemini API ({MODEL_NAME}).")

        try:
            # Make the API Call
            response = self.model.generate_content(
                complete_prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                stream=False  # Get the full response at once
            )

            # Process Response - similar logic to _generate_response_gemini
            if not response.candidates:
                log.warning("Gemini API returned no candidates. Prompt possibly blocked.")
                block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
                block_details = getattr(response.prompt_feedback, 'block_reason_message', 'No details')
                return f"Error: Analysis request blocked by safety filters (Reason: {block_reason}, Details: {block_details})."

            # Check the finish reason of the first candidate
            first_candidate = response.candidates[0]
            finish_reason = getattr(first_candidate, 'finish_reason', None)

            if finish_reason == 1:  # Successful completion
                log.info("Received successful response from Gemini API.")

                try:
                    analysis_text = response.text
                    return analysis_text
                except Exception as e:
                    log.error(f"Error accessing response text: {e}")
                    return f"Error: Failed to extract text from analysis response: {e}"

                return analysis_text.strip()
            else:
                log.error(f"Gemini API call failed with finish reason: {finish_reason}")
                error_text = ""
                try:
                    error_text = response.text
                except Exception:
                    pass
                return f"Error: Analysis failed with reason: {finish_reason}. {error_text}".strip()

        except Exception as e:
            log.error(f"Error calling Gemini API: {e}", exc_info=True)
            if isinstance(e, types.BlockedPromptException):
                return "Error: Analysis request blocked by safety filters before generation started."
            if isinstance(e, types.StopCandidateException):
                return "Error: Analysis generation stopped unexpectedly by the API."
            return f"Error: Failed to communicate with analysis service ({e.__class__.__name__})."