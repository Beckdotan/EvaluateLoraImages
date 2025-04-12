# services/gemini_visual_analysis.py
import os
import logging
from typing import List, Optional
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
# Corrected Import: Import the main 'types' module
from google.generativeai import types  

# Load environment variables (especially GOOGLE_API_KEY)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Configuration ---
# Make sure to set GOOGLE_API_KEY in your .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Choose the appropriate Gemini model with vision capabilities
# Options: 'gemini-pro-vision', 'gemini-1.5-flash-latest', 'gemini-1.5-pro-latest' (check availability/pricing)
MODEL_NAME = "gemini-1.5-flash-latest" # Using Flash for speed/cost balance

class GeminiVisualAnalysisService:
    def __init__(self):
        """
        Initializes the Visual Analysis Service using the Google Gemini API.
        """
        logging.info(f"Initializing GeminiVisualAnalysisService with model: {MODEL_NAME}")
        self.model: Optional[genai.GenerativeModel] = None
        
        
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
        # Use the imported 'types' module directly
        self.generation_config = types.GenerationConfig( # <-- CHANGE HERE (optional, just consistency)
            # candidate_count=1, # Default is 1
            # stop_sequences=['...'],
            max_output_tokens=1024, # Adjust as needed for detailed analysis
            # temperature=0.7,
            # top_p=1.0,
            # top_k=None
        )

        # Optional: Configure safety settings (adjust based on your content)
        # Levels: BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE, BLOCK_LOW_AND_ABOVE
        # Use the imported 'types' module directly
        self.safety_settings = {
            # Stricter settings example:
            # types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, # <-- CHANGE HERE
            # types.HarmCategory.HARM_CATEGORY_HARASSMENT: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, # <-- CHANGE HERE
            # types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, # <-- CHANGE HERE
            # types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, # <-- CHANGE HERE
            # More permissive example (use with caution):
            # types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: types.HarmBlockThreshold.BLOCK_ONLY_HIGH, # <-- CHANGE HERE
            # types.HarmCategory.HARM_CATEGORY_HARASSMENT: types.HarmBlockThreshold.BLOCK_ONLY_HIGH, # <-- CHANGE HERE
        }


    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Loads an image file using PIL."""
        try:
            img = Image.open(image_path)
            # It's often good practice to convert to RGB if transparency isn't needed
            # or if the model expects a consistent format. Test if needed.
            # img = img.convert('RGB')
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

        # Keep specific exceptions if needed, otherwise the generic one below handles them
        # except types.BlockedPromptException as bpe: # <-- CHANGE HERE (if you keep it)
        #      log.error(f"Gemini API blocked the prompt: {bpe}")
        #      return "Error: Analysis request blocked by safety filters before generation started. Please revise the prompt or images."
        # except types.StopCandidateException as sce: # <-- CHANGE HERE (if you keep it)
        #     log.error(f"Gemini API stopped generation unexpectedly: {sce}")
        #     return "Error: Analysis generation stopped unexpectedly by the API."
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


    def analyze_face_features(self, reference_image_paths: List[str], generated_image_path: str) -> str:
        """
        Analyzes and compares facial features using the Gemini API.
        """
        log.info(f"Analyzing face features via Gemini. References: {len(reference_image_paths)}, Generated: {generated_image_path}")
        all_image_paths = reference_image_paths + [generated_image_path]

        # Construct the prompt, explaining the image order
        prompt = (
             f"Analyze the faces in the provided images. The first {len(reference_image_paths)} image(s) are references, "
             f"and the last image is the generated one. Compare the generated face to the references, focusing on these features: "
             "overall face shape, eye color and shape, eyebrow shape and thickness, nose shape and size, lip shape and fullness, "
             "jawline definition, chin shape, overall facial proportions, and apparent skin tone. "
             "Clearly describe the similarities and differences found for each feature category where possible. "
             "Conclude with an assessment of how well the generated face matches the reference(s)."
        )

        return self._generate_response_gemini(prompt, all_image_paths)

    def analyze_body_features(self, reference_image_paths: List[str], generated_image_path: str) -> str:
        """
        Analyzes and compares body features using the Gemini API.
        """
        log.info(f"Analyzing body features via Gemini. References: {len(reference_image_paths)}, Generated: {generated_image_path}")
        all_image_paths = reference_image_paths + [generated_image_path]

        # Construct the prompt, explaining the image order
        prompt = (
             f"Analyze the bodies in the provided images. The first {len(reference_image_paths)} image(s) are references, "
             f"and the last image is the generated one. Compare the generated body to the references, focusing on these features: "
             "hair style and color, overall body size and build (e.g., thin, average, muscular, heavy), body proportions (e.g., limb length relative to torso), "
             "apparent number of limbs, posture (if clearly discernible), and any highly distinct characteristics visible (e.g., significant muscle definition, pregnancy, visible disabilities affecting structure). "
             "Ignore clothing details unless they directly reveal body shape or features mentioned above. "
             "Clearly describe the similarities and differences found for each relevant feature category. "
             "Conclude with an assessment of how well the generated body matches the reference(s)."
        )

        return self._generate_response_gemini(prompt, all_image_paths)