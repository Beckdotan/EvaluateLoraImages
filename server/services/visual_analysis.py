# services/visual_analysis.py
import requests
import base64
import os
import logging
from typing import List
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables (especially HF_API_TOKEN)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Define the model and API endpoint
# Check the HF Hub for the specific model ID you want to use
# ViP-LLaVA 7B: https://huggingface.co/llava-hf/vip-llava-7b-hf
MODEL_ID = "llava-hf/vip-llava-7b-hf"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"


class VisualAnalysisService:
    def __init__(self):
        """
        Initializes the Visual Analysis Service to use the Hugging Face Inference API.
        """
        log.info(f"Initializing VisualAnalysisService to use HF Inference API for model: {MODEL_ID}")
        self.api_token = os.getenv("HF_API_TOKEN")
        if not self.api_token:
            log.error("Hugging Face API token (HF_API_TOKEN) not found in environment variables.")
            # Depending on requirements, you could raise an error here to prevent startup
            # raise ValueError("HF_API_TOKEN environment variable not set.")
            # Or allow startup but log a warning, requests will fail later.
            log.warning("Service initialized without API token. Analysis calls will fail.")
        self.headers = {"Authorization": f"Bearer {self.api_token}"}

    def _prepare_image_payload(self, image_path: str) -> str:
        """Reads an image file and encodes it to base64."""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except FileNotFoundError:
            log.error(f"Image file not found at path: {image_path}")
            raise
        except Exception as e:
            log.error(f"Error reading or encoding image {image_path}: {e}")
            raise

    def _generate_response_api(self, prompt: str, image_paths: List[str]) -> str:
        """
        Generates a response by calling the Hugging Face Inference API.

        Args:
            prompt (str): The textual prompt for the model.
            image_paths (List[str]): A list of paths to the images to be analyzed.

        Returns:
            str: The generated textual response from the model API.

        Note: The exact payload structure for multi-image inputs can vary between models
              and API versions. This implementation uses a common pattern but might need
              adjustment based on the specific model's API requirements on HF.
              The current ViP-LLaVA API endpoint might primarily support single-image
              visual question answering directly. Sending multiple images might require
              a specific format or might not be fully supported via the simplest API task.

              This implementation sends the prompt and expects the model to understand
              the placeholders based on its training, sending images as separate data
              (which might not be standard for simple /models/{model_id} endpoint).
              A more robust way might involve specific task endpoints if available or
              client libraries that handle multi-modal input formatting.

              Let's try a payload structure common for VQA/Image-to-Text.
              We'll send the first image and the prompt. For multi-image comparison,
              the prompt needs to be *very* specific, and this approach might be limited.

              *** Update: A better approach for multi-image might be needed. ***
              Let's try sending *only the first image* and a modified prompt asking
              it to imagine the comparison, acknowledging this is a limitation.
              Or, we can try sending *all* images as base64, but the generic endpoint
              might not process them all as intended with the text prompt.

              Let's structure it for ONE image VQA first, then discuss multi-image.
        """
        if not self.api_token:
            log.error("Cannot make API call: Hugging Face API token is missing.")
            return "Error: Service is not configured with an API token."
        if not image_paths:
             log.error("No image paths provided for API call.")
             return "Error: No images provided for analysis."

        # --- Strategy for Multi-Image Comparison via API ---
        # The standard HF inference API for models like LLaVA often expects ONE image
        # for tasks like visual-question-answering or image-to-text.
        # Sending multiple images directly with a text prompt might not work as expected
        # via the generic endpoint.
        #
        # Option A (Simplest, but limited): Send only the *generated* image and ask the model
        #            to compare it based on the description of reference features in the prompt.
        #            (Less accurate as model doesn't see references).
        # Option B (Attempt Multi-modal Payload): Try sending all images base64 encoded.
        #            This relies on the endpoint supporting a non-standard payload or being
        #            smart enough to parse multiple images from the data. Risky.
        # Option C (Iterative Calls - Complex): Make separate calls for each reference vs generated,
        #            then synthesize results. Complex and costly.
        #
        # Let's try Option B, sending multiple images in the payload, hoping the endpoint
        # handles it or we can find the correct format. This is speculative.

        encoded_images = []
        for img_path in image_paths:
            try:
                encoded_images.append(self._prepare_image_payload(img_path))
            except Exception as e:
                log.error(f"Failed to prepare image {img_path}: {e}")
                return f"Error preparing image: {os.path.basename(img_path)}"

        # Construct the prompt for the model, assuming placeholders like <image>
        # are implicitly handled when images are provided.
        full_prompt = f"USER: {'<image> ' * len(image_paths)}{prompt} ASSISTANT:"

        # Construct the payload - **This structure is speculative for multi-image**
        # Check HF documentation or examples for the specific model if this fails.
        payload = {
            "inputs": full_prompt,
             # Parameters might be needed to control generation
             "parameters": {
                 "max_new_tokens": 512, # Adjust as needed
                 "do_sample": False,
             },
             # Attempt to send image data (structure might vary)
             # Option 1: Nested data (less common for HF API?)
             # "data": { "images": encoded_images }
             # Option 2: Top-level images key (some APIs use this)
             # "images": encoded_images
             # Option 3: Relying solely on the prompt and hoping the backend handles it (unlikely for raw API)

             # Let's stick to the basic payload and see if the model/endpoint handles it.
             # If not, we might need a more specific task or endpoint.
        }

        log.info(f"Sending request to HF API: {API_URL} for {len(image_paths)} images.")
        log.debug(f"Payload structure (images not shown): { {k: v for k, v in payload.items() if k != 'data'} }") # Avoid logging full base64

        try:
            response = requests.post(API_URL, headers=self.headers, json=payload, timeout=120) # Increased timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            log.info("Received response from HF API.")
            # log.debug(f"API Raw Response: {result}") # Careful logging potentially large responses

            # Response structure depends on the model/task.
            # For text generation, often a list with a dict containing 'generated_text'.
            if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                generated_text = result[0]['generated_text']
                # Clean the output - remove the input prompt part if it's included
                assistant_marker = "ASSISTANT:"
                assistant_response_start = generated_text.find(assistant_marker)
                if assistant_response_start != -1:
                    cleaned_output = generated_text[assistant_response_start + len(assistant_marker):].strip()
                else:
                     # Fallback if the marker isn't found (model might behave differently via API)
                     log.warning("ASSISTANT marker not found in API response. Returning text after prompt.")
                     # Try removing the original prompt text (less reliable)
                     if generated_text.startswith(full_prompt.split(assistant_marker)[0]):
                           cleaned_output = generated_text[len(full_prompt.split(assistant_marker)[0]):].strip()
                           # Further strip the potential marker if it's there
                           if cleaned_output.startswith(assistant_marker):
                               cleaned_output = cleaned_output[len(assistant_marker):].strip()
                     else:
                           cleaned_output = generated_text # Return as is if unsure

                return cleaned_output
            elif 'error' in result:
                 log.error(f"HF API returned an error: {result['error']}")
                 # Check for model loading errors
                 if "is currently loading" in result.get('error', ''):
                     estimated_time = result.get('estimated_time', 'a few minutes')
                     log.info(f"Model is loading on HF side (estimated time: {estimated_time}). Might need to retry.")
                     return f"Error: The analysis model is currently loading (estimated time: {estimated_time}). Please try again shortly."
                 return f"Error from analysis API: {result['error']}"
            else:
                log.error(f"Unexpected response format from HF API: {result}")
                return "Error: Received unexpected response format from analysis API."

        except requests.exceptions.RequestException as e:
            log.error(f"HTTP Request to HF API failed: {e}")
            return f"Error: Failed to connect to analysis service ({e.__class__.__name__})."
        except Exception as e:
            log.error(f"Error processing API response: {e}")
            log.exception("Traceback for API response processing error:")
            return "Error: Failed to process the response from the analysis service."


    def analyze_face_features(self, reference_image_paths: List[str], generated_image_path: str) -> str:
        """
        Analyzes and compares facial features using the HF Inference API.
        """
        log.info(f"Analyzing face features via API. References: {len(reference_image_paths)}, Generated: {generated_image_path}")
        all_image_paths = reference_image_paths + [generated_image_path]

        prompt = (
             f"Analyze the faces in the provided images. The first {len(reference_image_paths)} image(s) are references, "
             f"and the last image is generated. Compare the generated face to the references, focusing on: "
             "overall face shape, eye color/shape, eyebrows, nose shape/size, lip shape/fullness, "
             "jawline, chin shape, facial proportions, and skin tone. "
             "Describe similarities and differences. Assess the match."
        )

        # Disclaimer about potential multi-image limitations with the generic API endpoint
        if len(all_image_paths) > 1:
            log.warning("Attempting multi-image comparison via HF API. Success depends on endpoint capability.")

        return self._generate_response_api(prompt, all_image_paths)

    def analyze_body_features(self, reference_image_paths: List[str], generated_image_path: str) -> str:
        """
        Analyzes and compares body features using the HF Inference API.
        """
        log.info(f"Analyzing body features via API. References: {len(reference_image_paths)}, Generated: {generated_image_path}")
        all_image_paths = reference_image_paths + [generated_image_path]

        prompt = (
             f"Analyze the bodies in the provided images. The first {len(reference_image_paths)} image(s) are references, "
             f"and the last image is generated. Compare the generated body to the references, focusing on: "
             "hair style/color, overall body size/build (thin, muscular, heavy, etc.), body proportions, "
             "number of limbs, posture, and distinct characteristics (pregnancy, disabilities, muscles). "
             "Ignore clothing unless revealing shape. Describe similarities/differences. Assess the match."
        )

        # Disclaimer
        if len(all_image_paths) > 1:
            log.warning("Attempting multi-image comparison via HF API. Success depends on endpoint capability.")

        return self._generate_response_api(prompt, all_image_paths)