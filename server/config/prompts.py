# config/prompts.py
class GeminiPrompts:
    """Class to hold Gemini prompts for image analysis."""
    
    @property
    def face_analysis_prompt(self):
        """Returns the prompt for face analysis."""
        return """
        You are a professional visual facial analyst. Your task is to evaluate how closely a generated face matches a set of reference faces.

        ---

        ### Context:
        - You will receive {count} reference image(s) and 1 generated image.
        - Focus exclusively on **visible anatomical facial features**.
        - Organize your output in **Markdown format**, using clear **headings** and **bullet points**.

        ---

        ### Rules & Scope:

        **Include the following facial categories:**

        ---

        #### 1. Overall Facial Structure
        - Face shape (e.g., oval, round, square)
        - Facial proportions (forehead–midface–chin balance)
        - Forehead height and width
        - Cheekbone prominence

        ---

        #### 2. Eyes
        - Shape (almond, round, monolid, etc.)
        - Tilt (upturned, downturned)
        - Spacing (close, average, wide-set)
        - Eyelid type
        - Iris color
        - Visibility (note if eyes are partially blocked or obscured)

        ---

        #### 3. Eyebrows
        - Shape (arched, straight, curved)
        - Thickness and density
        - Distance from eyes

        ---

        #### 4. Nose
        - Bridge width and height
        - Nostril shape
        - Tip shape and orientation
        - Overall size and length

        ---

        #### 5. Lips & Mouth
        - Shape (e.g., bow, full, thin)
        - Fullness (upper and lower)
        - Width

        ---

        #### 6. Jaw, Chin & Cheeks
        - Jawline definition
        - Chin shape (e.g., pointed, round, cleft)
        - Cheek volume and contour

        ---

        #### 7. Skin
        - **Skin tone** (light, medium, dark, undertone if visible)
        - **Skin texture** (smooth, oily, visible pores, freckles, scars)
        - Permanent marks (freckles, moles, scars)

        ---

        #### 8. Hair & Facial Hair
        - Hairline shape and height
        - Sideburns, beard, mustache (if present)

        ---

        #### 9. Ears
        - Shape and size (if visible)

        ---

        #### 10. Accessories & Obstructions
        - Glasses:
          - **Reading/vision glasses**: Only mention if they obscure features
          - **Sunglasses**: Mark eye-related features as *inconclusive*
        - Obstructive accessories (e.g., hats, scarves, masks): Mark affected features as *inconclusive*
        - Do **not** comment on style or fashion

        ---

        ### 🧠 Reasoning Approach:
        As you evaluate each feature, follow this thought process:

        1. **Observe** the visible traits in the reference image(s)
        2. **Describe** the same trait in the generated image
        3. **Compare** the two (or more) by noting clear similarities and differences
        4. **Judge** the degree of resemblance (close match / partial match / different)
        5. If a feature is **not visible or obstructed**, label it as *inconclusive*

        Always ground your statements in **direct visual evidence**, and avoid assumptions. Be objective, detailed, and avoid vague or subjective language.

        ---

        ### ❌ Do NOT Comment On:
        - Lighting, shadows, exposure, image quality
        - Camera angle, head tilt, background
        - Facial expressions unless they visibly distort anatomy

        ---

        ### ✅ Output Format:
        Respond in **Markdown** using the provided feature sections.  
        Use **headings** and **bullet points** under each feature.  
        If a feature is **not visible or inconclusive**, say so explicitly.

        ---

        ### Example Output (Markdown Format):
        #### Overall Facial Structure
        - The face has a heart-shaped appearance with a pronounced chin and high cheekbones.

        #### Eyes
        - The eyes are almond-shaped with a slight upturned tilt. The spacing is close, and the iris color is hazel.

        #### Nose
        - The nose has a narrow bridge and a pointed tip, with slightly wider nostrils.

        #### Lips
        - The lips are full with a prominent upper lip, and the overall width is balanced with the face.

        #### Skin
        - The skin tone is light with a smooth texture and no visible scars or marks.

        #### Hair & Facial Hair
        - The hair is medium-length and dark brown with a straight texture. There is no visible facial hair.

        ---

        Now proceed to analyze the provided images and return your findings in **Markdown format**.
        """    
    @property
    def body_analysis_prompt(self):
        """Returns the prompt for body analysis."""
        return """
            Analyze the bodies in the provided images. The first {count} image(s) are references, and the last image is the generated one. Compare the generated body to the references, focusing on the following features:

            ---

            ### 1. Overall Body Size & Build
            - **Body size**: (e.g., thin, average, muscular, heavy, curvy, etc.)
            - **Body build**: (e.g., athletic, lean, stocky, etc.)
            - **Weight distribution**: Note general areas where weight is concentrated if visible.

            ---

            ### 2. Proportions
            - **Limb length relative to torso** (e.g., long arms/legs compared to torso size)
            - **Waist-to-hip ratio** (e.g., broader shoulders, narrow waist)
            - **Body symmetry** (balance between left and right side)

            ---

            ### 3. Limbs and Distinctive Features
            - **Number of limbs**: Note if any limbs are missing or visibly different.
            - **Distinctive body features**: (e.g., visible disabilities, muscle definition, scars, prosthetics, pregnancy)
            - **Visible marks**: (e.g., scars, birthmarks, tattoos, etc.) 

            ---

            ### 4. Hair Style, Body Hair, and Color
            - **Hair style**: (e.g., short, long, curly, straight, bald, etc.)
            - **Hair color**: (e.g., blonde, brown, black, red, gray)
            - **Hair texture**: (e.g., fine, coarse, wavy, curly, straight)
            - **Facial hair**: (e.g., beard, mustache, goatee)
            - **Body hair**: (e.g., chest hair, leg hair, arm hair, back hair, etc.)
            - **Presence and distribution**: Describe if hair is present on visible body areas.

            ---

            ### 🧠 Thought Process:
            For each category, follow this reasoning approach:
            1. **Observe** the visible traits in the reference body.
            2. **Describe** the same traits in the generated body.
            3. **Compare** the two bodies by noting similarities and differences.
            4. **Judge** the degree of resemblance (close match / partial match / different).
            5. If a feature is **not visible or obstructed**, mark it as *inconclusive*.
            6. Be **inclusive**, avoid bias, and focus on observable physical features only.
            
            Always ground your statements in **direct visual evidence**, and avoid assumptions. Be objective, detailed, and avoid vague or subjective language.

            ---

            ### ❌ Do NOT Comment On:
            - **Facial features** (as another agent is analyzing the face)
            - Lighting, shadows, exposure, image quality
            - Camera angle, head tilt, background
            - Clothing or any items that obscure the body

            ---

            ### ✅ Output Format:
            Respond in **Markdown format** with **headings** and **bullet points** for each category. If a category is **inconclusive** or the feature is **not visible**, say so explicitly.

            ---

            ### Example Output (Markdown Format):
            #### Body Size & Build
            - The body is of medium build with a moderate amount of muscle definition.

            #### Limbs & Proportions
            - The limbs appear slightly shorter in relation to the torso, and the arms are notably thinner compared to the upper body.

            #### Distinctive Features
            - No visible scars or marks were noted on the body.

            #### Hair Style, Body Hair, and Color
            - The hair is short and black, with a straight texture and no visible facial hair.
            - The body has minimal body hair, with no visible chest or arm hair.

            """
    
    @property
    def improvement_prompt(self):
        """Returns the prompt for improvement suggestions."""
        return """
        Based on the facial and body analysis provided, identify the 3-5 most critical improvements needed to make the generated image more similar to the reference image(s). For each improvement, provide specific, actionable guidance that a digital artist or painter would need to make the adjustments. Focus on concrete details like specific color adjustments, precise shape modifications, or exact proportion changes. Format each improvement point as a clear, numbered instruction that begins with an action verb. Ensure the suggestions are detailed enough for practical implementation but concise enough to be clearly understood.
        """
    
    