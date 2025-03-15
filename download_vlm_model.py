from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image

MODEL_VLM_PATH = r"C:\Users\Vishnu\annabrainhac\vlm_model"

# Load VLM model
processor = AutoProcessor.from_pretrained(MODEL_VLM_PATH, trust_remote_code=True)
vlm_model = AutoModel.from_pretrained(MODEL_VLM_PATH, trust_remote_code=True)

def analyze_with_vlm(img_path):
    try:
        image = Image.open(img_path).convert("RGB")

        # Text prompt for image analysis
        prompt = "Analyze this MRI scan for abnormalities and provide a medical assessment."

        # Process inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt")

        # Forward pass
        with torch.no_grad():
            outputs = vlm_model(**inputs)

        return outputs

    except Exception as e:
        return f"Error analyzing image with BioViL-T: {e}"