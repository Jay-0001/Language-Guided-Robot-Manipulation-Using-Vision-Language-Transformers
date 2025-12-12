import torch
import re
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------
# Load BLIP models
# --------------------------------------------

caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(device)

qa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
qa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
).to(device)


# --------------------------------------------
# Caption Generation
# --------------------------------------------

def generate_caption(image_path, max_tokens=50):
    image = Image.open(image_path).convert("RGB")
    inputs = caption_processor(image, return_tensors="pt").to(device)
    
    output = caption_model.generate(**inputs, max_new_tokens=max_tokens)
    caption = caption_processor.decode(output[0], skip_special_tokens=True)

    return caption


# --------------------------------------------
# Structured Representation Extraction
# --------------------------------------------

def extract_structure(caption_text):
    """
    Extracts:
        - objects (noun phrases)
        - attributes (colors / descriptors)
        - spatial relations (left, right, behind)
        - possible actions (verbs)
    Uses lightweight NLP rules for now.
    """

    caption = caption_text.lower()

    # Common color terms
    colors = [
        "red", "green", "blue", "yellow", "black", "white",
        "orange", "pink", "purple", "brown", "gray"
    ]

    # Spatial relations
    relations = [
        "left", "right", "front", "behind", "on top of",
        "near", "beside", "next to", "in front of"
    ]

    # Detect noun phrases (very rough heuristic)
    objects = re.findall(r"\b[a-zA-Z]+(?:\s[a-zA-Z]+)?(?=\s(?:on|in|near|next|with|by|under|over|behind|and|,|\.))", caption)

    # Attribute extraction
    attributes = [c for c in colors if c in caption]

    # Relations detection
    detected_relations = [r for r in relations if r in caption]

    # Verb / action detection
    verbs = re.findall(r"\b[a-zA-Z]+ing\b", caption)  # simple gerund match

    return {
        "objects": list(set(objects)),
        "attributes": list(set(attributes)),
        "relations": list(set(detected_relations)),
        "actions": list(set(verbs))
    }


# --------------------------------------------
# Vision-Language Parsing (Caption + Structure)
# --------------------------------------------

def parse_vlp(image_path):
    caption = generate_caption(image_path)
    structure = extract_structure(caption)

    return {
        "caption": caption,
        "structure": structure
    }


# --------------------------------------------
# Optional: BLIP Question Answering Interface
# --------------------------------------------

def ask_blip(image_path, question):
    """
    Ask BLIP a natural-language question about the image.
    """
    image = Image.open(image_path).convert("RGB")

    inputs = qa_processor(image, question, return_tensors="pt").to(device)
    output = qa_model.generate(**inputs, max_new_tokens=20)

    return qa_processor.decode(output[0], skip_special_tokens=True)


# --------------------------------------------
# Test block
# --------------------------------------------

if __name__ == "__main__":
    img = "/home/jay/Language Aware Manipulation/dataset/dataset/spheres_3/sample_0000/rgb.png"  
    query = "describe this simulation scene containing robot arm and spheres"
    print("\n=== FULL PARSE ===")
    print(parse_vlp(img))
