import torch
import clip
from PIL import Image
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------
# Load CLIP (ViT-L/14 recommended for robotics)
# ---------------------------------------------------
model, preprocess = clip.load("ViT-L/14", device=device)


# ---------------------------------------------------
# Candidate vocabulary for zero-shot classification
# ---------------------------------------------------
# You can expand this depending on how many affordance categories you need.
CANDIDATE_LABELS = [
    "red sphere", "green sphere", "blue sphere", "yellow sphere", "pink sphere", "black sphere",
    "sphere", "robot arm", "franka panda robot", "table", "workspace", "floor", "shadow"
]

# We generate text tokens once for efficiency.
TEXT_TOKENS = clip.tokenize(CANDIDATE_LABELS).to(device)


# ---------------------------------------------------
# Zero-shot object matching
# ---------------------------------------------------
def classify_image_clip(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(TEXT_TOKENS)

        # Normalize for cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = logits[0].cpu().numpy()

    # Return a sorted list of label:probability pairs
    ranked = sorted(
        list(zip(CANDIDATE_LABELS, probs)),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked


# ---------------------------------------------------
# Structured representation extraction
# ---------------------------------------------------
def extract_structure_from_clip(ranked_labels, top_k=5):
    """
    Turns CLIP’s ranked outputs into a structured set of:
      - detected objects
      - attributes (colors)
      - confidence scores
    """
    top = ranked_labels[:top_k]
    objects = []
    attributes = []

    colors = ["red", "green", "blue", "yellow", "pink", "black"]

    for label, score in top:
        # Record objects like "red sphere"
        objects.append({"label": label, "confidence": float(score)})

        # Extract colors
        for c in colors:
            if c in label:
                attributes.append(c)

    return {
        "objects": objects,
        "attributes": list(set(attributes)),
        "relations": [],     # CLIP cannot infer relations without spatial cues
        "actions": []        # CLIP cannot infer actions directly
    }


# ---------------------------------------------------
# Unified VLP Parse
# ---------------------------------------------------
def parse_vlp_clip(image_path, top_k=5):
    ranked = classify_image_clip(image_path)
    structure = extract_structure_from_clip(ranked, top_k=top_k)

    return {
        "clip_ranked_labels": ranked,
        "structure": structure
    }


# ---------------------------------------------------
# Test block
# ---------------------------------------------------
if __name__ == "__main__":
    img = "/home/jay/Language Aware Manipulation/dataset/dataset/spheres_3/sample_0000/rgb.png"
    
    print("\n=== CLIP ZERO-SHOT RESULTS ===")
    vlp_output = parse_vlp_clip(img, top_k=5)

    for lbl, p in vlp_output["clip_ranked_labels"][:5]:
        print(f"{lbl:20s}  {p:.4f}")

    print("\n=== STRUCTURED REPRESENTATION ===")
    print(vlp_output["structure"])
