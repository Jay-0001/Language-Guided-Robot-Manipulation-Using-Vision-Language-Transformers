import os
import json
from PIL import Image, ImageDraw
import numpy as np

# === IMPORT YOUR FUNCTIONS ===
# from n_sphere_generation import compute_bounding_boxes, project_point
from grounding_stage import run_grounding_on_image, load_model
# ^ adjust paths as needed

###############################################
# COLOR / PHRASE MAPPING
###############################################
SPHERE_PHRASES = [
    "red sphere",
    "green sphere",
    "blue sphere",
    "yellow sphere",
    "black sphere",
    "pink sphere"
]

GT_COLOR   = (0, 255, 0)     # green
PRED_COLOR = (255, 0, 0)     # red
TEXT_COLOR = (255, 255, 0)   # yellow

###############################################
# DRAW UTILITIES
###############################################

def draw_box(draw, box, color, label=None):
    """
    box = [cx, cy, w, h] in pixel coords
    """
    if box is None:
        return

    cx, cy, w, h = box
    x0 = cx - w/2
    y0 = cy - h/2
    x1 = cx + w/2
    y1 = cy + h/2

    draw.rectangle([x0, y0, x1, y1], outline=color, width=4)

    if label:
        draw.text((x0, y0), label, fill=color)


###############################################
# VISUALIZATION ENTRY
###############################################

def visualize_sample(model, sample_dir, output_path):
    """
    sample_dir structure:
      rgb.png
      object_bounding_boxes.json
    """

    # === LOAD IMAGE AND GT ===
    rgb_path = os.path.join(sample_dir, "rgb.png")
    image = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    W, H = image.size

    gt_path = os.path.join(sample_dir, "object_bounding_boxes.json")
    with open(gt_path) as fp:
        gt_boxes = json.load(fp)

    # === FOR EACH SPHERE ===
    for sphere_idx, phrase in enumerate(SPHERE_PHRASES):
        # break if beyond number of spheres in this dataset folder
        if sphere_idx >= len(gt_boxes):
            break

        gt_box = gt_boxes[sphere_idx]

        # Run GroundingDINO prediction
        pred_box = run_grounding_on_image(model, image, phrase)

        # Draw GT box (green)
        draw_box(draw, gt_box, GT_COLOR, label=f"GT {phrase}")

        # Draw Predicted box (red)
        draw_box(draw, pred_box, PRED_COLOR, label=f"PRED {phrase}")

        # If both exist, draw center lines to visualize alignment
        if gt_box is not None and pred_box is not None:
            gcx, gcy, _, _ = gt_box
            pcx, pcy, _, _ = pred_box
            draw.line([(gcx, gcy), (pcx, pcy)], fill=TEXT_COLOR, width=2)

    # === SAVE OUTPUT ===
    image.save(output_path)
    print(f"[✓] Visualization saved → {output_path}")


###############################################
# MAIN (example usage)
###############################################
if __name__ == "__main__":
    # Load your GroundingDINO model
    CONFIG = "/home/jay/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    CHECKPOINT = "/home/jay/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    model = load_model(CONFIG, CHECKPOINT)

    # Choose sample directory manually
    SAMPLE = "/home/jay/Language Aware Manipulation/dataset/dataset/spheres_3/sample_0000"
    OUT = "debug_alignment.png"

    visualize_sample(model, SAMPLE, OUT)
