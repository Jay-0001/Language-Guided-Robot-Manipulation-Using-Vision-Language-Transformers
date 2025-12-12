import os
import json
import numpy as np
from PIL import Image
import csv

from grounding_stage import run_grounding_on_image, load_model       # your module
from metrics import box_iou, center_error                            # existing
from collections import defaultdict

###############################################
# COLOR → PHRASE MAP
###############################################

SPHERE_PHRASES = [
    "red sphere",
    "green sphere",
    "blue sphere",
    "yellow sphere",
    "black sphere",
    "pink sphere"
]


###############################################
# HELPER FUNCTIONS
###############################################

def compute_contrast(image_np, cx, cy, w, h):
    """Estimate local contrast inside GT bbox."""
    x0 = int(max(cx - w/2, 0))
    y0 = int(max(cy - h/2, 0))
    x1 = int(min(cx + w/2, image_np.shape[1]))
    y1 = int(min(cy + h/2, image_np.shape[0]))

    patch = image_np[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.std(patch))   # high std = high contrast


def classify_failure(iou, center_err):
    """Categorize prediction error."""
    if iou >= 0.4:
        return "Good"
    elif iou < 0.1:
        return "Miss"
    elif center_err > 50:
        return "Wrong Object"
    else:
        return "Partial"


###############################################
# EVALUATION CORE
###############################################

def evaluate_folder(model, folder_path, sphere_count):
    results_path = os.path.join(folder_path, f"metrics_{sphere_count}.csv")

    # Track confusion matrix for deeper post-analysis
    confusion_matrix = defaultdict(int)

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample", 
            "color", 
            "gt_box", 
            "pred_box", 
            "IoU", 
            "center_err",
            "contrast",
            "bbox_pixel_area",
            "failure_type",
            "predicted_color"
        ])

        samples = sorted([s for s in os.listdir(folder_path) if s.startswith("sample")])

        for sample in samples:
            sample_dir = os.path.join(folder_path, sample)

            # Load image
            rgb = Image.open(os.path.join(sample_dir, "rgb.png"))
            rgb_np = np.array(rgb)     # for contrast estimation

            # Load GT boxes
            with open(os.path.join(sample_dir, "object_bounding_boxes.json")) as fp:
                gt_boxes = json.load(fp)

            # Evaluate each sphere separately
            for i in range(sphere_count):
                phrase = SPHERE_PHRASES[i]
                gt_box = gt_boxes[i]

                pred_box = run_grounding_on_image(model, rgb, phrase)

                iou = box_iou(gt_box, pred_box)
                ce  = center_error(gt_box, pred_box)

                # Visibility score (contrast)
                if gt_box is not None:
                    cx, cy, w, h = gt_box
                    contrast = compute_contrast(rgb_np, cx, cy, w, h)
                    bbox_area = w * h
                else:
                    contrast = 0.0
                    bbox_area = 0.0

                # Failure classification
                failure_type = classify_failure(iou, ce)

                # Predicted color label (or None)
                predicted_color = phrase if iou > 0.4 else "other"

                # Update confusion matrix
                confusion_matrix[(phrase, predicted_color)] += 1

                writer.writerow([
                    sample,
                    phrase,
                    gt_box,
                    pred_box,
                    round(iou, 4),
                    round(ce, 2),
                    round(contrast, 4),
                    bbox_area,
                    failure_type,
                    predicted_color
                ])

            print(f"[✓] Evaluated {sample} with {sphere_count} spheres")

    print(f"\nSaved evaluation file → {results_path}")
    print("\nColor Confusion Summary:")
    for key, val in confusion_matrix.items():
        print(f"{key}: {val}")

    # Save confusion matrix as a separate JSON for deeper analysis
    with open(os.path.join(folder_path, f"confusion_{sphere_count}.json"), "w") as fp:
        json.dump({str(k): v for k, v in confusion_matrix.items()}, fp, indent=2)


###############################################
# MASTER EVALUATION LOOP
###############################################

def evaluate_dataset(model, root):
    for N in [3,4,5,6]:
        folder = os.path.join(root, f"spheres_{N}")
        if not os.path.exists(folder):
            print(f"Skipping {folder}, not found.")
            continue

        print(f"\n=== Evaluating N = {N} spheres ===")
        evaluate_folder(model, folder, N)


if __name__ == "__main__":
    # ---- UPDATE THESE ----
    CONFIG = "/home/jay/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    CHECKPOINT = "/home/jay/GroundingDINO/weights/groundingdino_swint_ogc.pth"

    model = load_model(CONFIG, CHECKPOINT)
    root = "/home/jay/Language Aware Manipulation/dataset/dataset"
    evaluate_dataset(model, root)
