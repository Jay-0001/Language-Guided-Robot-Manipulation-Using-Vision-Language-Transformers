import os
import json
import shutil
from PIL import Image

# ---------------------------------------
# GLOBALS
# ---------------------------------------
SPHERE_PHRASES = [
    "red sphere",
    "green sphere",
    "blue sphere",
    "yellow sphere",
    "black sphere",
    "pink sphere",
]

OUTPUT_ROOT = "grounding_finetune"
IMG_OUT_DIR = os.path.join(OUTPUT_ROOT, "val/images")
ANNOTATION_FILE = os.path.join(OUTPUT_ROOT, "val/annotations.json")

os.makedirs(IMG_OUT_DIR, exist_ok=True)


# ---------------------------------------
# HELPER – Convert cxcywh → normalized xywh
# ---------------------------------------
def normalize_bbox(cx, cy, w, h, W, H):
    x0 = cx - w / 2
    y0 = cy - h / 2

    return [
        max(0, x0 / W),
        max(0, y0 / H),
        min(1.0, w / W),
        min(1.0, h / H)
    ]


# ---------------------------------------
# HELPER – Build caption + token spans
# ---------------------------------------
def build_caption_and_spans(phrases):
    """
    Example output:
    caption: "red sphere. blue sphere. green sphere."
    spans: [[0, 11], [13, 24], [26, 38]]
    """
    caption = ""
    spans = []
    idx = 0

    for p in phrases:
        start = idx
        caption += p + "."
        end = len(caption)
        spans.append([start, end - 1])
        caption += " "
        idx = len(caption)

    caption = caption.strip()
    return caption, spans


# ---------------------------------------
# MAIN CONVERSION FUNCTION
# ---------------------------------------
def convert_dataset(dataset_root):
    annotations = []
    image_id_counter = 1

    # Loop over spheres_3/, spheres_4/, ...
    for folder in sorted(os.listdir(dataset_root)):
        folder_path = os.path.join(dataset_root, folder)
        if not os.path.isdir(folder_path):
            continue

        # Loop over sample_xxxx folders
        for sample in sorted(os.listdir(folder_path)):
            sample_path = os.path.join(folder_path, sample)
            if not os.path.isdir(sample_path):
                continue

            # ---------- Load rgb ----------
            rgb_path = os.path.join(sample_path, "rgb.png")
            if not os.path.exists(rgb_path):
                continue

            img = Image.open(rgb_path)
            W, H = img.size

            # Copy image to output folder
            new_name = f"{image_id_counter:04d}.png"
            out_img_path = os.path.join(IMG_OUT_DIR, new_name)
            shutil.copy(rgb_path, out_img_path)

            # ---------- Load GT boxes ----------
            bbox_path = os.path.join(sample_path, "object_bounding_boxes.json")
            with open(bbox_path) as fp:
                gt_boxes = json.load(fp)

            # Filter out None boxes
            valid_boxes = []
            valid_phrases = []

            for idx, box in enumerate(gt_boxes):
                if box is None:
                    continue
                phrase = SPHERE_PHRASES[idx]
                valid_boxes.append(box)      # [cx,cy,w,h] pixels
                valid_phrases.append(phrase) # "red sphere", ...

            # ---------- Build caption ----------
            caption, spans = build_caption_and_spans(valid_phrases)

            # ---------- Build annotations ----------
            ann_list = []

            for (box, phrase, span) in zip(valid_boxes, valid_phrases, spans):
                cx, cy, w, h = box
                norm_box = normalize_bbox(cx, cy, w, h, W, H)

                ann_list.append({
                    "bbox": norm_box,
                    "category_name": phrase,
                    "token_span": [span]
                })

            annotations.append({
                "image_id": image_id_counter,
                "file_name": new_name,
                "height": H,
                "width": W,
                "caption": caption,
                "annotations": ann_list
            })

            image_id_counter += 1

    # Save JSON
    with open(ANNOTATION_FILE, "w") as fp:
        json.dump(annotations, fp, indent=2)

    print(f"[✓] Dataset conversion complete.")
    print(f"[✓] Output saved in: {OUTPUT_ROOT}")
    print(f"[✓] Total images: {image_id_counter - 1}")


# ---------------------------------------
# RUN
# ---------------------------------------
if __name__ == "__main__":
    root="/home/jay/Language Aware Manipulation/dataset/valdataset"
    convert_dataset(root)
