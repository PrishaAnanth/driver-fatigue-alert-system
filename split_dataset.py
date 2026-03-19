import os
import json
import shutil

# Paths
RAW_DIR = "data/raw/Driver drowsiness detection.v3i.coco"
OUTPUT_DIR = "data/processed"

# Create output dirs
for cls in ["awake", "drowsy"]:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# Go through train/valid/test
for split in ["train", "valid", "test"]:
    annotations_file = os.path.join(RAW_DIR, split, "_annotations.coco.json")
    images_dir = os.path.join(RAW_DIR, split)

    if not os.path.exists(annotations_file):
        print(f"❌ No annotation file found in {split}")
        continue

    # Load annotations
    with open(annotations_file, "r") as f:
        coco = json.load(f)

    # Build category dict
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # Map images to categories
    image_id_to_category = {}
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        cat_name = categories[ann["category_id"]]
        image_id_to_category[image_id] = cat_name

    # Copy images into awake/drowsy folders
    for img in coco["images"]:
        file_name = img["file_name"]
        image_id = img["id"]

        if image_id not in image_id_to_category:
            continue

        label = image_id_to_category[image_id].lower()

        if "drowsy" in label:
            target_dir = os.path.join(OUTPUT_DIR, "drowsy")
        else:
            target_dir = os.path.join(OUTPUT_DIR, "awake")

        src = os.path.join(images_dir, file_name)
        dst = os.path.join(target_dir, file_name)

        if os.path.exists(src):
            shutil.copy(src, dst)

    print(f"✅ Processed {split} set.")
