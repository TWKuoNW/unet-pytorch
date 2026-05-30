import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm

from unet import Unet

# ---- Config ----
ROOT_DIR = "/home/kuonw/文件/Coast/seagrass"

FOLDER_DICT = {
    "seagrass_000001": "seagrass_000001",
    "seagrass_000002": "seagrass_000002",
}

OUTPUT_CSV = "coverage_results.csv"
SEAGRASS_CLASS = 1  # 0=background, 1=seagrass
# ----------------

IMG_EXTENSIONS = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')


def compute_coverage(pr_mask):
    return np.sum(pr_mask == SEAGRASS_CLASS) / pr_mask.size


def main():
    unet = Unet()
    rows = []

    for label, subfolder in FOLDER_DICT.items():
        folder_path = os.path.join(ROOT_DIR, subfolder)
        if not os.path.isdir(folder_path):
            print(f"Warning: {folder_path} not found, skipping.")
            continue

        img_names = sorted(
            f for f in os.listdir(folder_path) if f.lower().endswith(IMG_EXTENSIONS)
        )

        for img_name in tqdm(img_names, desc=label):
            img_path = os.path.join(folder_path, img_name)
            image = Image.open(img_path)
            _, pred_mask = unet.detect_image(image, return_mask=True)
            coverage = compute_coverage(pred_mask)
            rows.append({
                "filename": os.path.join(subfolder, img_name),
                "coverage": f"{coverage:.6f}",
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "coverage"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done! Saved {len(rows)} records to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
