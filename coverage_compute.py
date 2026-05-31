import os
import csv
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

from unet import Unet

# ---- Config ----
ROOT_DIR = "img"

FOLDER_DICT = {
    "": "",
}

OUTPUT_CSV = "v2_1.csv"
SEAGRASS_CLASS = 1  # 0=background, 1=seagrass
# ----------------

IMG_EXTENSIONS = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')


def compute_coverage(pr_mask):
    return np.sum(pr_mask == SEAGRASS_CLASS) / pr_mask.size


def main():
    unet = Unet()
    count = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "coverage"])
        writer.writeheader()

        for label, subfolder in FOLDER_DICT.items():
            folder_path = os.path.join(ROOT_DIR, subfolder)
            if not os.path.isdir(folder_path):
                print(f"Warning: {folder_path} not found, skipping.")
                continue

            img_names = sorted(
                fn for fn in os.listdir(folder_path) if fn.lower().endswith(IMG_EXTENSIONS)
            )

            for img_name in tqdm(img_names, desc=label):
                img_path = os.path.join(folder_path, img_name)
                try:
                    image = Image.open(img_path)
                    _, pred_mask = unet.detect_image(image, return_mask=True)
                    coverage = compute_coverage(pred_mask)
                    writer.writerow({
                        "filename": os.path.join(subfolder, img_name),
                        "coverage": f"{coverage:.6f}",
                    })
                    f.flush()
                    count += 1
                except Exception as e:
                    print(f"\nSkip {img_name}: {e}")

    print(f"Done! Saved {count} records to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
