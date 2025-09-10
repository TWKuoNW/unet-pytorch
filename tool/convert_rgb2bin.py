import cv2
import numpy as np
import os
from tqdm import tqdm 

input_folder = r"mask_origin"
output_folder = r"mask"

os.makedirs(output_folder, exist_ok=True) 

files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]

for filename in tqdm(files, desc="Processing images"):
    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    binary = np.where(image == 0, 0, 1).astype(np.uint8)

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, binary)

print("處理完成~~")