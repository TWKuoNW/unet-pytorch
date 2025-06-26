import cv2
import numpy as np
import os 

input_folder = r"datasets\data\VOCdevkit\VOC2012\SegmentationClassOrigin"
output_folder = r"datasets\data\VOCdevkit\VOC2012\SegmentationClass"

os.makedirs(output_folder, exist_ok=True) # 如果資料夾不存在就建立

for filename in os.listdir(input_folder): # 讀取資料夾內所有檔案
    if filename.lower().endswith(('.png')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        binary = np.where(image==0, 0, 1).astype(np.uint8)  

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, binary)

print("處理完成~~")