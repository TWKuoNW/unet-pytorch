# 這支適用於0和1的二值化圖片轉換
import cv2
import numpy as np
import os

input_folder = "mask_origin"   # 這裡請放輸入資料夾
output_folder = "mask"  # 這裡請放輸出資料夾
count = 0

for filename in os.listdir(input_folder): # 讀取資料夾內所有檔案
    if filename.lower().endswith(('.png')):
        count = count + 1
        path = os.path.join(input_folder, filename) 
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # 讀取灰階通道
        img = np.clip(img, 0, 1) 
        inverted_img = img # 1 - img 是轉換功能
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, inverted_img)
        if count % 100 == 0:
            print(f"已處理 {count} 張圖片...")

print("處理完成~~")