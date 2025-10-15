# 這支適用於0和1的二值化圖片轉換
import cv2
import numpy as np
import os

input_folder = "mask"   # 這裡請放輸入資料夾
output_folder = "mask"  # 這裡請放輸出資料夾

for filename in os.listdir(input_folder): # 讀取資料夾內所有檔案
    if filename.lower().endswith(('.png')):
        path = os.path.join(input_folder, filename) 
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # 讀取灰階通道
        img = np.clip(img, 0, 1) 
        inverted_img = 1 - img # 1 - 0 = 1 、 1 - 1 = 0 ， 這行是轉換功能
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, inverted_img)
        print(f"已處理 {filename}")

print("處理完成~~")