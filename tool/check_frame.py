import cv2
import numpy as np
import pandas as pd

# 讀取圖片
img = cv2.imread(r'VOCdevkit\VOC2007\SegmentationClass\iso1.out0414.png')
img = cv2.resize(img, (240, 320))

# 將 BGR（OpenCV 讀圖預設順序）轉成 RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 建立儲存資料的列表（每個像素用 "R,G,B" 字串表示）
data = []

for row in img:
    row_data = []
    for pixel in row:
        rgb_str = f"{pixel[0]},{pixel[1]},{pixel[2]}"
        row_data.append(rgb_str)
    data.append(row_data)

# 建立 DataFrame 並寫入 Excel
df = pd.DataFrame(data)
df.to_excel("output.xlsx", index=False, header=False)
