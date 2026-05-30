import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
#   修改 trainval_percent 來調整訓練與驗證集的比例
#   修改 train_percent 來調整訓練集在 trainval 中的比例 (例如 9:1)
#   
#   目前此程式庫將測試集當作驗證集使用，不獨立劃分測試集
#-------------------------------------------------------#
trainval_percent    = 1
train_percent       = 0.8
#-------------------------------------------------------#
#   指向 VOC 格式資料集的資料夾路徑
#   預設指向根目錄下的 VOCdevkit
#-------------------------------------------------------#
VOCdevkit_path      = 'VOCdevkit'

if __name__ == "__main__":
    random.seed(0)
    print("開始檢查並產生 ImageSets 中的 txt 檔案...")
    
    imgfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/JPEGImages')
    segfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')
    
    # 確保資料夾存在
    os.makedirs(segfilepath, exist_ok=True)
    os.makedirs(saveBasePath, exist_ok=True)

    # 1. 檢查圖片並自動生成缺失的 mask
    if os.path.exists(imgfilepath):
        generated_mask_count = 0
        temp_img = os.listdir(imgfilepath)
        for img in temp_img:
            if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                name = os.path.splitext(img)[0]
                mask_name = name + ".png"
                mask_path = os.path.join(segfilepath, mask_name)
                
                # 如果 img 有圖，但 mask 沒找到
                if not os.path.exists(mask_path):
                    print(f"找不到 {name} 的 mask，自動生成 mask")
                    img_path = os.path.join(imgfilepath, img)
                    try:
                        with Image.open(img_path) as image:
                            w, h = image.size
                        # 生成純 0 的 mask (8位元灰階圖)
                        mask = Image.new('L', (w, h), 0)
                        mask.save(mask_path)
                        generated_mask_count += 1
                    except Exception as e:
                        print(f"無法讀取圖片 {img} 以生成 mask: {e}")
        
        if generated_mask_count > 0:
            print(f"自動產生 {generated_mask_count} 張 mask")
    else:
        print(f"警告：找不到圖片資料夾 {imgfilepath}，跳過自動生成 mask 的步驟。")

    # 2. 獲取所有 mask 並分割數據集
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num         = len(total_seg)  
    list_index  = range(num)  
    tv          = int(num * trainval_percent)  
    tr          = int(tv * train_percent)  
    trainval    = random.sample(list_index, tv)  
    train       = random.sample(trainval, tr)  
    
    print(f"訓練與驗證集大小 (trainval size): {tv}")
    print(f"訓練集大小 (train size): {tr}")
    
    ftrainval   = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath, 'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath, 'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath, 'val.txt'), 'w')  
    
    for i in list_index:  
        name = total_seg[i][:-4] + '\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("txt 檔案產生完成。")

    # 3. 檢查資料集格式
    print("正在檢查資料集格式是否符合要求，這可能需要一點時間...")
    classes_nums = np.zeros([256], np.int64)
    for i in tqdm(list_index):
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError(f"未檢測到標籤圖片 {png_file_name}，請檢查該路徑下檔案是否存在，且副檔名是否為 png。")
        
        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print(f"標籤圖片 {name} 的 shape 為 {str(np.shape(png))}，不屬於灰階圖或 8 位元彩色圖，請仔細檢查資料集格式。")
            print("標籤圖片需要是灰階圖或 8 位元彩色圖，標籤的每個像素點的值代表該像素點所屬的類別。")

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("印出像素點的值與數量：")
    print('-' * 37)
    print(f"| {'Key':>15} | {'Value':>15} |")
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print(f"| {str(i):>15} | {str(classes_nums[i]):>15} |")
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("檢測到標籤中像素點的值僅包含 0 與 255，資料格式有誤。")
        print("二分類問題需要將標籤修改為：背景像素值為 0，目標像素值為 1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("檢測到標籤中僅包含背景像素點。如果是因為剛剛自動生成的純 0 mask 則為正常現象，否則請仔細檢查資料集格式。")

    print("JPEGImages 中的圖片應為 .jpg 等格式，SegmentationClass 中的標籤圖片應為 .png 格式。")
    print("如果格式有誤，請參考：")
    print("https://github.com/bubbliiiing/segmentation-format-fix")
