import os
import shutil
import urllib.request

# 下載模型
url = 'https://github.com/bubbliiiing/unet-pytorch/releases/download/v1.0/unet_vgg_voc.pth'
urllib.request.urlretrieve(url, 'unet_vgg_voc.pth')

# 複製檔案
shutil.copy('unet_vgg_voc.pth', 'model_data/')

# 建立資料夾（不會出錯）
os.makedirs('VOCdevkit/VOC2007/JPEGImages', exist_ok=True)
os.makedirs('VOCdevkit/VOC2007/SegmentationClass', exist_ok=True)
os.makedirs('VOCdevkit/VOC2007/SegmentationClass_Origin', exist_ok=True)
