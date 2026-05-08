import os

VOCdevkit_path = 'VOCdevkit'

folders = [
    'VOC2007/JPEGImages',
    'VOC2007/SegmentationClass',
    'VOC2007/SegmentationClass_Origin',
    'VOC2007/ImageSets/Segmentation',
]

for folder in folders:
    path = os.path.join(VOCdevkit_path, folder)
    os.makedirs(path, exist_ok=True)
    print(f"已建立: {path}")

print("VOC 資料夾結構建立完成")
