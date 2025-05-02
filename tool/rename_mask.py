import os

# 設定資料夾路徑
folder_path = 'VOCdevkit/VOC2007/SegmentationClass'

# 取得資料夾內所有以 pm_ 開頭的檔案
for filename in os.listdir(folder_path):
    if filename.startswith('pm_'):
        old_path = os.path.join(folder_path, filename)
        new_filename = filename[3:]  # 去掉 'pm_'
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {new_filename}')
