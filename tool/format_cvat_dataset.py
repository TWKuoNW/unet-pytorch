import os
import shutil
from PIL import Image
from tqdm import tqdm

def process_dataset(folder_a, folder_b):
    """
    整理 CVAT 下載的資料集結構
    folder_a: 來源資料夾 (裡面預期包含 JPEGImages 與 SegmentationClass)
    folder_b: 目標空資料夾
    """
    # 定義要處理的分割 (如 train, test 等)
    splits = ['train', 'test']
    
    for split in splits:
        # A 資料夾內部的來源路徑
        src_jpeg_split = os.path.join(folder_a, 'JPEGImages', split)
        src_seg_split = os.path.join(folder_a, 'SegmentationClass', split)
        
        # B 資料夾內部的目標路徑
        # B/train/pic, B/train/mask 等等
        dst_pic_dir = os.path.join(folder_b, split, 'img')
        dst_mask_dir = os.path.join(folder_b, split, 'mask_origin')
        
        # 若來源端沒有對應的 split (例如沒有 test 資料)，就跳過
        if not os.path.exists(src_jpeg_split):
            print(f"[{split}] 找不到來源圖片資料夾 {src_jpeg_split}，跳過此部分。")
            continue
            
        print(f"\n========== 正在處理 {split} 資料 ==========")
        # 建立目標資料夾
        os.makedirs(dst_pic_dir, exist_ok=True)
        os.makedirs(dst_mask_dir, exist_ok=True)
        
        pic_basenames = []
        
        # 1. 處理 JPEGImages (pic)
        print(f"[{split}] 正在複製並重新命名圖片...")
        for root, dirs, files in os.walk(src_jpeg_split):
            for file in files:
                if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                src_pic_path = os.path.join(root, file)
                
                # 計算相對於 src_jpeg_split (如 train 資料夾) 的相對路徑
                rel_path = os.path.relpath(src_pic_path, src_jpeg_split)
                
                # 將路徑分隔符號改為底線 "_"，例如 "dir1/dir2/img.jpg" -> "dir1_dir2_img.jpg"
                rel_parts = rel_path.split(os.sep)
                new_filename = "_".join(rel_parts)
                
                dst_pic_path = os.path.join(dst_pic_dir, new_filename)
                
                # 複製圖片
                shutil.copy2(src_pic_path, dst_pic_path)
                
                # 紀錄檔名（不含副檔名），稍後用來比對 mask
                base_name = os.path.splitext(new_filename)[0]
                pic_basenames.append((base_name, dst_pic_path))
                
        # 2. 處理 SegmentationClass (mask)
        if os.path.exists(src_seg_split):
            print(f"[{split}] 正在複製並重新命名標籤(mask)...")
            for root, dirs, files in os.walk(src_seg_split):
                for file in files:
                    if not file.lower().endswith('.png'):
                        continue
                        
                    src_mask_path = os.path.join(root, file)
                    rel_path = os.path.relpath(src_mask_path, src_seg_split)
                    rel_parts = rel_path.split(os.sep)
                    new_filename = "_".join(rel_parts)
                    
                    dst_mask_path = os.path.join(dst_mask_dir, new_filename)
                    shutil.copy2(src_mask_path, dst_mask_path)
        
        # 3. 以 pic 為準，檢查並自動生成缺失的全黑 mask
        generated_count = 0
        for base_name, dst_pic_path in tqdm(pic_basenames, desc=f"檢查 {split} 缺失的 mask"):
            expected_mask_name = base_name + '.png'
            expected_mask_path = os.path.join(dst_mask_dir, expected_mask_name)
            
            # 若目標資料夾內找不到對應的 mask，自動生成一個全黑 (0) 的 PNG
            if not os.path.exists(expected_mask_path):
                try:
                    with Image.open(dst_pic_path) as img:
                        w, h = img.size
                    
                    mask = Image.new('L', (w, h), 0)
                    mask.save(expected_mask_path)
                    generated_count += 1
                except Exception as e:
                    print(f"無法讀取圖片 {dst_pic_path} 以生成 mask: {e}")
                    
        print(f"[{split}] 處理完成！共自動生成了 {generated_count} 張純黑 mask。")
    print("\n所有資料集整理完畢！")

if __name__ == "__main__":
    # =========================================================
    # 請在此處設定您的 A 與 B 資料夾路徑
    # =========================================================
    FOLDER_A = "/home/kuonw/Documents/coast/Datasets/v31_only"
    
    # 預設會在下載資料夾建立一個新的空資料夾作為 B
    FOLDER_B = "/home/kuonw/Documents/coast/Datasets/v3_1_only"
    
    print(f"來源資料夾 (A): {FOLDER_A}")
    print(f"目標資料夾 (B): {FOLDER_B}")
    
    process_dataset(FOLDER_A, FOLDER_B)
