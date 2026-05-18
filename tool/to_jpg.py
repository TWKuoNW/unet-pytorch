# jpeg_to_jpg_simple.py
from pathlib import Path
from tqdm import tqdm

folder = Path("img")  # 換成你的資料夾
paths = list(folder.glob("*.jpeg")) + list(folder.glob("*.JPEG"))

def unique_jpg_path(p: Path) -> Path:
    out = p.with_suffix(".jpg")
    i = 1
    while out.exists():
        out = out.with_name(f"{out.stem}_{i}.jpg")
        i += 1
    return out

for src in tqdm(paths, desc="Renaming", unit="file"):
    dst = unique_jpg_path(src)
    src.rename(dst)  # 直接改名（同資料夾最省事）

print(f"Done. Renamed {len(paths)} files.")
