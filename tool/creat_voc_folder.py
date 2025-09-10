import os

def create_voc_structure(root_dir="VOCdevkit", year="VOC2007"):
    voc_path = os.path.join(root_dir, year)
    sub_dirs = [
        "JPEGImages",
        "SegmentationClass",
        "SegmentationClassOrigin",
        os.path.join("ImageSets", "Segmentation"),
    ]
    
    for sub in sub_dirs:
        path = os.path.join(voc_path, sub)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")
    
    print("\nVOC folder structure created successfully!")

if __name__ == "__main__":
    create_voc_structure(root_dir="VOCdevkit", year="VOC2007")
