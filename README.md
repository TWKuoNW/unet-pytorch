<div align="center">

# 🌊 UNet-PyTorch — 語義分割訓練框架

**基於 PyTorch 的 U-Net 語義分割，支援 VGG16 / ResNet50 backbone**

[![Python](https://img.shields.io/badge/Python-3.8-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 📌 什麼是 U-Net？

U-Net 是一種專為影像分割設計的卷積神經網路架構，因其對稱的 encoder-decoder 結構形似字母 **U** 而得名。透過 skip connection 保留細節特徵，特別適合醫學影像、遙測影像等需要精細分割的任務。

```
Input Image
    │
    ▼
┌─────────────────────────────────────────┐
│  Encoder (Backbone: VGG16 / ResNet50)   │
│  ┌──────┐  ┌──────┐  ┌──────┐           │
│  │ Conv │→ │ Conv │→ │ Conv │→ ...       │
│  └──┬───┘  └──┬───┘  └──┬───┘           │
│     │  Skip   │  Skip   │               │
│     ▼         ▼         ▼               │
│  Decoder (Upsample + Concat)            │
│  └──────┘  └──────┘  └──────┘           │
└─────────────────────────────────────────┘
    │
    ▼
Segmentation Map (0 = background, 1 = target)
```

---

## 🗂️ 資料夾結構

```
unet-pytorch/
├── 📁 VOCdevkit/
│   └── VOC2007/
│       ├── JPEGImages/          ← 訓練用原始影像 (.jpg)
│       ├── SegmentationClass_Origin/  ← 原始 Mask (.png)
│       ├── SegmentationClass/   ← 二值化後的 Mask (.png)
│       └── ImageSets/Segmentation/   ← train/val 清單
├── 📁 img/                      ← 測試用影像
├── 📁 nets/                     ← UNet 模型定義
│   ├── unet.py
│   ├── vgg.py
│   └── resnet.py
├── 📁 utils/                    ← 工具函式
├── 📁 logs/                     ← 訓練權重 & Loss 曲線
├── 📁 tool/
│   ├── download_pth.py          ← 下載預訓練權重
│   └── rename_mask.py           ← 統一影像與 Mask 檔名
├── train.py                     ← 訓練入口
├── unet.py                      ← 預測入口
├── voc_annotation.py            ← 產生 train/val 清單
└── convert_binary_img.py        ← Mask 二值化轉換
```

---

## ⚡ 快速開始

### Step 1 — Clone 專案

```bash
git clone https://github.com/TWKuoNW/unet-pytorch.git
cd unet-pytorch
```

### Step 2 — 建立 Conda 環境

```bash
# Linux
conda env create -f environment_linux.yml
conda activate unet_env

# macOS (Apple Silicon)
conda env create -f environment.yml
conda activate unet_env
```

### Step 3 — 下載預訓練權重

```bash
python tool/download_pth.py
```

---

## 📂 準備資料集

將資料放入對應資料夾，並清空 `img/` 內的舊測試圖：

| 類型 | 放置路徑 |
|------|---------|
| 🖼️ 原始影像 | `VOCdevkit/VOC2007/JPEGImages/` |
| 🎭 Mask（原始） | `VOCdevkit/VOC2007/SegmentationClass_Origin/` |
| 🔍 測試影像 | `img/` |

> **注意：** 影像格式需為 `.jpg`，Mask 格式需為 `.png`

---

## 🔄 資料前處理流程

```
原始影像 + Mask
       │
       ▼
① rename_mask.py        統一 Image 與 Mask 的檔名
       │
       ▼
② convert_binary_img.py 將 Mask 轉成二值格式
   (有顏色 → 1, 黑色 → 0)
       │
       ▼
③ voc_annotation.py     產生 train / val 清單
       │
       ▼
   ✅ 資料準備完成
```

### Step 4 — 統一影像與 Mask 檔名

> 如果影像與 Mask 的檔名已相同，可跳過此步驟。

```bash
python tool/rename_mask.py
```

### Step 5 — Mask 二值化轉換

將 `SegmentationClass_Origin/` 的 Mask 轉換為 binary 格式，輸出至 `SegmentationClass/`：

```bash
python VOCdevkit/VOC2007/convert_seg.py
```

> **格式要求：** 背景像素值 = `0`，目標像素值 = `1`

### Step 6 — 產生訓練/驗證清單

```bash
python voc_annotation.py
```

預設以 **9:1** 比例切分 train / val。

---

## 🚀 開始訓練

```bash
python train.py
```

訓練過程中的 Loss 曲線與權重檔會自動儲存於 `logs/` 資料夾。

```
logs/
├── loss_2024_xx_xx_xx_xx_xx/
│   ├── epoch_loss_train.txt
│   └── epoch_loss_val.txt
└── best_epoch_weights.pth
```

> **Tip：** 若無 GPU，請在 `train.py` 中將 `Cuda = True` 改為 `Cuda = False`

---

## 🔮 推理預測

```bash
python unet.py
```

測試影像從 `img/` 讀取，輸出分割結果。

---

## 📊 Backbone 比較

| Backbone | 特點 | 適合場景 |
|----------|------|---------|
| **VGG16** | 結構簡單、訓練穩定 | 資料量較少 |
| **ResNet50** | 更深層、表現更好 | 資料量充足 |

---

## 🛠️ 常見問題

**Q：Mask 格式不對怎麼辦？**
> 確認像素值是否為 0 和 1，若為 0 和 255 請執行 `convert_binary_img.py` 轉換。

**Q：影像和 Mask 檔名不一致？**
> 執行 `tool/rename_mask.py` 自動對齊。

**Q：訓練沒有收斂？**
> 觀察 val loss 趨勢，若持續下降代表正在收斂；若平台化表示已收斂或需調整 learning rate。

---

<div align="center">

Made with ❤️ | [回報問題](https://github.com/TWKuoNW/unet-pytorch/issues)

</div>
