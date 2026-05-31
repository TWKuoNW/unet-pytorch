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

## 📈 模型表現比較 (Model Performance)

以下為近期基於 ResNet50 的各版本模型在驗證/測試集上的表現：

| Model | mRecall | mPrecision | mAcc | mFallout | mIoU | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v2_1** | 0.7550 | 0.7354 | 0.7332 | 0.2450 | 0.5742 | 0.7451 |
| **v3_0_0** | 0.8128 | 0.8683 | 0.8532 | 0.1872 | 0.7140 | 0.8396 |
| **v3_0_1** | 0.8458 | 0.8794 | 0.8745 | 0.1542 | 0.7548 | 0.8623 |
| **v3_1** | **0.8460** | **0.9000** | **0.8817** | **0.1540** | **0.7634** | **0.8722** |

*註：最新版本的 `v3_1` 模型在準確率、召回率與 mIoU (76.34%) 上皆有顯著提升。*

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

---

## Coast AI 訓練流程

### 事前準備

- 確認已安裝 Python 環境與相依套件（參考 `env_yal/requirements.txt`）
- 所有指令皆在專案根目錄（`unet-pytorch/`）執行

---

### PART 1：整理 CVAT 資料集

**步驟 1：解壓縮 CVAT 下載的資料包**

將從 CVAT 下載的 `cvat.zip` 解壓縮，得到一個資料夾（以下稱「CVAT來源資料夾」）。
該資料夾內部結構應長這樣：

```
CVAT來源資料夾/
├── JPEGImages/
│   ├── train/   ← 訓練用原始圖片
│   └── test/    ← 測試用原始圖片
└── SegmentationClass/
    ├── train/   ← 訓練用標注遮罩
    └── test/    ← 測試用標注遮罩
```

**步驟 2：執行資料整理腳本**

開啟 `tool/format_cvat_dataset.py`，修改最底部的兩個路徑設定：

```python
folder_a = "/你的路徑/CVAT來源資料夾"   # 步驟1解壓縮後的資料夾
folder_b = "/你的路徑/整理後輸出資料夾"  # 可以是任意空資料夾
```

然後執行：

```bash
python tool/format_cvat_dataset.py
```

執行完成後，輸出資料夾（`folder_b`）的結構會變成：

```
整理後輸出資料夾/
├── train/
│   ├── img/          ← 訓練用圖片（已整理）
│   └── mask_origin/  ← 訓練用原始遮罩（已補齊缺失的 mask）
└── test/
    ├── img/          ← 測試用圖片
    └── mask_origin/  ← 測試用原始遮罩
```

---

### PART 2：訓練模型

**步驟 3：建立 VOC 標準資料夾結構**

```bash
python tool/creat_voc_folder.py
```

這會在專案根目錄自動建立以下空資料夾（不需手動建立）：

```
VOCdevkit/VOC2007/JPEGImages/
VOCdevkit/VOC2007/SegmentationClass_Origin/
VOCdevkit/VOC2007/SegmentationClass/
VOCdevkit/VOC2007/ImageSets/Segmentation/
```

**步驟 4：複製訓練資料到 VOC 資料夾**

將 PART 1 整理出來的訓練資料複製進去：

| 來源 | 目的地 |
|------|--------|
| `整理後輸出資料夾/train/img/` 內的所有檔案 | `VOCdevkit/VOC2007/JPEGImages/` |
| `整理後輸出資料夾/train/mask_origin/` 內的所有檔案 | `VOCdevkit/VOC2007/SegmentationClass_Origin/` |

> 注意：是複製「資料夾內的檔案」，不是複製資料夾本身

**步驟 5：將遮罩轉換為二值化格式**

開啟 `tool/convert_binary_img.py`，確認路徑設定正確（預設值如下）：

```python
input_folder  = "VOCdevkit/VOC2007/SegmentationClass_Origin"
output_folder = "VOCdevkit/VOC2007/SegmentationClass"
```

然後執行：

```bash
python tool/convert_binary_img.py
```

這會把原始遮罩（灰度值 0~255）轉換成二值圖（只有 0 和 1），結果存入 `SegmentationClass/`。

**步驟 6：產生訓練集與驗證集的清單檔**

開啟 `tool/voc_annotation.py`，確認以下設定（預設 80% 訓練 / 20% 驗證）：

```python
trainval_percent = 1    # 全部資料都參與劃分（不建議改動）
train_percent    = 0.8  # 80% 訓練集，20% 驗證集（可依需求調整）
VOCdevkit_path   = 'VOCdevkit'  # 不需修改
```

然後執行：

```bash
python tool/voc_annotation.py
```

這會自動產生 `VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt` 和 `val.txt`。

**步驟 7：設定訓練參數並開始訓練**

開啟 `train.py`，找到並修改以下關鍵參數：

```python
num_classes = 2          # 類別數（背景 + 分割目標數）
                         # 只分割一種物體 → 填 2
model_path  = "pth_folder/unet_resnet_voc.pth"
                         # 預訓練模型路徑；從頭訓練請填空字串 ""
input_shape = [512, 512] # 輸入圖片大小，必須是 32 的倍數
```

確認設定後執行：

```bash
python train.py
```

訓練完成的模型權重（`.pth` 檔）會自動儲存在 `logs/` 資料夾內。

---

### PART 3：測試模型效果

**步驟 8：準備測試資料**

將 PART 1 整理出來的測試資料放到專案根目錄的對應資料夾：

| 來源 | 目的地 |
|------|--------|
| `整理後輸出資料夾/test/img/` 內的所有檔案 | `img/` |
| `整理後輸出資料夾/test/mask_origin/` 內的所有檔案 | `mask_origin/` |

> `img/` 和 `mask_origin/` 若不存在請手動建立

**步驟 9：將測試遮罩轉換為二值化格式**

開啟 `tool/convert_binary_img.py`，修改路徑設定：

```python
input_folder  = "mask_origin"
output_folder = "mask"
```

然後執行：

```bash
python tool/convert_binary_img.py
```

**步驟 10：設定要測試的模型並執行預測**

開啟 `unet.py`，找到 `model_path` 並改成你要測試的模型路徑：

```python
model_path = "logs/你的模型.pth"   # 改成 logs/ 內實際的 .pth 檔名
```

然後執行：

```bash
python predict_performance.py
```

預測結果圖片會輸出到 `img_out/` 資料夾內。

---

### 小工具說明

| 腳本 | 功能 | 使用時機 |
|------|------|---------|
| `tool/rename_mask.py` | 移除 `SegmentationClass/` 內所有檔名開頭的 `pm_` 前綴 | 遮罩檔名是 `pm_xxxx.png` 格式時 |
| `tool/to_jpg.py` | 將 `JPEGImages/` 內的 `.jpeg` 副檔名批次改為 `.jpg` | CVAT 匯出的圖片副檔名是 `.jpeg` 時 |
