import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import threading, time
import datetime
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import shutil
import cv2
import os
import random

from PIL import Image
from functools import partial
from torch.utils.data import DataLoader
from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch
from pathlib import Path
from tqdm import tqdm 

# ---------------- 模擬訓練 / 推論 worker ----------------
def run_prediction(progress, log, total=20):
    log.insert(tk.END, "Start predicting...\n")
    for i in range(1, total + 1):
        time.sleep(0.2)
        pct = int(i / total * 100)
        progress["value"] = pct
        log.insert(tk.END, f"Predicted {i}/{total} images.\n")
        log.see(tk.END)
    log.insert(tk.END, "Prediction finished.\n")

def run_training(progress ,log , cuda = False, epoch = 100, save = 5):
    Cuda = cuda
    seed            = 11
    distributed     = False
    sync_bn         = False 
    fp16            = False
    num_classes = 2
    backbone    = "vgg"
    pretrained  = False
    model_path  = "pth_folder/unet_vgg_voc.pth"
    input_shape = [512, 512]
    Init_Epoch          = 0
    Freeze_Epoch        = int(epoch*0.2)
    Freeze_batch_size   = 16
    UnFreeze_Epoch      = epoch
    Unfreeze_batch_size = 16
    Freeze_Train        = True
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    lr_decay_type       = 'cos'
    save_period         = save
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 1
    VOCdevkit_path  = 'VOCdevkit'
    dice_loss       = False
    focal_loss      = False
    cls_weights     = np.ones([num_classes], np.float32)
    num_workers     = 0
    
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else: # <- 通常走這
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)

    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train() # 執行訓練任務
    if not pretrained:
        weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:

            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
        
    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )

    if True:
        UnFreeze_flag = False
        
        if Freeze_Train:
            model.freeze_backbone()    
        
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        safe_log(log, f"Start training for {UnFreeze_Epoch} epochs...")

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            pct = int(epoch / UnFreeze_Epoch * 100)
            progress["value"] = pct
            
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                model.unfreeze_backbone()
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

            safe_progress(progress, int((epoch) / UnFreeze_Epoch * 100))
            safe_log(log, f"Epoch {epoch+1}/{UnFreeze_Epoch} done.")


        if local_rank == 0:
            loss_history.writer.close()  
    
    safe_log(log, "Training finished.")

# ---------------- GUI ----------------
root = tk.Tk()
root.title("UNet Trainer (Tkinter)")
root.geometry("900x600")

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# ===== Train Tab =====
frame_train = ttk.Frame(notebook)
notebook.add(frame_train, text="Train")

# Dataset group
dataset_frame = ttk.LabelFrame(frame_train, text="Dataset Paths")
dataset_frame.pack(fill="x", padx=10, pady=5)
ttk.Label(dataset_frame, text="Images:").grid(row=0, column=0, sticky="w")
le_images = ttk.Entry(dataset_frame, width=60)
le_images.grid(row=0, column=1)
ttk.Button(dataset_frame, text="Browse", command=lambda: le_images.insert(0, filedialog.askdirectory())).grid(row=0, column=2)

ttk.Label(dataset_frame, text="Masks:").grid(row=1, column=0, sticky="w")
le_masks = ttk.Entry(dataset_frame, width=60)
le_masks.grid(row=1, column=1)
ttk.Button(dataset_frame, text="Browse", command=lambda: le_masks.insert(0, filedialog.askdirectory())).grid(row=1, column=2)

# Hyperparams group
hyper_frame = ttk.LabelFrame(frame_train, text="Hyperparameters")
hyper_frame.pack(fill="x", padx=10, pady=5)
ttk.Label(hyper_frame, text="Epochs:").grid(row=0, column=0)
sb_epochs = ttk.Spinbox(hyper_frame, from_=1, to=1000, width=5)
sb_epochs.set(100)
sb_epochs.grid(row=0, column=1)

ttk.Label(hyper_frame, text="Batch:").grid(row=0, column=2)
sb_batch = ttk.Spinbox(hyper_frame, from_=1, to=128, width=5)
sb_batch.set(16)
sb_batch.grid(row=0, column=3)

ttk.Label(hyper_frame, text="Save:").grid(row=0, column=4)
sb_save = ttk.Spinbox(hyper_frame, from_=1, to=128, width=5)
sb_save.set(5)
sb_save.grid(row=0, column=5)

ttk.Label(hyper_frame, text="CUDA:").grid(row=0, column=6, padx=(20,0))
chk_cuda = ttk.Checkbutton(hyper_frame, variable=tk.BooleanVar(value=True))
chk_cuda.grid(row=0, column=7)

ttk.Label(hyper_frame, text="New Data:").grid(row=0, column=8, padx=(4,0))
chk_new_data = ttk.Checkbutton(hyper_frame, variable=tk.BooleanVar(value=True))
chk_new_data.grid(row=0, column=9)

# Train buttons + progress
train_ctrl_frame = ttk.Frame(frame_train)
train_ctrl_frame.pack(fill="x", padx=10, pady=5)
btn_start_train = ttk.Button(train_ctrl_frame, text="Start Training")
btn_start_train.pack(side="left", padx=5)
btn_stop_train = ttk.Button(train_ctrl_frame, text="Stop")
btn_stop_train.pack(side="left", padx=5)
pb_train = ttk.Progressbar(train_ctrl_frame, length=300)
pb_train.pack(side="left", padx=10)

# Train log
te_train_log = tk.Text(frame_train, height=15)
te_train_log.pack(fill="both", padx=10, pady=5, expand=True)

# ===== Predict Tab =====
frame_pred = ttk.Frame(notebook)
notebook.add(frame_pred, text="Predict")

pred_frame = ttk.LabelFrame(frame_pred, text="Paths")
pred_frame.pack(fill="x", padx=10, pady=5)
ttk.Label(pred_frame, text="Checkpoint:").grid(row=0, column=0, sticky="w")
le_ckpt = ttk.Entry(pred_frame, width=60)
le_ckpt.grid(row=0, column=1)
ttk.Button(pred_frame, text="Browse", command=lambda: le_ckpt.insert(0, filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt *.pth")]))).grid(row=0, column=2)

ttk.Label(pred_frame, text="Input Images:").grid(row=1, column=0, sticky="w")
le_pred_in = ttk.Entry(pred_frame, width=60)
le_pred_in.grid(row=1, column=1)
ttk.Button(pred_frame, text="Browse", command=lambda: le_pred_in.insert(0, filedialog.askdirectory())).grid(row=1, column=2)

ttk.Label(pred_frame, text="Output Folder:").grid(row=2, column=0, sticky="w")
le_pred_out = ttk.Entry(pred_frame, width=60)
le_pred_out.grid(row=2, column=1)
ttk.Button(pred_frame, text="Browse", command=lambda: le_pred_out.insert(0, filedialog.askdirectory())).grid(row=2, column=2)

# Predict buttons + progress
pred_ctrl_frame = ttk.Frame(frame_pred)
pred_ctrl_frame.pack(fill="x", padx=10, pady=5)
btn_run_pred = ttk.Button(pred_ctrl_frame, text="Run Predict")
btn_run_pred.pack(side="left", padx=5)

pb_pred = ttk.Progressbar(pred_ctrl_frame, length=300)
pb_pred.pack(side="left", padx=10)

# Predict log
te_pred_log = tk.Text(frame_pred, height=15)
te_pred_log.pack(fill="both", padx=10, pady=5, expand=True)

# ---------- 綁定功能 ----------
def ui_log(msg):
    te_train_log.insert(tk.END, msg + "\n")
    te_train_log.see(tk.END)
    te_train_log.update_idletasks()   # 讓 Tk 把字畫出來
    root.update()                     # 真的處理事件（可選，較暴力）

def clean_old_train_data(log):
    ROOT = Path("VOCdevkit/VOC2007")
    TARGETS = [ROOT/"ImageSets/Segmentation", ROOT/"JPEGImages", ROOT/"SegmentationClass", ROOT/"SegmentationClassOrigin"]

    for d in TARGETS:
        if not d.exists():
            print(f"[skip] {d} 不存在")
            continue
        for p in d.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink()
            else:
                shutil.rmtree(p)
        log.insert(tk.END, f"已清空：{d}\n")

def copy_data(src_dir, dst_dir):
    src = Path(src_dir)
    dst = Path(dst_dir)

    if not src.is_dir():
        raise ValueError(f"來源不是資料夾：{src}")

    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)  # 合併資料夾、覆蓋同名檔
        else:
            shutil.copy2(item, target)  # 複製檔案並保留中繼資料

def convert_rgb2bin(input_folder, output_folder):
    
    os.makedirs(output_folder, exist_ok=True) 

    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]

    for filename in tqdm(files, desc="Processing images"):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        binary = np.where(image == 0, 0, 1).astype(np.uint8)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, binary)

def annotation():
    trainval_percent    = 1
    train_percent       = 0.9
    VOCdevkit_path      = 'VOCdevkit'
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num     = len(total_seg)  
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  
    
    print("train and val size",tv)
    print("traub suze",tr)
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name = total_seg[i][:-4]+'\n'  
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
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    print("检查数据集格式是否符合要求，这可能需要一段时间。")
    classes_nums        = np.zeros([256], np.int64)
    for i in tqdm(list):
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片%s，请查看具体路径下文件是否存在以及后缀是否为png。"%(png_file_name))
        
        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。"%(name, str(np.shape(png))))
            print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。"%(name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
        print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

    print("JPEGImages中的图片应当为.jpg文件、SegmentationClass中的图片应当为.png文件。")
    print("如果格式有误，参考:")
    print("https://github.com/bubbliiiing/segmentation-format-fix")

def to_jpg():
    folder = Path("VOCdevkit/VOC2007/JPEGImages") 
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

def safe_log(widget: tk.Text, msg: str):
    # 在主執行緒安全寫入 Text
    if widget and widget.winfo_exists():
        widget.after(0, lambda m=msg: (widget.insert(tk.END, m + "\n"),
                                       widget.see(tk.END)))

def safe_progress(pbar: ttk.Progressbar, value: int):
    # 在主執行緒安全更新進度條
    value = max(0, min(100, int(value)))
    if pbar and pbar.winfo_exists():
        pbar.after(0, lambda v=value: pbar.config(value=v))

def start_train_thread():
    new_data = chk_new_data.instate(['selected']) # 確認是否是新資料
    if(new_data): # 是的話執行以下
        ui_log("正在初始化資料夾...")
        clean_old_train_data(te_train_log)
        try:
            ui_log("正在複製資料集...")
            if(le_images.get() == ""):
                ui_log("資料夾路徑空～")
            else:
                img_path = le_images.get()
                mask_path = le_masks.get()
                copy_data(img_path, "VOCdevkit/VOC2007/JPEGImages")
                copy_data(mask_path, "VOCdevkit/VOC2007/SegmentationClassOrigin")
                ui_log("複製完成～")
        except:
            print("複製資料集時出問題...")

        try:
            ui_log("正在統一副檔名...")
            t = threading.Thread(
                target=convert_rgb2bin, 
                args=("VOCdevkit/VOC2007/SegmentationClassOrigin",
                      "VOCdevkit/VOC2007/SegmentationClass"),
                daemon=True
            )
            t.start()
            t.join()
            ui_log("統一副檔名完成～")
        except:
            print("統一副檔名時出問題...")

        try:
            ui_log("正在轉換為binary...")
            t = threading.Thread(
                target=to_jpg, 
                daemon=True
            )
            t.start()
            t.join()
            ui_log("轉換完成～")
        except:
            print("轉換binary時出問題...")

        try:
            ui_log("正在建立註解...")
            t = threading.Thread(target=annotation, daemon=True)
            t.start()
            t.join()
            ui_log("建立完成～")
        except:
            print("建立註解時出問題...")
    try:
        ui_log("正在開始訓練...")
        cuda = chk_cuda.instate(['selected'])
        threading.Thread(
            target=run_training, 
            args=(pb_train, te_train_log, cuda, 100,  5), 
            daemon=True
        ).start()
    except:
        print("訓練時出錯...")

def start_pred_thread():
    threading.Thread(target=run_prediction, args=(pb_pred, te_pred_log), daemon=True).start()

btn_start_train.config(command=start_train_thread)
btn_run_pred.config(command=start_pred_thread)

# Stop 按鈕示範（實際要加 flag）
btn_stop_train.config(command=lambda: messagebox.showinfo("Stop", "Stop pressed (implement stop logic)"))

# ---------- Run ----------
root.mainloop()


