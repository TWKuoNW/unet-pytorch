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
import datetime


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

class U_Net_pro():
    def __init__(self):
        self.dataset_root = Path("VOCdevkit/VOC2007")
        self.annotation_path = self.dataset_root/"ImageSets/Segmentation"
        self.JPEGImages_path = self.dataset_root/"JPEGImages"
        self.SegmentationClass_path = self.dataset_root/"SegmentationClass"
        self.SegmentationClassOrigin_path = self.dataset_root/"SegmentationClassOrigin"
    
    def train(self, train_imgs_path, train_mask_path, epochs, batch_size, lr, cuda, model_path, backbone, new_data):
        self.train_imgs_path = train_imgs_path
        self.train_mask_path = train_mask_path
        self.epoch = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.cuda = cuda
        self.model_path = model_path
        self.backbone = backbone
        self.new_data = new_data

        if self.new_data:
            self.clean_old_train_data()
            print("開始複製資料集...")
            self.copy_data(self.train_imgs_path, self.JPEGImages_path)
            self.copy_data(self.train_mask_path, self.SegmentationClassOrigin_path)
            print("完成複製資料集...")    
            self.convert_rgb2bin()
            self.to_jpg()
            self.annotation()
        
        self.train_script()
    
    # clean
    def clean_old_train_data(self):
        print("開始清除舊資料集...")
        targets = [self.annotation_path,
            self.JPEGImages_path,
            self.SegmentationClass_path,
            self.SegmentationClassOrigin_path]

        for d in targets:
            if not d.exists():
                print(f"[skip] {d} 不存在")
                continue
            for p in d.iterdir():
                if p.is_file() or p.is_symlink():
                    p.unlink()
                else:
                    shutil.rmtree(p)
        print("已清除舊資料集...")

    # copy
    def copy_data(self, src_dir, dst_dir):
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

    # rgb2bin
    def convert_rgb2bin(self):
        input_folder = self.SegmentationClassOrigin_path
        output_folder = self.SegmentationClass_path
        os.makedirs(output_folder, exist_ok=True) 
        files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]
        for filename in tqdm(files, desc="Processing images"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            binary = np.where(image == 0, 0, 1).astype(np.uint8)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, binary)
    
    # 2JPG
    def to_jpg(self):
        folder = self.JPEGImages_path
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

    # annotation
    def annotation(self):
        trainval_percent    = 1
        train_percent       = 0.9
        random.seed(0)
        print("Generate txt in ImageSets.")
        segfilepath     = self.SegmentationClass_path
        saveBasePath    = self.annotation_path 

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

    # train script 
    def train_script(self):
        Cuda = self.cuda
        seed            = 11
        distributed     = False
        sync_bn         = False 
        fp16            = False
        num_classes = 2
        backbone    = self.backbone
        pretrained  = False
        model_path  = self.model_path
        input_shape = [512, 512]
        Init_Epoch          = 0
        Freeze_Epoch        = int(self.epoch*0.2)
        Freeze_batch_size   = 16
        UnFreeze_Epoch      = self.epoch
        Unfreeze_batch_size = 16
        Freeze_Train        = True
        Init_lr             = 1e-4
        Min_lr              = Init_lr * 0.01
        optimizer_type      = "adam"
        momentum            = 0.9
        weight_decay        = 0
        lr_decay_type       = 'cos'
        save_period         = 5
        save_dir            = 'logs'
        eval_flag           = True
        eval_period         = 1
        VOCdevkit_path  = 'VOCdevkit'
        dice_loss       = False
        focal_loss      = False
        cls_weights     = np.ones([num_classes], np.float32)
        num_workers     = 0
        
        ngpus_per_node  = torch.cuda.device_count()
        
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

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

            for epoch in range(Init_Epoch, UnFreeze_Epoch):
                
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


            if local_rank == 0:
                loss_history.writer.close()  
        
if __name__ == "__main__":
    r = U_Net_pro()
    r.train(
        train_imgs_path = r"E:\naiwen_folder\訓練資料\測試_文獻_東沙_420\test_420", 
        train_mask_path = r"E:\naiwen_folder\訓練資料\測試_文獻_東沙_420\test_420_mask", 
        epochs = 100, 
        batch_size = 32, 
        lr = 0.01,
        cuda = True,
        model_path = r"model_folder\unet_resnet_voc.pth",
        backbone = "resnet50", # resnet50
        new_data = False, # False
    )

