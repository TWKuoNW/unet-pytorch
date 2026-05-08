import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch

'''
訓練自己的語義分割模型一定需要注意以下幾點：
1、訓練前仔細檢查自己的格式是否滿足要求，該庫要求數據集格式為VOC格式，需要準備好的內容有輸入圖片和標簽
   輸入圖片為.jpg圖片，無需固定大小，傳入訓練前會自動進行resize。
   灰度圖會自動轉成RGB圖片進行訓練，無需自己修改。
   輸入圖片如果後綴非jpg，需要自己批量轉成jpg後再開始訓練。

   標簽為png圖片，無需固定大小，傳入訓練前會自動進行resize。
   由於許多同學的數據集是網絡上下載的，標簽格式並不符合，需要再度處理。一定要注意！標簽的每個像素點的值就是這個像素點所屬的種類。
   網上常見的數據集總共對輸入圖片分兩類，背景的像素點值為0，目標的像素點值為255。這樣的數據集可以正常運行但是預測是沒有效果的！
   需要改成，背景的像素點值為0，目標的像素點值為1。
   如果格式有誤，參考：https://github.com/bubbliiiing/segmentation-format-fix

2、損失值的大小用於判斷是否收斂，比較重要的是有收斂的趨勢，即驗證集損失不斷下降，如果驗證集損失基本上不改變的話，模型基本上就收斂了。
   損失值的具體大小並沒有什麽意義，大和小只在於損失的計算方式，並不是接近於0才好。如果想要讓損失好看點，可以直接到對應的損失函數里面除上10000。
   訓練過程中的損失值會保存在logs文件夾下的loss_%Y_%m_%d_%H_%M_%S文件夾中
   
3、訓練好的權值文件保存在logs文件夾中，每個訓練世代（Epoch）包含若幹訓練步長（Step），每個訓練步長（Step）進行一次梯度下降。
   如果只是訓練了幾個Step是不會保存的，Epoch和Step的概念要捋清楚一下。
'''
if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    是否使用Cuda
    #           沒有GPU可以設置成False
    #---------------------------------#
    Cuda = True
    #----------------------------------------------#
    #   Seed    用於固定隨機種子
    #           使得每次獨立訓練都可以獲得一樣的結果
    #----------------------------------------------#
    seed            = 1
    #---------------------------------------------------------------------#
    #   distributed     用於指定是否使用單機多卡分布式運行
    #                   終端指令僅支持Ubuntu。CUDA_VISIBLE_DEVICES用於在Ubuntu下指定顯卡。
    #                   Windows系統下默認使用DP模式調用所有顯卡，不支持DDP。
    #   DP模式：
    #       設置            distributed = False
    #       在終端中輸入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       設置            distributed = True
    #       在終端中輸入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False # 分散式
    #---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    #---------------------------------------------------------------------#
    sync_bn         = False 
    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度訓練
    #               可減少約一半的顯存、需要pytorch1.7.1以上
    #---------------------------------------------------------------------#
    fp16            = False
    #-----------------------------------------------------#
    #   num_classes     訓練自己的數據集必須要修改的
    #                   自己需要的分類個數+1，如2+1
    #-----------------------------------------------------#
    num_classes = 2
    #-----------------------------------------------------#
    #   主幹網絡選擇
    #   vgg
    #   resnet50
    #-----------------------------------------------------#
    backbone    = "resnet50"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      是否使用主幹網絡的預訓練權重，此處使用的是主幹的權重，因此是在模型構建的時候進行加載的。
    #                   如果設置了model_path，則主幹的權值無需加載，pretrained的值無意義。
    #                   如果不設置model_path，pretrained = True，此時僅加載主幹開始訓練。
    #                   如果不設置model_path，pretrained = False，Freeze_Train = Fasle，此時從0開始訓練，且沒有凍結主幹的過程。
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained  = False
    #----------------------------------------------------------------------------------------------------------------------------#
    #   權值文件的下載請看README，可以通過網盤下載。模型的 預訓練權重 對不同數據集是通用的，因為特征是通用的。
    #   模型的 預訓練權重 比較重要的部分是 主幹特征提取網絡的權值部分，用於進行特征提取。
    #   預訓練權重對於99%的情況都必須要用，不用的話主幹部分的權值太過隨機，特征提取效果不明顯，網絡訓練的結果也不會好
    #   訓練自己的數據集時提示維度不匹配正常，預測的東西都不一樣了自然維度不匹配
    #
    #   如果訓練過程中存在中斷訓練的操作，可以將model_path設置成logs文件夾下的權值文件，將已經訓練了一部分的權值再次載入。
    #   同時修改下方的 凍結階段 或者 解凍階段 的參數，來保證模型epoch的連續性。
    #   
    #   當model_path = ''的時候不加載整個模型的權值。
    #
    #   此處使用的是整個模型的權重，因此是在train.py進行加載的，pretrain不影響此處的權值加載。
    #   如果想要讓模型從主幹的預訓練權值開始訓練，則設置model_path = ''，pretrain = True，此時僅加載主幹。
    #   如果想要讓模型從0開始訓練，則設置model_path = ''，pretrain = Fasle，Freeze_Train = Fasle，此時從0開始訓練，且沒有凍結主幹的過程。
    #   
    #   一般來講，網絡從0開始的訓練效果會很差，因為權值太過隨機，特征提取效果不明顯，因此非常、非常、非常不建議大家從0開始訓練！
    #   如果一定要從0開始，可以了解imagenet數據集，首先訓練分類模型，獲得網絡的主幹部分權值，分類模型的 主幹部分 和該模型通用，基於此進行訓練。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path  = "pth_folder/unet_resnet_voc.pth"
    #-----------------------------------------------------#
    #   input_shape     輸入圖片的大小，32的倍數
    #-----------------------------------------------------#
    input_shape = [512, 512]
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   訓練分為兩個階段，分別是凍結階段和解凍階段。設置凍結階段是為了滿足機器性能不足的同學的訓練需求。
    #   凍結訓練需要的顯存較小，顯卡非常差的情況下，可設置Freeze_Epoch等於UnFreeze_Epoch，此時僅僅進行凍結訓練。
    #   
    #   在此提供若幹參數設置建議，各位訓練者根據自己的需求進行靈活調整：
    #   （一）從整個模型的預訓練權重開始訓練： 
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（凍結）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（不凍結）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（凍結）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（不凍結）
    #       其中：UnFreeze_Epoch可以在100-300之間調整。
    #   （二）從主幹網絡的預訓練權重開始訓練：
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（凍結）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（不凍結）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 120，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（凍結）
    #           Init_Epoch = 0，UnFreeze_Epoch = 120，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（不凍結）
    #       其中：由於從主幹網絡的預訓練權重開始訓練，主幹的權值不一定適合語義分割，需要更多的訓練跳出局部最優解。
    #             UnFreeze_Epoch可以在120-300之間調整。
    #             Adam相較於SGD收斂的快一些。因此UnFreeze_Epoch理論上可以小一點，但依然推薦更多的Epoch。
    #   （三）batch_size的設置：
    #       在顯卡能夠接受的範圍內，以大為好。顯存不足與數據集大小無關，提示顯存不足（OOM或者CUDA out of memory）請調小batch_size。
    #       由於resnet50中有BatchNormalization層
    #       當主幹為resnet50的時候batch_size不可為1
    #       正常情況下Freeze_batch_size建議為Unfreeze_batch_size的1-2倍。不建議設置的差距過大，因為關系到學習率的自動調整。
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   凍結階段訓練參數
    #   此時模型的主幹被凍結了，特征提取網絡不發生改變
    #   占用的顯存較小，僅對網絡進行微調
    #   Init_Epoch          模型當前開始的訓練世代，其值可以大於Freeze_Epoch，如設置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       會跳過凍結階段，直接從60代開始，並調整對應的學習率。
    #                       （斷點續練時使用）
    #   Freeze_Epoch        模型凍結訓練的Freeze_Epoch
    #                       (當Freeze_Train=False時失效)
    #   Freeze_batch_size   模型凍結訓練的batch_size
    #                       (當Freeze_Train=False時失效)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 20
    Freeze_batch_size   = 8
    #------------------------------------------------------------------#
    #   解凍階段訓練參數
    #   此時模型的主幹不被凍結了，特征提取網絡會發生改變
    #   占用的顯存較大，網絡所有的參數都會發生改變
    #   UnFreeze_Epoch          模型總共訓練的epoch
    #   Unfreeze_batch_size     模型在解凍後的batch_size
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 8
    #------------------------------------------------------------------#
    #   Freeze_Train    是否進行凍結訓練
    #                   默認先凍結主幹訓練後解凍訓練。
    #------------------------------------------------------------------#
    Freeze_Train        = True

    #------------------------------------------------------------------#
    #   其它訓練參數：學習率、優化器、學習率下降有關
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大學習率
    #                   當使用Adam優化器時建議設置  Init_lr=1e-4
    #                   當使用SGD優化器時建議設置   Init_lr=1e-2
    #   Min_lr          模型的最小學習率，默認為最大學習率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的優化器種類，可選的有adam、sgd
    #                   當使用Adam優化器時建議設置  Init_lr=1e-4
    #                   當使用SGD優化器時建議設置   Init_lr=1e-2
    #   momentum        優化器內部使用到的momentum參數
    #   weight_decay    權值衰減，可防止過擬合
    #                   adam會導致weight_decay錯誤，使用adam時建議設置為0。
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的學習率下降方式，可選的有'step'、'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     多少個epoch保存一次權值
    #------------------------------------------------------------------#
    save_period         = 5
    #------------------------------------------------------------------#
    #   save_dir        權值與日志文件保存的文件夾
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       是否在訓練時進行評估，評估對象為驗證集
    #   eval_period     代表多少個epoch評估一次，不建議頻繁的評估
    #                   評估需要消耗較多的時間，頻繁評估會導致訓練非常慢
    #   此處獲得的mAP會與get_map.py獲得的會有所不同，原因有二：
    #   （一）此處獲得的mAP為驗證集的mAP。
    #   （二）此處設置評估參數較為保守，目的是加快評估速度。
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 1
    
    #------------------------------#
    #   數據集路徑
    #------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #------------------------------------------------------------------#
    #   建議選項：
    #   種類少（幾類）時，設置為True
    #   種類多（十幾類）時，如果batch_size比較大（10以上），那麽設置為True
    #   種類多（十幾類）時，如果batch_size比較小（10以下），那麽設置為False
    #------------------------------------------------------------------#
    dice_loss       = False
    #------------------------------------------------------------------#
    #   是否使用focal loss來防止正負樣本不平衡
    #------------------------------------------------------------------#
    focal_loss      = False
    #------------------------------------------------------------------#
    #   是否給不同種類賦予不同的損失權值，默認是平衡的。
    #   設置的話，注意設置成numpy形式的，長度和num_classes一樣。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    #------------------------------------------------------------------#
    cls_weights     = np.ones([num_classes], np.float32)
    #------------------------------------------------------------------#
    #   num_workers     用於設置是否使用多線程讀取數據，1代表關閉多線程
    #                   開啟後會加快數據讀取速度，但是會占用更多內存
    #                   keras里開啟多線程有些時候速度反而慢了許多
    #                   在IO為瓶頸的時候再開啟多線程，即GPU運算速度遠大於讀取圖片的速度。
    #------------------------------------------------------------------#
    num_workers     = 4

    seed_everything(seed)
    #------------------------------------------------------#
    #   設置用到的顯卡
    #------------------------------------------------------#
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

    #----------------------------------------------------#
    #   下載預訓練權重
    #----------------------------------------------------#
    if pretrained: # <- False 不運行
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
        #------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
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
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    #----------------------#
    #   记录Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   多卡同步Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   多卡平行运行
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
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
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        if Freeze_Train:
            model.freeze_backbone()
            
        #-------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
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
        
        #----------------------#
        #   记录eval的map曲线
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   获得学习率下降的公式
                #---------------------------------------#
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
