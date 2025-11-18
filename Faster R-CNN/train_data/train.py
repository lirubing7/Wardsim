#-------------------------------------#
#       Train on your dataset
#-------------------------------------#
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.frcnn import FasterRCNN
from nets.frcnn_training import (FasterRCNNTrainer, get_lr_scheduler,
                                 set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import (get_classes, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

'''
Important notes when training your own object detection model:
1. Before training, carefully check whether your dataset format meets the requirements.
   This repo requires VOC format. You must prepare input images and labels.
   Input images must be .jpg (any size is allowed; resizing is performed automatically before training).
   Grayscale images will be automatically converted to RGB.

   If image suffixes are not .jpg, convert them in batch first.

   Labels must be .xml files containing target information corresponding to each image.

2. The loss value helps determine whether the model is converging.
   What matters is the convergence trend — e.g., validation loss should decrease steadily.
   The absolute value of the loss is NOT meaningful (since it depends on how the loss is computed).
   Loss logs are saved under the "logs/loss_%Y_%m_%d_%H_%M_%S" folder.

3. Trained weight files will be saved in the logs folder.
   Each Epoch contains multiple Steps, and each Step performs one gradient descent update.
   If you only trained a few Steps, weights will NOT be saved.
   Understand the difference between epoch and step clearly.
'''
if __name__ == "__main__":
    #-------------------------------#
    #   Whether to use CUDA
    #   Set to False if no GPU
    #-------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed for reproducibility
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   train_gpu: GPU IDs to use for training
    #   Default = first GPU; multi-GPU example: [0,1], [0,1,2]
    #   Batch size per GPU = total_batch / num_gpu
    #---------------------------------------------------------------------#
    train_gpu       = [0,]
    #---------------------------------------------------------------------#
    #   fp16: whether to use mixed precision training
    #         reduces VRAM usage by ~50%; requires torch>=1.7.1
    #---------------------------------------------------------------------#
    fp16            = False
    #---------------------------------------------------------------------#
    #   classes_path: points to the txt under model_data
    #                 must match your own dataset categories
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Pretrained weights can be downloaded from README.
    #   Pretrained backbone weights are universal across datasets since features are universal.
    #
    #   If training was interrupted, you can set model_path to a weight file in logs,
    #   so you can resume training from intermediate weights.
    #
    #   If model_path = '', weights will NOT be loaded.
    #
    #   If you want to start training from backbone-only pretrained weights:
    #       set model_path = '' and pretrained=True
    #
    #   If you want to train from scratch:
    #       set model_path = '', pretrained=False, Freeze_Train=False
    #
    #   Training from scratch is strongly NOT recommended because the backbone is random
    #   and feature extraction is extremely weak.
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/voc_weights_resnet.pth'
    #------------------------------------------------------#
    #   Input image shape
    #------------------------------------------------------#
    input_shape     = [600, 600]
    #---------------------------------------------#
    #   Backbone: vgg or resnet50
    #---------------------------------------------#
    backbone        = "resnet50"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained: whether to load backbone pretrained weights
    #   If model_path is provided, pretrained is ignored.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = False
    #------------------------------------------------------------------------#
    #   anchors_size sets anchor box scales; each scale generates 3 anchors.
    #   If detecting smaller objects, decrease early anchor scales.
    #------------------------------------------------------------------------#
    anchors_size    = [8, 16, 32]

    #----------------------------------------------------------------------------------------------------------------------------#
    #   Training has two phases: frozen and unfrozen.
    #   Frozen phase trains only the detection head; unfrozen trains full network.
    #
    #   Frozen training needs less VRAM and is useful for low-end GPUs.
    #----------------------------------------------------------------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 4

    #------------------------------------------------------------------#
    #   Unfreeze phase: train all layers
    #   Uses larger memory and updates all parameters
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 2
    #------------------------------------------------------------------#
    #   Freeze_Train: whether to freeze backbone first
    #------------------------------------------------------------------#
    Freeze_Train        = True
    
    #------------------------------------------------------------------#
    #   Other hyperparameters: learning rate, optimizer, LR decay
    #------------------------------------------------------------------#
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    lr_decay_type       = 'cos'
    save_period         = 5
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag: whether to run validation during training
    #   eval_period: evaluate every N epochs (too frequent eval slows training)
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 5
    #------------------------------------------------------------------#
    #   num_workers: number of data loading threads
    #------------------------------------------------------------------#
    num_workers         = 4

    #----------------------------------------------------#
    #   Load dataset annotation txt files
    #----------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'
    
    #----------------------------------------------------#
    #   Load classes
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    #------------------------------------------------------#
    #   Set CUDA device
    #------------------------------------------------------#
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))
    seed_everything(seed)
    
    model = FasterRCNN(num_classes, anchor_scales = anchors_size, backbone = backbone, pretrained = pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   Load matching keys from pretrained weights
        #------------------------------------------------------#
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44mNote: Missing keys in the head are normal. Missing keys in the backbone indicate errors.\033[0m")

    #----------------------#
    #   Create loss logger
    #----------------------#
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history    = LossHistory(log_dir, model, input_shape=input_shape)

    #------------------------------------------------------------------#
    #   fp16 training requires torch>=1.7.1
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    #---------------------------#
    #   Read dataset txt files
    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    
    show_config(
        classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )

    '''
    Total Epochs = how many times the entire dataset is iterated.
    Total Steps = total gradient updates.
    Each epoch contains multiple steps.
    Recommended minimum number of steps (for unfrozen phase only):
        5e4 for SGD
        1.5e4 for Adam
    '''
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        if num_train // Unfreeze_batch_size == 0:
            raise ValueError('Dataset too small to train. Please add more data.')
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] Recommended total steps for %s optimizer: %d+\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] Current training samples: %d, batch size: %d, total epochs: %d\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch))
        print("\033[1;33;44m[Warning] Total steps %d < recommended %d → Suggested epochs: %d\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    #   Frozen training reduces VRAM usage
    #   Freeze_Epoch = how long backbone is frozen
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False

        if Freeze_Train:
            for param in model.extractor.parameters():
                param.requires_grad = False
        model.freeze_bn()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 5e-2
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
            raise ValueError("Dataset too small to continue training. Please add more samples.")

        train_dataset   = FRCNNDataset(train_lines, input_shape, train = True)
        val_dataset     = FRCNNDataset(val_lines, input_shape, train = False)

        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=frcnn_dataset_collate, 
                                    worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=frcnn_dataset_collate, 
                                    worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

        train_util      = FasterRCNNTrainer(model_train, optimizer)
        eval_callback   = EvalCallback(model_train, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                        eval_flag=eval_flag, period=eval_period)

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   Unfreeze backbone at epoch >= Freeze_Epoch
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                for param in model.extractor.parameters():
                    param.requires_grad = True
                model.freeze_bn()

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset too small to continue training. Please add more samples.")

                gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=frcnn_dataset_collate, 
                                            worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=frcnn_dataset_collate, 
                                            worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

                UnFreeze_flag = True
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            fit_one_epoch(model, train_util, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir)
            
        loss_history.writer.close()
