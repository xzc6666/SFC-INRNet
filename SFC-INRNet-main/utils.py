import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from warmup_scheduler import GradualWarmupScheduler

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# ==============================================================================
# SFC-INRNet Utilities
# Contains functions for initialization, data augmentation, normalization,
# and optimization tracking tailored for Spatial-Frequency Collaborative Networks.
# ==============================================================================

def seed_pytorch(seed=42):
    """
    Set fixed random seeds to ensure experimental reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 and classname.find('SplAtConv2d') == -1:
        init.xavier_normal_(m.weight.data)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class SFC_SpatialGradient(nn.Module):
    """
    Computes spatial gradients. Can be utilized for edge-aware loss formulations
    to preserve the high-frequency structural integrity of small targets.
    """

    def __init__(self):
        super(SFC_SpatialGradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)
        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        return x0


def random_crop(img, mask, patch_size, pos_prob=None):
    """
    Randomly crops image and mask into a specified patch size (e.g., 256x256).
    Applies zero-padding if the original image dimensions are smaller than patch_size.
    """
    h, w = img.shape
    # Pad spatial dimensions to the target patch size if necessary
    if min(h, w) < patch_size:
        pad_h = max(h, patch_size) - h
        pad_w = max(w, patch_size) - w
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
        h, w = img.shape

    while True:
        h_start = random.randint(0, h - patch_size)
        h_end = h_start + patch_size
        w_start = random.randint(0, w - patch_size)
        w_end = w_start + patch_size

        img_patch = img[h_start:h_end, w_start:w_end]
        mask_patch = mask[h_start:h_end, w_start:w_end]

        if pos_prob is None or random.random() > pos_prob:
            break
        elif mask_patch.sum() > 0:  # Ensure target presence in crop when prob threshold met
            break

    return img_patch, mask_patch


def Normalized(img, img_norm_cfg):
    return (img - img_norm_cfg['mean']) / img_norm_cfg['std']


def Denormalization(img, img_norm_cfg):
    return img * img_norm_cfg['std'] + img_norm_cfg['mean']


def get_img_norm_cfg(dataset_name, dataset_dir):
    """
    Retrieves the specific normalization parameters (mean, std) for the datasets evaluated in SFC-INRNet.
    Dynamically computes them if the dataset is not predefined.
    """
    if dataset_name == 'NUAA-SIRST':
        img_norm_cfg = dict(mean=101.06385, std=34.61960)
    elif dataset_name == 'NUDT-SIRST':
        img_norm_cfg = dict(mean=107.80905, std=33.02274)
    elif dataset_name == 'IRSTD-1K':
        img_norm_cfg = dict(mean=87.46618, std=39.71953)
    else:
        # Dynamic calculation for custom datasets
        train_list_path = os.path.join(dataset_dir, dataset_name, 'img_idx', f'train_{dataset_name}.txt')
        test_list_path = os.path.join(dataset_dir, dataset_name, 'img_idx', f'test_{dataset_name}.txt')

        with open(train_list_path, 'r') as f:
            train_list = f.read().splitlines()
        with open(test_list_path, 'r') as f:
            test_list = f.read().splitlines()

        img_list = train_list + test_list
        img_dir = os.path.join(dataset_dir, dataset_name, 'images')

        mean_list, std_list = [], []
        for img_pth in img_list:
            img_path = os.path.join(img_dir, f'{img_pth}.png')
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, f'{img_pth}.bmp')

            img = np.array(Image.open(img_path).convert('I'), dtype=np.float32)
            mean_list.append(img.mean())
            std_list.append(img.std())

        img_norm_cfg = dict(mean=float(np.mean(mean_list)), std=float(np.mean(std_list)))

    return img_norm_cfg


def get_optimizer(net, optimizer_name, scheduler_name, optimizer_settings, scheduler_settings):
    """
    Configures the optimizer and learning rate scheduler.
    Uses Cosine Annealing with Warmup to ensure stable gradient descent for deep implicit modeling.
    """
    # Optimizer Selection
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=optimizer_settings['lr'],
                                      weight_decay=optimizer_settings.get('weight_decay', 1e-2))
    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=optimizer_settings['lr'],
                                    momentum=0.9,
                                    weight_decay=scheduler_settings.get('weight_decay', 1e-4))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Scheduler Selection
    if scheduler_name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=scheduler_settings['step'],
                                                         gamma=scheduler_settings['gamma'])
    elif scheduler_name == 'CosineAnnealingLR':
        warmup_epochs = 10
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_settings['epochs'] - warmup_epochs,
            eta_min=scheduler_settings['eta_min']
        )
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)
    elif scheduler_name == 'CosineAnnealingLRw50':
        warmup_epochs = 50
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_settings['epochs'] - warmup_epochs,
            eta_min=scheduler_settings['eta_min']
        )
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)
    elif scheduler_name == 'CosineAnnealingLRw0':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_settings['epochs'],
            eta_min=scheduler_settings['eta_min']
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return optimizer, scheduler


def PadImg(img, times=32):
    """
    Pads the input image ensuring its height and width are strictly divisible by the network's stride (e.g., 32).
    This prevents spatial dimensional collapse during the multi-scale pooling stages of the backbone.
    """
    h, w = img.shape
    pad_h = 0 if h % times == 0 else (h // times + 1) * times - h
    pad_w = 0 if w % times == 0 else (w // times + 1) * times - w

    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')

    return img




