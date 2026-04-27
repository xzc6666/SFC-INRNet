import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# Assuming these functions are correctly implemented in your utils.py
from utils import get_img_norm_cfg, Normalized, random_crop, PadImg

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class SFC_Augmentation(object):
    """
    Data augmentation specifically tailored for SFC-INRNet training.
    Provides robust spatial transformations to enhance the generalization of
    the spatial-frequency collaborative extraction process.
    """

    def __call__(self, input_img, target_mask):
        if random.random() < 0.5:  # Horizontal flip
            input_img = input_img[::-1, :]
            target_mask = target_mask[::-1, :]
        if random.random() < 0.5:  # Vertical flip
            input_img = input_img[:, ::-1]
            target_mask = target_mask[:, ::-1]
        if random.random() < 0.5:  # Transpose
            input_img = input_img.transpose(1, 0)
            target_mask = target_mask.transpose(1, 0)
        return input_img, target_mask


class SFC_TrainDataset(Dataset):
    """
    Dataloader for training the Spatial-Frequency Collaborative Implicit Representation Network (SFC-INRNet).
    Ensures precise patch extraction to preserve high-frequency target priors for the AFER module.
    """

    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(SFC_TrainDataset, self).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(dataset_dir, dataset_name)
        self.patch_size = patch_size

        list_path = os.path.join(self.dataset_dir, 'img_idx', f'train_{dataset_name}.txt')
        with open(list_path, 'r') as f:
            self.train_list = f.read().splitlines()

        if img_norm_cfg is None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

        self.transform = SFC_Augmentation()

    def __getitem__(self, idx):
        img_name = self.train_list[idx]
        img_path_png = os.path.join(self.dataset_dir, 'images', f'{img_name}.png')
        mask_path_png = os.path.join(self.dataset_dir, 'masks', f'{img_name}.png')
        img_path_bmp = os.path.join(self.dataset_dir, 'images', f'{img_name}.bmp')
        mask_path_bmp = os.path.join(self.dataset_dir, 'masks', f'{img_name}.bmp')

        # Robust cross-format loading for IRSTD datasets
        if os.path.exists(img_path_png):
            img = Image.open(img_path_png).convert('I')
            mask = Image.open(mask_path_png)
        else:
            img = Image.open(img_path_bmp).convert('I')
            mask = Image.open(mask_path_bmp)

        # Normalization
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        # Explicitly crop into fixed patch size (e.g., 256x256) as described in the paper
        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)

        # Apply spatial data augmentation
        img_patch, mask_patch = self.transform(img_patch, mask_patch)

        # Add channel dimension to simulate [C, H, W] for dual-domain encoder
        img_patch = img_patch[np.newaxis, :]
        mask_patch = mask_patch[np.newaxis, :]

        # Convert to continuous PyTorch Tensors
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)


class SFC_TestDataset(Dataset):
    """
    Dataloader for evaluating SFC-INRNet.
    Applies boundary padding to ensure full-resolution continuous coordinate mapping for the FAID module.
    """

    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(SFC_TestDataset, self).__init__()
        self.dataset_dir = os.path.join(dataset_dir, test_dataset_name)

        list_path = os.path.join(self.dataset_dir, 'img_idx', f'test_{test_dataset_name}.txt')
        with open(list_path, 'r') as f:
            self.test_list = f.read().splitlines()

        if img_norm_cfg is None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        img_name = self.test_list[idx]
        img_path_png = os.path.join(self.dataset_dir, 'images', f'{img_name}.png')
        mask_path_png = os.path.join(self.dataset_dir, 'masks', f'{img_name}.png')
        img_path_bmp = os.path.join(self.dataset_dir, 'images', f'{img_name}.bmp')
        mask_path_bmp = os.path.join(self.dataset_dir, 'masks', f'{img_name}.bmp')

        if os.path.exists(img_path_png):
            img = Image.open(img_path_png).convert('I')
            mask = Image.open(mask_path_png)
        else:
            img = Image.open(img_path_bmp).convert('I')
            mask = Image.open(mask_path_bmp)

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape

        # Pad image to prevent boundary degradation during explicit feature fusion
        img = PadImg(img)
        mask = PadImg(mask)

        img = img[np.newaxis, :]
        mask = mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        return img, mask, [h, w], img_name

    def __len__(self):
        return len(self.test_list)


class SFC_EvalDataset(Dataset):
    """
    Dataloader strictly for computing quantitative metrics (IoU, nIoU, Pd, Fa)
    between SFC-INRNet predictions and Ground Truths.
    """

    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(SFC_EvalDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name

        list_path = os.path.join(self.dataset_dir, 'img_idx', f'test_{test_dataset_name}.txt')
        with open(list_path, 'r') as f:
            self.test_list = f.read().splitlines()

    def __getitem__(self, idx):
        img_name = self.test_list[idx]

        pred_path = os.path.join(self.mask_pred_dir, self.test_dataset_name, self.model_name, f'{img_name}.png')
        gt_path = os.path.join(self.dataset_dir, 'masks', f'{img_name}.png')

        mask_pred = Image.open(pred_path)
        mask_gt = Image.open(gt_path)

        mask_pred = np.array(mask_pred, dtype=np.float32) / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32) / 255.0

        if len(mask_pred.shape) == 3:
            mask_pred = mask_pred[:, :, 0]

        h, w = mask_pred.shape

        mask_pred = mask_pred[np.newaxis, :]
        mask_gt = mask_gt[np.newaxis, :]

        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))

        return mask_pred, mask_gt, [h, w]

    def __len__(self):
        return len(self.test_list)