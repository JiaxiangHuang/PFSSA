# -*- coding: utf-8 -*-
"""
@File ：data_input.py
@Auth ：Jiaxiang Huang
@Time ：09/20/23 4:17 PM
"""
from __future__ import print_function
import os
import torch
import scipy.io
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from scipy import ndimage
import torch.nn.functional as F

def apply_data_augmentation(X_train, y_train, desired_total_samples):
    augmented_X_train = torch.empty((desired_total_samples, X_train.size()[1], X_train.size()[2], X_train.size()[3]))
    augmented_y_train = torch.empty((desired_total_samples, y_train.size()[1], y_train.size()[2], y_train.size()[3]))
    augmented_X_train[0:X_train.size()[0], :, :, :] = X_train[:]
    augmented_y_train[0:y_train.size()[0], :, :, :] = y_train[:]

    t = X_train.size()[0]
    while t < desired_total_samples:
        random_idx = random.randint(0, len(X_train) - 1)
        ori_patch = X_train[random_idx].numpy()
        ori_patch_gt = y_train[random_idx].numpy()

        num = random.randint(0, 2)
        if (num == 0):
            aug_patch = np.flip(ori_patch, 1)     # up and down
            aug_patch_gt = np.flip(ori_patch_gt, 1)
        if (num == 1):
            aug_patch = np.flip(ori_patch, 2)    # left and right
            aug_patch_gt = np.flip(ori_patch_gt, 2)
        if (num == 2):
            angle = random.choice([90, 180, 270])
            aug_patch = ndimage.rotate(ori_patch, angle, axes=(1,2))
            aug_patch_gt = ndimage.rotate(ori_patch_gt, angle, axes=(1,2))
        usq_aug_patch = torch.from_numpy(aug_patch.copy()).unsqueeze(0)
        usq_aug_patch_gt = torch.from_numpy(aug_patch_gt.copy()).unsqueeze(0)

        augmented_X_train[t:t+1, :, :, :] = usq_aug_patch[:]
        augmented_y_train[t:t+1, :, :, :] = usq_aug_patch_gt[:]
        t = t + 1

    return augmented_X_train, augmented_y_train


def padding_and_split(input_image, patch_size, stride, opt):
    data = input_image.squeeze(0)

    padding_left = opt.pad_left
    padding_top = opt.pad_top
    padding_right = opt.pad_right
    padding_bottom = opt.pad_bottom
    
    channels, original_height, original_width = data.shape
    new_height = original_height + padding_top + padding_bottom
    new_width = original_width + padding_left + padding_right

    padded_data = np.zeros((channels, new_height, new_width))
    padded_data[:, padding_top:padding_top + original_height, padding_left:padding_left + original_width] = data.numpy()

    padded_data[:, 0:padding_top, :] = padded_data[:, padding_top:2 * padding_top, :]
    padded_data[:, new_height-padding_bottom:new_height, :] = padded_data[:, new_height-2 * padding_bottom:new_height-padding_bottom, :]
    padded_data[:, :, 0:padding_left] = padded_data[:, :, padding_left:2 * padding_left]
    padded_data[:, :, new_width-padding_right:new_width] = padded_data[:, :, new_width-2 * padding_right:new_width-padding_right]

    image = torch.from_numpy(padded_data).unsqueeze(0)

    # splitting
    image_depth = image.size()[1]
    patch_height = patch_size
    patch_width = patch_size

    patches = F.unfold(image, (patch_height, patch_width), stride=stride)

    num_patches = patches.shape[-1]
    patches = patches.view(1, image_depth, patch_height, patch_width, num_patches)
    patches = patches.permute(0, 4, 1, 2, 3).contiguous().view(-1, image_depth, patch_height, patch_width)

    return patches


def get_dataloader(opt, batch_size, patch_size, train_set_ratio, valid_set_ratio, aug_times):
    root = opt.src_dir
    training_file = opt.training_file
    labels_file = opt.labels_file
    n_rows = opt.nrow_ori_img
    stride = opt.stride

    batch_size = batch_size
    patch_size = patch_size
    train_set_ratio = train_set_ratio
    valid_set_ratio = valid_set_ratio
    aug_times = aug_times

    root = os.path.expanduser(root)
    img_folder = 'Data_Matlab'
    gt_folder = 'GroundTruth'

    check_exists = os.path.exists(os.path.join(root, img_folder, training_file)) and os.path.exists(os.path.join(root, gt_folder, labels_file))
    if not check_exists:
        raise RuntimeError("Dataset not found." + " You can use 'https://rslab.ut.ac.ir/data' to download it")

    PATH = os.path.join(root, img_folder, training_file)
    PATH_L = os.path.join(root, gt_folder, labels_file)

    training_data = scipy.io.loadmat(PATH)
    labels = scipy.io.loadmat(PATH_L)

    traind = training_data['V'].T
    labs = labels['A'].T  # 9025,3

    img1d = torch.from_numpy(traind)
    label1d = torch.from_numpy(labs)

    n_spectral = img1d.shape[1]
    n_endm = label1d.shape[1]

    img2d = img1d.reshape(n_rows, -1, n_spectral).unsqueeze(0).permute(0, 3, 1, 2)

    # edge value padding
    patches_img = padding_and_split(img2d, patch_size, stride, opt)
    label2d = label1d.reshape(n_rows, -1, n_endm).unsqueeze(0).permute(0, 3, 1, 2)
    patches_label = padding_and_split(label2d, patch_size, stride, opt)

    # split data into train, valid, test
    num_all_samples = patches_img.size()[0]
    num_train = int(num_all_samples * train_set_ratio)
    num_valid = int(num_all_samples * valid_set_ratio)
    index_list = list(range(num_all_samples))
    np.random.seed(42)
    np.random.shuffle(index_list)
    train_idx, valid_idx, test_idx = index_list[:num_train], index_list[num_train: (num_train + num_valid)], index_list[(num_train + num_valid):]

    X_train = patches_img[train_idx, :, :, :]
    y_train = patches_label[train_idx, :, :, :]
    X_valid = patches_img[valid_idx, :, :, :]
    y_valid = patches_label[valid_idx, :, :, :]
    X_test = patches_img[test_idx, :, :, :]
    y_test = patches_label[test_idx, :, :, :]

    all_set = TensorDataset(torch.Tensor(patches_img), torch.Tensor(patches_label))
    test_set = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    loader_all = DataLoader(all_set, batch_size=len(all_set), shuffle=False, num_workers=0, pin_memory=True)
    loader_test = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    if X_valid.size()[0] != 0:
        valid_set = TensorDataset(torch.Tensor(X_valid), torch.Tensor(y_valid))
        loader_valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    else:
        print('valid_set_ratio==0')
        loader_valid = None

    # augment training sample
    aug_X_train, aug_y_train = apply_data_augmentation(X_train, y_train, int(aug_times * X_train.size()[0]))
    train_set = TensorDataset(torch.Tensor(aug_X_train), torch.Tensor(aug_y_train))
    loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True) #, num_workers=16, pin_memory=True

    return loader_train, loader_valid, loader_test, loader_all
