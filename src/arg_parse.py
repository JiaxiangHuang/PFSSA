# -*- coding: utf-8 -*-
"""
@File ：PFSSA.py
@Auth ：Jiaxiang Huang
@Time ：11/20/23 4:17 PM
"""
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Args ')

    parser.add_argument('--src_dir', '-src_dir', type=str, required=True,
                        help='System path to the data directory.')
    parser.add_argument('--img_name', '-img_name', type=str, required=True,
                        help="Image name for save results.")
    parser.add_argument('--training_file', '-training_file', type=str, required=True,
                        help="Hyperspectral image")
    parser.add_argument('--labels_file', '-labels_file', type=str, required=True,
                        help='Abundance ground truth.')
    parser.add_argument('--nrow_ori_img', '-nrow_ori_img', type=int, required=True,
                        help='Number of rows of the hyperspectral image. '
                             'Samson=95, JaperRidge=100, Urban=307, Spheric=128')
    parser.add_argument('--ncol_ori_img', '-ncol_ori_img', type=int, required=True,
                        help='Number of columns of the hyperspectral image. '
                             'Samson=95, JaperRidge=100, Urban=307, Spheric=128')
    parser.add_argument('--num_bands', '-num_bands', type=int, required=True,
                        help='Number of num bands of the hyperspectral image. '
                             'Samson=156, JaperRidge=198, Urban=162, Spheric=431')
    parser.add_argument('--end_members', '-end_members', type=int, required=True,
                        help='Number of end members of the hyperspectral image. '
                             'Samson=3, JaperRidge=4, Urban=4, Spheric=5')
    parser.add_argument('--pad_left', '-pad_left', type=int, required=True,
                        help='Number of columns padding on the left for image before splitting. pad_top=pad_left.'
                             'Samson=0, JaperRidge=0, Urban=0, Spheric=0')
    parser.add_argument('--pad_right', '-pad_right', type=int, required=True,
                        help='Number of columns padding on the right for image before splitting. pad_bottom=pad_right'
                             'Samson=1, JaperRidge=0, Urban=1, Spheric=0')

    parser.add_argument('--epochs', '-epochs', type=int, default=500,
                        help="Total epochs for training.")
    parser.add_argument('--batch_size', '-batch_size', type=int, default=32,
                        help="Batch size.")
    parser.add_argument('--lr', '-lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--train_set_ratio', '-train_set_ratio', type=float, default=0.2,
                        help='Ratio of samples for training.')
    parser.add_argument('--valid_set_ratio', '-valid_set_ratio', type=float, default=0.1,
                        help='Ratio of samples for validation.')
    parser.add_argument('--aug_times', '-aug_times', type=int, default=5,
                        help="Times for augument training sample.")
    parser.add_argument('--patch_size', '-patch_size', type=int, default=4,
                        help="Patch size for splitting hyperspectral image into patches.")
    parser.add_argument('--stride', '-stride', type=int, default=4,
                        help='Stride for splitting hyperspectral image into patches.')
    parser.add_argument('--gamma', '-gamma', type=float, default=0.8,
                        help='Values of gamma for optim.lr_scheduler.')
    parser.add_argument('--pca_out_dim', '-pca_out_dim', type=int, default=64,
                        help='Out dimension for PCA.')
    parser.add_argument('--weight', '-weight', type=float, default=0.2,
                        help='Weight of loss.')

    parser.add_argument('--threshold', '-threshold', type=float, default=1.0,
                        help='Defines the threshold for the softplus activation function.')
    parser.add_argument('--model_name', '-model_name', type=str, default='PFSSA',
                        help="Model for training.")
    parser.add_argument('--planes', '-planes', type=int, default=64,
                        help="Attention network planes.")
    parser.add_argument('--save_ckpt_dir', '-save_ckpt_dir', type=str, default='../checkpoints',
                        help='System path for saving checkpoints.')
    parser.add_argument('--save_ckpt_freq', '-save_ckpt_freq', type=int, default=500,
                        help='Frequences (epochs) for saving checkpoints.')
    parser.add_argument('--print_freq', '-print_freq', type=int, default=20,
                        help='Frequences (epochs) for output print.')

    opt = parser.parse_args()

    return opt
