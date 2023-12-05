# -*- coding: utf-8 -*-
"""
@File ：PFSSA.py
@Auth ：Jiaxiang Huang
@Time ：11/20/23 4:17 PM
"""

from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_input import get_dataloader
from arg_parse import get_args
from model import PFSSA
import numpy as np
import random
import matplotlib.pyplot as plt


def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ---------Abundance Angle Distance------------
class AAD(nn.Module):
    def __init__(self):
        super(AAD, self).__init__()

    def forward(self, input, target):
        num_e = input.size()[1]  # 3

        input0 = input.permute(0, 2, 3, 1).contiguous()  # [20,12,12,3]
        input1 = input0.view(-1, num_e)  # [2280,3]
        a = input1.unsqueeze(1)  # [2880,1,3]
        b = input1.unsqueeze(2)  # [2880,3,1]
        input_norm = torch.sqrt(torch.bmm(a, b))

        target0 = target.permute(0, 2, 3, 1).contiguous()
        target1 = target0.view(-1, num_e)
        c = target1.unsqueeze(1)  # [2880,1,3]
        d = target1.unsqueeze(2)  # [2880,3,1]
        target_norm = torch.sqrt(torch.bmm(c, d))

        summation = torch.bmm(a, d)
        denominator = input_norm * target_norm

        try:
            value = summation / denominator
        except ZeroDivisionError:
            print("You can't divide by zero!")

        epsilon = 1e-7
        angle = torch.acos(torch.clamp(value, -1 + epsilon, 1 - epsilon))

        return angle


def stable(dataloader, seed):
    seed_torch(seed)
    return dataloader


# ------------------Training-------------------
def pfssa_train(model, train_dataloader, optimizer, opt, seed, epoch, device, patch_size, weight):
    model.train()
    batches_loss = 0.0
    batches_loss_rmse=  0.0
    batches_loss_aad_rmse=  0.0
    total_trian_loss_aad_sum = 0.0
    num_patches_train = 0
    for batch_idx, (X, abd_GT) in enumerate(stable(train_dataloader, seed + epoch)):
        X = X.to(device, non_blocking=True).float()
        abd_GT = abd_GT.to(device, non_blocking=True).float()
        cnn_out = model(X)
        # rmse
        c_mse = torch.nn.MSELoss(reduction='mean')
        loss = c_mse(cnn_out, abd_GT)
        loss_rmse = loss.sqrt()
        # loss_aad_sum
        c_aad = AAD()
        loss_aad_vec = c_aad(cnn_out, abd_GT)
        loss_aad_sum = torch.sum(loss_aad_vec).float()
        # loss_aad_mse1
        sqz_vec = torch.squeeze(loss_aad_vec)  # 2880
        zero_vec = torch.zeros_like(sqz_vec)
        loss_aad_rmse1 = c_mse(sqz_vec, zero_vec).sqrt()  # [0,pi]
        # loss_aad_mse2
        sp = sqz_vec.pow(2)
        spmean = sp.mean()
        loss_aad_rmse = spmean.sqrt()
        loss1= (1-weight)*loss_rmse + weight*loss_aad_rmse

        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        this_batch_size = abd_GT.size()[0]
        batches_loss += loss.item() * this_batch_size
        batches_loss_rmse += loss_rmse.item() * this_batch_size
        batches_loss_aad_rmse += loss_aad_rmse.item() * this_batch_size
        total_trian_loss_aad_sum  += loss_aad_sum.item()
        num_patches_train += this_batch_size

    train_loss = batches_loss / num_patches_train
    train_loss_rmse = batches_loss_rmse / num_patches_train
    train_loss_aad_rmse = batches_loss_aad_rmse / num_patches_train
    train_loss_aad_avg = total_trian_loss_aad_sum / (num_patches_train * patch_size * patch_size) #per pixel [0,pai]
    return train_loss, train_loss_rmse, train_loss_aad_rmse, train_loss_aad_avg


# --------------Testing-------------------
def pfssa_test(model, test_loader, opt, seed, device, patch_size):
    model.eval()
    num_patches_test = 0
    batches_test_loss = 0
    batches_test_loss_rmse = 0
    batches_test_loss_aad_rmse = 0
    total_test_loss_aad_sum = 0
    ems_loss = np.zeros(opt.end_members)
    ems_loss_rmse = np.zeros(opt.end_members)

    with torch.no_grad():
        for data, target in (stable(test_loader, seed)):
            data, target = data.to(device, non_blocking=True).float(), target.to(device, non_blocking=True).float()
            output = model(data.float())
            this_batch_size = target.size(0)
            # ---------loss-----loss_rmse--------------
            c_mse = nn.MSELoss(reduction='mean')
            loss = c_mse(output, target)
            loss_rmse = loss.sqrt()
            # total loss of this batch
            batches_test_loss += loss.item() * this_batch_size
            batches_test_loss_rmse += loss_rmse.item() * this_batch_size
            # ---------loss_aad_sum-------------------
            # loss_aad_sum
            c_aad = AAD()
            loss_aad_vec = c_aad(output, target)
            loss_aad_sum = torch.sum(loss_aad_vec).float()
            total_test_loss_aad_sum += loss_aad_sum.item()
            # ---------loss_aad_mse-------------------
            # loss_aad_mse1
            sqz_vec = torch.squeeze(loss_aad_vec)  # 2880
            zero_vec = torch.zeros_like(sqz_vec)
            loss_aad_rmse1 = c_mse(sqz_vec, zero_vec).sqrt()  # [0,pi]
            # loss_aad_mse2
            sp = sqz_vec.pow(2)
            spmean = sp.mean()
            loss_aad_rmse = spmean.sqrt()
            batches_test_loss_aad_rmse += loss_aad_rmse.item() * this_batch_size
            # --count number of samples, i.e. patches-
            num_patches_test += this_batch_size

            for k in range(0, opt.end_members):
                c_mse = nn.MSELoss(reduction='mean')
                e_loss = c_mse(output[:,k,:,:], target[:,k,:,:])
                e_loss_rmse = e_loss.sqrt()
                ems_loss[k] +=  e_loss.item() * this_batch_size
                ems_loss_rmse[k] +=  e_loss_rmse.item() * this_batch_size

        test_loss = batches_test_loss / num_patches_test
        test_loss_rmse = batches_test_loss_rmse / num_patches_test
        test_loss_aad_rmse = batches_test_loss_aad_rmse / num_patches_test
        test_loss_aad_avg = total_test_loss_aad_sum / (num_patches_test * patch_size*patch_size)
        test_ems_loss = ems_loss / num_patches_test
        test_ems_loss_rmse = ems_loss_rmse / num_patches_test
    return test_loss, test_loss_rmse, test_loss_aad_rmse, test_loss_aad_avg, test_ems_loss, test_ems_loss_rmse


def plot_abundances_map(num_endmembers, abundances, gt, figure_nr=10, Title=None):
    if Title is not None:
        st = plt.suptitle(Title)
    fig, axs = plt.subplots(2, num_endmembers, figsize=(15, 8))
    for i in range(num_endmembers):
        axs[0, i].imshow(gt[i, :, :], cmap='jet')
        axs[1, i].imshow(abundances[i, :, :], cmap='jet')
    plt.tight_layout()
    if Title is not None:
        st.set_y(0.95)
        fig.subplots_adjust(top=0.88)
    plt.show()


def join_abundances_map(model, all_loader, opt, seed, device, patch_size):
    model.eval()
    img = torch.zeros((opt.end_members, opt.nrow_ori_img, opt.ncol_ori_img))
    img_pad = torch.zeros((opt.end_members, opt.nrow_ori_img+opt.pad_left+opt.pad_right, opt.ncol_ori_img+opt.pad_top+opt.pad_bottom)) #########

    p_size = [1, patch_size, patch_size]
    num_patches_col = int((opt.ncol_ori_img+ opt.pad_left+ opt.pad_right)/patch_size)
    reconstructed_img = torch.Tensor(np.zeros_like(img_pad, dtype=float)).to(device)
    reconstructed_gt = torch.Tensor(np.zeros_like(img_pad, dtype=float)).to(device)

    with torch.no_grad():
        for data, target in (stable(all_loader, seed)):
            data, target = data.to(device).float(), target.to(device).float()
            output = model(data.float())
            # patches to abundance map
            patches_tensor = output
            for i in range(num_patches_col):
                for j in range(num_patches_col):
                    reconstructed_img[:, i*p_size[1]:(i +1)* p_size[1], j*p_size[2]:(j + 1)*p_size[2]] = patches_tensor[i * num_patches_col + j, :, :, :]
                    reconstructed_gt[:, i*p_size[1]:(i +1)* p_size[1], j*p_size[2]:(j + 1)*p_size[2]] = target[i * num_patches_col + j, :, :, :]

    # crop
    pad_top = opt.pad_top
    reconstructed_img = reconstructed_img[:, pad_top:pad_top+opt.nrow_ori_img, opt.pad_left:opt.pad_left+opt.ncol_ori_img]
    reconstructed_gt = reconstructed_gt[:, pad_top:pad_top+opt.nrow_ori_img, opt.pad_left:opt.pad_left+opt.ncol_ori_img]

    xr = torch.rot90(reconstructed_img, k=3, dims=[1, 2])
    xr = xr.cpu().numpy()
    x = np.flip(xr, 2)
    xr_t = torch.rot90(reconstructed_gt, k=3, dims=[1, 2])
    xr_t = xr_t.cpu().numpy()
    x_t = np.flip(xr_t, 2)

    abundances = x
    plot_abundances_map(opt.end_members, x, x_t,figure_nr=10, Title=None)
    return abundances


def main():
    seed = 42
    seed_torch(seed)
    opt = get_args();
    epochs = opt.epochs
    batch_size = opt.batch_size
    lr =opt.lr
    train_set_ratio = opt.train_set_ratio
    valid_set_ratio = opt.valid_set_ratio
    aug_times = opt.aug_times
    gamma = opt.gamma
    patch_size = opt.patch_size
    weight = opt.weight

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opt.model_name == 'PFSSA':
        model = PFSSA(opt.num_bands, opt.threshold, opt.end_members, opt.planes, opt.pca_out_dim)
        print('Model is PFSSA.')
    else:
        print('No Model.')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=gamma)

    # Dataloader
    print("\n=============Start load data=============")
    print('Dataset is: %s' %opt.src_dir)
    train_dataloader, valid_loader, test_loader, all_loader = get_dataloader(opt, batch_size, patch_size, train_set_ratio, valid_set_ratio, aug_times)
    print('Dataloader finished!')

    # Training
    print("\n=============Start training==============")
    for epoch in range(epochs):
        train_loss, train_loss_rmse, train_loss_aad_rmse, train_loss_aad_avg = pfssa_train(model, train_dataloader,
                                                                                           optimizer, opt, seed, epoch, device, patch_size, weight)
        # valid_loss, valid_loss_rmse, valid_loss_aad_rmse, valid_loss_aad_avg,valid_ems_loss, valid_ems_loss_rmse = pfssa_test(model, valid_loader,
        #                                                                                     opt, seed, epoch, device, patch_size)
        scheduler.step()
        # Save checkpoint
        if ((epoch + 1) % opt.save_ckpt_freq == 0) | ((epoch + 1) % epochs == 0):
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f"{opt.save_ckpt_dir}/{opt.model_name}_{opt.img_name}_{epoch + 1}.pt")
        # Print
        if ((epoch + 1) % opt.print_freq == 0) | ((epoch + 1) % epochs == 0):
            print(f'\nEpoch {epoch + 1:04d} / {epochs:04d}', end='\n-----------------\n')
            # print("Train_loss: %.4f" % (train_loss))
            print("Train_loss_rmse: %.4f" % (train_loss_rmse))
            print("Train_loss_aad_rmse: %.4f" % (train_loss_aad_rmse))
            print("Train_loss_aad_avg: %.4f" % (train_loss_aad_avg))

    print("\n=============Start testing===============")
    # Load checkpoint
    checkpoint_dir = opt.save_ckpt_dir
    checkpoint_file = f"{opt.model_name}_{opt.img_name}_{epochs}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Checkpoint loaded!')

    # Testing
    test_loss, test_loss_rmse, test_loss_aad_rmse, test_loss_aad_avg,test_ems_loss, test_ems_loss_rmse = pfssa_test(model, test_loader,
                                                                                            opt, seed, device, patch_size)
    # Save loss
    np.savez('loss-' + opt.img_name + '-' + opt.model_name, test_loss=test_loss, test_loss_rmse=test_loss_rmse,
             test_loss_aad_rmse=test_loss_aad_rmse, test_loss_aad_avg=test_loss_aad_avg,
             test_ems_loss=test_ems_loss, test_ems_loss_rmse=test_ems_loss_rmse)

    for i in range(0, opt.end_members):
        print("Test_ems_loss_rmse of %d end_m: %.4f" % (i, test_ems_loss_rmse[i]))
    print("Test_loss_rmse: %.4f" % (test_loss_rmse))
    print("Test_loss_aad_rmse: %.4f" % (test_loss_aad_rmse))
    print("Test_loss_aad_avg: %.4f" % (test_loss_aad_avg))

    # Join abundance map and save
    print("\n=============Start plot abundance========")
    abundances = join_abundances_map(model, all_loader, opt, seed, device, patch_size)
    np.savez('abd-' + opt.img_name + '-'+opt.model_name, abd=abundances)
    print("Plot and save finished!")


if __name__ == '__main__':
    # -----option 1-----
    main()

    # -----option 2: Time-----
    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp_wrapper = lp(main)
    # lp_wrapper()
    # lp.print_stats()




