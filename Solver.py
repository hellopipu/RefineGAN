# @project      : Pytorch implementation of RefineGAN
# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers
# @Code         : https://github.com/hellopipu/RefineGAN

import os
import torch
import torch.optim as optim
from RefineGAN import Refine_G, Discriminator
from read_data import MyData
from torch.optim.lr_scheduler import LinearLR
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from datetime import datetime
from loss import cal_loss
from utils import cal_psnr, output2complex
from os.path import join
import skimage.io
import numpy as np
import matplotlib.pyplot as plt


class Solver():
    def __init__(self, args):
        self.args = args

        ################  experiment settings  ################
        self.data_name = 'brain'
        self.mask_name = self.args.mask_type  # mask type
        self.sampling_rate = self.args.sampling_rate  # sampling rate, 10, 20 ,30 ,40 ...
        self.imageDir_train = self.args.train_path  # train path
        self.imageDir_val = self.args.val_path  # val path while training
        self.num_epoch = self.args.num_epoch
        self.batch_size = self.args.batch_size  # batch size
        self.maskDir = 'data/mask/' + self.mask_name + '/mask_' + str(
            self.sampling_rate // 10) + '/'  # different sampling rate
        self.saveDir = 'weight'  # model save path while training
        if not os.path.isdir(self.saveDir):
            os.makedirs(self.saveDir)

        self.imageDir_test = self.args.test_path  # test path
        self.model_path = 'weight/' + self.data_name + '_' + self.mask_name + '_' + str(
            self.sampling_rate) + '_' + 'best.pth'  # model load path for test and visualization
        ###################################################################################

    def train(self):

        ############################################ Specify network ############################################
        G = Refine_G()
        D = Discriminator()
        optimizer_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999), eps=1e-3, weight_decay=1e-10)
        optimizer_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999), eps=1e-3, weight_decay=1e-10)
        scheduler_lr_G = LinearLR(optimizer_G, start_factor=1., end_factor=6e-2, total_iters=500)
        scheduler_lr_D = LinearLR(optimizer_G, start_factor=1., end_factor=6e-2, total_iters=500)

        G.cuda()
        D.cuda()
        ############################################ load data ############################################

        dataset_train = MyData(self.imageDir_train, self.maskDir, is_training=True)
        loader_train = Data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                       num_workers=4, pin_memory=True)
        dataset_val = MyData(self.imageDir_val, self.maskDir, is_training=False)
        loader_val = Data.DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                     num_workers=4, pin_memory=True)

        print("slices of 2d train data: ", len(dataset_train))
        print("slices of 2d validation data: ", len(dataset_val))

        ############################################ setting for tensorboard ############################################
        TIMESTAMP = self.data_name + '_' + self.mask_name + '_' + str(
            self.sampling_rate) + "_{0:%Y-%m-%dT%H-%M-%S/}".format(
            datetime.now())
        writer = SummaryWriter('log/' + TIMESTAMP)

        ############################################ start to run epochs ############################################

        start_epoch = 0
        weight_cliping_limit = 0.01
        D.train()
        best_val_psnr = 0
        for epoch in range(start_epoch, self.num_epoch):
            ####################### 1. training #######################
            G.train()
            for data_dict in loader_train:
                im_A, im_A_und, k_A_und, im_B, im_B_und, k_B_und, mask = data_dict['im_A'].float().cuda(), data_dict[
                    'im_A_und'].float().cuda(), data_dict['k_A_und'].float().cuda(), \
                                                                         data_dict['im_B'].float().cuda(), data_dict[
                                                                             'im_B_und'].float().cuda(), data_dict[
                                                                             'k_B_und'].float().cuda(), data_dict[
                                                                             'mask'].float().cuda()
                for p in D.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update

                Sp1, S1, Tp1, T1 = G(im_A_und, k_A_und, mask)
                Sp2, S2, Tp2, T2 = G(im_B_und, k_B_und, mask)
                S1_dis_real = D(im_A)
                S2_dis_real = D(im_B)

                ############################################# 1.1 update discriminator #############################################
                with torch.no_grad():
                    S1_clone = S1.detach().clone()
                    T1_clone = T1.detach().clone()
                    S2_clone = S2.detach().clone()
                    T2_clone = T2.detach().clone()

                S1_dis_fake = D(S1_clone)
                T1_dis_fake = D(T1_clone)
                S2_dis_fake = D(S2_clone)
                T2_dis_fake = D(T2_clone)

                loss_d, d_loss_list = cal_loss(im_A, k_A_und, im_B, k_B_und, mask, Sp1, S1, Tp1, T1, Sp2, S2, Tp2, T2,
                                               S1_dis_real, S1_dis_fake, T1_dis_fake, S2_dis_real, S2_dis_fake,
                                               T2_dis_fake, cal_G=False)
                optimizer_D.zero_grad()
                loss_d.backward()
                optimizer_D.step()

                # Clamp parameters to a range [-c, c], c=self.weight_cliping_limit , for WGAN
                for p in D.parameters():
                    p.data.clamp_(-weight_cliping_limit, weight_cliping_limit)

                ############################################# 1.2 update generator #############################################
                for p in D.parameters():  # reset requires_grad
                    p.requires_grad = False  # they are set to False below in netG update

                S1_dis_fake = D(S1)
                T1_dis_fake = D(T1)
                S2_dis_fake = D(S2)
                T2_dis_fake = D(T2)
                loss_g, g_loss_list = cal_loss(im_A, k_A_und, im_B, k_B_und, mask, Sp1, S1, Tp1, T1, Sp2, S2, Tp2, T2,
                                               S1_dis_real, S1_dis_fake, T1_dis_fake, S2_dis_real, S2_dis_fake,
                                               T2_dis_fake, cal_G=True)
                optimizer_G.zero_grad()
                loss_g.backward()
                optimizer_G.step()
            scheduler_lr_G.step()
            scheduler_lr_D.step()
            ####################### 2. validate #######################
            base_psnr = 0
            test_psnr = 0
            G.eval()
            with torch.no_grad():
                for data_dict in loader_val:
                    im_A, im_A_und, k_A_und, mask = data_dict['im_A'].float().cuda(), data_dict[
                        'im_A_und'].float().cuda(), \
                                                    data_dict['k_A_und'].float().cuda(), \
                                                    data_dict['mask'].float().cuda()
                    Sp1, S1, Tp1, T1 = G(im_A_und, k_A_und, mask)

                    ############## convert model ouput to complex value in original range
                    T1 = output2complex(T1)
                    im_A = output2complex(im_A)
                    im_A_und = output2complex(im_A_und)

                    ########################## 2.1 cal psnr for validation ###################################

                    base_psnr += cal_psnr(im_A, im_A_und, maxp=1.)
                    test_psnr += cal_psnr(im_A, T1, maxp=1.)

            base_psnr /= len(dataset_val)
            test_psnr /= len(dataset_val)
            ## save the best model according to validation psnr
            if best_val_psnr < test_psnr:
                best_val_psnr = test_psnr
                best_name = self.data_name + '_' + self.mask_name + '_' + str(self.sampling_rate) + '_' + 'best.pth'
                state = {'net': G.state_dict(), 'net_D': D.state_dict(), 'start_epoch': epoch, 'psnr': test_psnr}
                torch.save(state, join(self.saveDir, best_name))
            ########################## 3. print and tensorboard ########################
            print("Epoch {}/{}".format(epoch + 1, self.num_epoch))
            print(" base PSNR:\t\t{:.6f}".format(base_psnr))
            print(" test PSNR:\t\t{:.6f}".format(test_psnr))
            ## write to tensorboard
            writer.add_scalar("loss/G_loss_total", loss_g, epoch)
            writer.add_scalar("loss/D_loss_total", loss_d, epoch)
            writer.add_scalar("loss/G_loss_AA", g_loss_list[0], epoch)
            writer.add_scalar("loss/G_loss_Aa", g_loss_list[1], epoch)
            writer.add_scalar("loss/recon_img_AA", g_loss_list[2], epoch)
            writer.add_scalar("loss/recon_img_Aa", g_loss_list[3], epoch)
            writer.add_scalar("loss/error_img_AA", g_loss_list[4], epoch)
            writer.add_scalar("loss/error_img_Aa", g_loss_list[5], epoch)
            writer.add_scalar("loss/recon_frq_AA", g_loss_list[6], epoch)
            writer.add_scalar("loss/recon_frq_Aa", g_loss_list[7], epoch)
            writer.add_scalar("loss/smoothness_AA", g_loss_list[8], epoch)
            writer.add_scalar("loss/smoothness_Aa", g_loss_list[9], epoch)
            writer.add_scalar("loss/D_loss_AA", d_loss_list[0], epoch)
            writer.add_scalar("loss/D_loss_Aa", d_loss_list[1], epoch)
            writer.add_scalar("loss/D_loss_AB", d_loss_list[2], epoch)
            writer.add_scalar("loss/D_loss_Ab", d_loss_list[3], epoch)
            writer.add_scalar("loss/base_psnr", base_psnr, epoch)
            writer.add_scalar("loss/test_psnr", test_psnr, epoch)

        writer.close()

    def test(self):

        ############################################ load data ################################
        dataset_val = MyData(self.imageDir_test, self.maskDir, is_training=False)
        loader_val = Data.DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                     num_workers=4, pin_memory=True)
        len_data = len(dataset_val)
        print("slices of 2d validation data: ", len_data)

        ####################### load model #######################
        G = Refine_G()
        G.load_state_dict(torch.load(self.model_path)['net'])
        G.cuda()
        G.eval()
        #######################validate
        base_psnr = 0
        test_psnr = 0
        with torch.no_grad():
            for data_dict in loader_val:
                im_A, im_A_und, k_A_und, mask = data_dict['im_A'].float().cuda(), data_dict['im_A_und'].float().cuda(), \
                                                data_dict['k_A_und'].float().cuda(), \
                                                data_dict['mask'].float().cuda()
                Sp1, S1, Tp1, T1 = G(im_A_und, k_A_und, mask)
                ############## convert model ouput to complex value in original range
                T1 = output2complex(T1)
                im_A = output2complex(im_A)
                im_A_und = output2complex(im_A_und)

                ########################## cal psnr ###################################

                base_psnr += cal_psnr(im_A, im_A_und, maxp=1.)
                test_psnr += cal_psnr(im_A, T1, maxp=1.)

        base_psnr /= len_data
        test_psnr /= len_data

        print(" base PSNR:\t\t{:.6f}".format(base_psnr))
        print(" test PSNR:\t\t{:.6f}".format(test_psnr))

    def visualize(self):

        sampling_rate_list = [10, 20, 30, 40]
        ####################### evaluate #######################

        for sampling_rate in sampling_rate_list:
            ############################################ load data ################################
            maskDir = 'data/mask/' + self.mask_name + '/mask_' + str(
                sampling_rate // 10) + '/'  # different sampling rate
            model_path = 'weight/' + self.data_name + '_' + self.mask_name + '_' + str(
                sampling_rate) + '_' + 'best.pth'  # model load path for test and visualization

            dataset_val = MyData(self.imageDir_test, maskDir, is_training=False)
            loader_val = Data.DataLoader(dataset_val, batch_size=1, shuffle=False, drop_last=False,
                                         num_workers=4, pin_memory=True)
            data_dict = next(iter(loader_val))

            ####################### load model #######################
            G = Refine_G()
            G.load_state_dict(torch.load(model_path)['net'])
            G.cuda()
            G.eval()
            with torch.no_grad():
                im_A, im_A_und, k_A_und, mask = data_dict['im_A'].float().cuda(), data_dict['im_A_und'].float().cuda(), \
                                                data_dict['k_A_und'].float().cuda(), \
                                                data_dict['mask'].float().cuda()
                Sp1, S1, Tp1, T1 = G(im_A_und, k_A_und, mask)
                ############## convert model ouput to complex value in original range
                T1 = output2complex(T1)
                im_A = output2complex(im_A)
                im_A_und = output2complex(im_A_und)

                ########################## cal psnr ###################################
                base_psnr = cal_psnr(im_A, im_A_und, maxp=1.)
                test_psnr = cal_psnr(im_A, T1, maxp=1.)
                print('sampling_rate: ', sampling_rate)
                print(" base PSNR:\t\t{:.6f}".format(base_psnr))
                print(" test PSNR:\t\t{:.6f}".format(test_psnr))
                ######################### visualization ###############################

                mask = mask[0, 0].cpu().numpy()
                T1 = T1.abs()[0].cpu().numpy()
                im_A = im_A.abs()[0].cpu().numpy()
                im_A_und = im_A_und.abs()[0].cpu().numpy()
                error_und = abs(im_A_und - im_A)
                err_T1 = abs(T1 - im_A)

                np_visual_stack = np.hstack((mask, im_A_und, T1,
                                             im_A))
                np_visual_stack_error = np.hstack((error_und, err_T1))

                plt.imsave('{}_{}_error.png'.format(self.mask_name, sampling_rate), np_visual_stack_error,vmin=0,vmax=1.)
                plt.imsave('{}_{}.png'.format(self.mask_name, sampling_rate), np_visual_stack, vmin=0,vmax=1., cmap='gray')
                # skimage.io.imsave('{}_{}.png'.format(self.mask_name, sampling_rate), np_visual_stack)
