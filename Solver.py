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
from utils import cal_psnr
from os.path import join


class Solver():
    def __init__(self, args):
        self.args = args

    def train(self):
        ################  experiment settings  ################
        data_name = 'brain'
        mask_name = self.args.mask_type  # mask type
        sampling_rate = self.args.sampling_rate  # sampling rate, 10, 20 ,30 ,40 ...
        imageDir_train = self.args.train_path  # train path
        imageDir_val = self.args.val_path  # val path
        batch_size = self.args.batch_size  # batch size
        saveDir = 'weight'  # model save path
        maskDir = 'data/mask/' + mask_name + '/mask_' + str(sampling_rate // 10) + '/'  # different sampling rate
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)
        ###################################################################################

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

        img_size = 256
        dataset_train = MyData(imageDir_train, maskDir, is_training=True)
        loader_train = Data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True,
                                       num_workers=4, pin_memory=True)
        dataset_val = MyData(imageDir_val, maskDir, is_training=False)
        loader_val = Data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False,
                                     num_workers=4, pin_memory=True)

        print("slices of 2d train data: ", len(dataset_train))
        print("slices of 2d validation data: ", len(dataset_val))

        ############################################ setting for tensorboard ############################################
        TIMESTAMP = data_name + '_' + mask_name + '_' + str(sampling_rate) + "_{0:%Y-%m-%dT%H-%M-%S/}".format(
            datetime.now())
        writer = SummaryWriter('log/' + TIMESTAMP)

        ############################################ start to run epochs ############################################

        start_epoch = 0
        num_epoch = self.args.num_epoch
        weight_cliping_limit = 0.01
        D.train()
        best_val_psnr = 0
        for epoch in range(start_epoch, num_epoch):
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
                    T1 = (T1 / 2.0 + 0.5) * 255.0
                    im_A = (im_A / 2.0 + 0.5) * 255.0
                    im_A_und = (im_A_und / 2.0 + 0.5) * 255.0

                    T1 = torch.view_as_complex(T1.permute(0, 2, 3, 1).contiguous()).cpu()
                    im_A = torch.view_as_complex(im_A.permute(0, 2, 3, 1).contiguous()).cpu()
                    im_A_und = torch.view_as_complex(im_A_und.permute(0, 2, 3, 1).contiguous()).cpu()

                    ########################## 2.1 cal psnr for validation ###################################
                    for im_i, und_i, pred_i in zip(im_A,
                                                   im_A_und,
                                                   T1):
                        base_psnr += cal_psnr(im_i, und_i)
                        test_psnr += cal_psnr(im_i, pred_i)

            base_psnr /= len(dataset_val)
            test_psnr /= len(dataset_val)
            ## save the best model according to validation psnr
            if best_val_psnr < test_psnr:
                best_val_psnr = test_psnr
                best_name = data_name + '_' + mask_name + '_' + str(sampling_rate) + '_' + 'best.pth'
                state = {'net': G.state_dict(), 'net_D': D.state_dict(), 'start_epoch': epoch, 'psnr': test_psnr}
                torch.save(state, join(saveDir, best_name))
            ########################## 3. print and tensorboard ########################
            print("Epoch {}/{}".format(epoch + 1, num_epoch))
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
        ################  experiment settings  ################
        data_name = 'brain'
        mask_name = self.args.mask_type  # mask type
        sampling_rate = self.args.sampling_rate  # sampling rate, 10, 20 ,30 ,40 ...
        imageDir_test = self.args.test_path  # test path
        batch_size = self.args.batch_size  # batch size
        model_path = 'weight/' + data_name + '_' + mask_name + '_' + str(
            sampling_rate) + '_' + 'best.pth'  # model path to load
        maskDir = 'data/mask/' + mask_name + '/mask_' + str(sampling_rate // 10) + '/'  # different sampling rate

        ###################################################################################

        ############################################ load data ################################
        dataset_val = MyData(imageDir_test, maskDir, is_training=False)
        loader_val = Data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False,
                                     num_workers=4, pin_memory=True)
        len_data = len(dataset_val)
        print("slices of 2d validation data: ", len_data)

        ####################### load model #######################
        G = Refine_G()
        G.load_state_dict(torch.load(model_path)['net'])
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
                T1 = (T1 / 2.0 + 0.5) * 255.0
                im_A = (im_A / 2.0 + 0.5) * 255.0
                im_A_und = (im_A_und / 2.0 + 0.5) * 255.0
                T1 = torch.view_as_complex(T1.permute(0, 2, 3, 1).contiguous()).cpu()
                im_A = torch.view_as_complex(im_A.permute(0, 2, 3, 1).contiguous()).cpu()
                im_A_und = torch.view_as_complex(im_A_und.permute(0, 2, 3, 1).contiguous()).cpu()

                for im_i, und_i, pred_i in zip(im_A, im_A_und, T1):
                    base_psnr += cal_psnr(im_i, und_i)
                    test_psnr += cal_psnr(im_i, pred_i)

        base_psnr /= len_data
        test_psnr /= len_data

        print(" base PSNR:\t\t{:.6f}".format(base_psnr))
        print(" test PSNR:\t\t{:.6f}".format(test_psnr))
