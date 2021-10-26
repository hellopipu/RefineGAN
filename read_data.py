# @project      : Pytorch implementation of RefineGAN
# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers
# @Code         : https://github.com/hellopipu/RefineGAN

from os.path import join as join
import glob
import random
import numpy as np
import skimage.io
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import RandomApply, ToTensor, RandomResizedCrop, \
    RandomAffine, RandomHorizontalFlip, RandomVerticalFlip, RandomPerspective
from utils import undersample


class MyData(Dataset):
    def __init__(self, imageDir, maskDir, img_size=256, is_training=True):
        super(MyData, self).__init__()
        images = glob.glob(join(imageDir, '*'))
        self.images = sorted(images)
        masks = glob.glob(join(maskDir, '*'))
        self.masks = sorted(masks)
        self.is_training = is_training
        self.img_size = img_size
        self.len = len(self.images)

    def transform(self, img_A, img_B):
        totensor = ToTensor()  # rescale to [0,1.]
        random_resize_crop = RandomApply(
            torch.nn.ModuleList([RandomResizedCrop(self.img_size, scale=(0.7, 1.0), ratio=(0.8, 1.0))]),
            p=0.3)
        random_affine = RandomApply(torch.nn.ModuleList([RandomAffine(20, translate=(0.1, 0.1), scale=(0.9, 1.1),
                                                                      shear=(-5, 5, -5, 5),
                                                                      interpolation=transforms.InterpolationMode.BILINEAR)]),
                                    p=0.3)
        random_h_flip = RandomHorizontalFlip(p=0.3)
        random_v_flip = RandomVerticalFlip(p=0.3)
        random_perspective = RandomPerspective(0.05, 0.3)
        for i, img in enumerate([img_A, img_B]):
            img = totensor(img)
            img = random_resize_crop(img)
            img = random_affine(img)
            img = random_h_flip(img)
            img = random_v_flip(img)
            img = random_perspective(img)
            if i == 0:
                img_A = img
            else:
                img_B = img
        return img_A, img_B

    def __getitem__(self, i):
        if self.is_training:
            image_A = skimage.io.imread(self.images[i])
            index_B = random.randint(0, self.len - 1)
            image_B = skimage.io.imread(self.images[index_B])
            ########################### image preprocessing ###########################
            # same random transform to image A and B
            image_A, image_B = self.transform(image_A[..., np.newaxis], image_B[..., np.newaxis])

            # complex value normalize to [-1-j,1+j],
            # so for the 2 channel real representation, pixel range is [-1.,1.]
            image_A = (image_A[0] - (0.5 + 0.5j)) * 2.0
            image_B = (image_B[0] - (0.5 + 0.5j)) * 2.0

            # random mask when training
            mask_idx = random.randint(0, len(self.masks) - 1)
            mask = skimage.io.imread(self.masks[mask_idx])
            mask = torch.from_numpy(mask / 255.)
            # generate zero-filled image x_und, k_und, k
            image_A_und, k_A_und, k_A = undersample(image_A, mask)
            image_B_und, k_B_und, k_B = undersample(image_B, mask)

            ########################## complex to 2 channel ##########################
            im_A = torch.view_as_real(image_A).permute(2, 0, 1).contiguous()
            im_A_und = torch.view_as_real(image_A_und).permute(2, 0, 1).contiguous()
            k_A_und = torch.view_as_real(k_A_und).permute(2, 0, 1).contiguous()
            im_B = torch.view_as_real(image_B).permute(2, 0, 1).contiguous()
            im_B_und = torch.view_as_real(image_B_und).permute(2, 0, 1).contiguous()
            k_B_und = torch.view_as_real(k_B_und).permute(2, 0, 1).contiguous()
            mask = torch.view_as_real(mask * (1. + 1.j)).permute(2, 0, 1).contiguous()

            return {'im_A': im_A, 'im_A_und': im_A_und, 'k_A_und': k_A_und,
                    'im_B': im_B, 'im_B_und': im_B_und, 'k_B_und': k_B_und, 'mask': mask}
        else:
            image_A = skimage.io.imread(self.images[i])
            image_A = torch.from_numpy((image_A / 255. - (0.5 + 0.5j)) * 2.0)
            mask = skimage.io.imread(self.masks[0])
            mask = torch.from_numpy(mask / 255.)
            # generate x_und (zero-filled image ), k_und (k-space of x_und), k (k space of image_A)
            image_A_und, k_A_und, k_A = undersample(image_A, mask)

            ########################## complex to 2 channel ##########################
            im_A = torch.view_as_real(image_A).permute(2, 0, 1).contiguous()
            im_A_und = torch.view_as_real(image_A_und).permute(2, 0, 1).contiguous()
            k_A_und = torch.view_as_real(k_A_und).permute(2, 0, 1).contiguous()
            mask = torch.view_as_real(mask * (1. + 1.j)).permute(2, 0, 1).contiguous()

            return {'im_A': im_A, 'im_A_und': im_A_und, 'k_A_und': k_A_und, 'mask': mask}

    def __len__(self):
        return self.len
