import os
from os.path import join as join
import glob
import random
import numpy as np
import skimage.io
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import RandomApply, RandomRotation, ToTensor, RandomResizedCrop, \
    Compose, RandomAffine, RandomHorizontalFlip, RandomVerticalFlip, RandomPerspective
from utils import undersample, to_tensor_format


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
        totensor = ToTensor()
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
            image_A = (image_A / 255. - 0.5) * 2.0
            image_B = (image_B / 255. - 0.5) * 2.0
            # same random transform to image A and B
            image_A, image_B = self.transform(image_A[..., np.newaxis], image_B[..., np.newaxis])
            image_A = image_A[0].numpy()
            image_B = image_B[0].numpy()
            # random mask when training
            mask_idx = random.randint(0, len(self.masks) - 1)

            mask = skimage.io.imread(self.masks[mask_idx])
            mask = mask / 255.
            # generate zero-filled image x_und, k_und, k
            image_A_und, k_A_und, k_A = undersample(image_A, mask)
            image_B_und, k_B_und, k_B = undersample(image_B, mask)

            # complex to 2 channel
            im_A = to_tensor_format(np.complex64(image_A))
            im_A_und = to_tensor_format(image_A_und)
            k_A_und = to_tensor_format(k_A_und)
            im_B = to_tensor_format(np.complex64(image_B))
            im_B_und = to_tensor_format(image_B_und)
            k_B_und = to_tensor_format(k_B_und)
            mask = to_tensor_format(mask, mask=True)

            return {'im_A': im_A, 'im_A_und': im_A_und, 'k_A_und': k_A_und,
                    'im_B': im_B, 'im_B_und': im_B_und, 'k_B_und': k_B_und, 'mask': mask}
        else:
            image_A = skimage.io.imread(self.images[i])
            image_A = (image_A / 255. - 0.5) * 2.0
            mask = skimage.io.imread(self.masks[0])
            mask = mask / 255.
            # generate x_und (zero-filled image ), k_und (k-space of x_und), k (k space of image_A)
            image_A_und, k_A_und, k_A = undersample(image_A, mask)

            # complex to 2 channel
            im_A = to_tensor_format(np.complex64(image_A))
            im_A_und = to_tensor_format(image_A_und)
            k_A_und = to_tensor_format(k_A_und)
            mask = to_tensor_format(mask, mask=True)

            return {'im_A': im_A, 'im_A_und': im_A_und, 'k_A_und': k_A_und, 'mask': mask}

    def __len__(self):
        return self.len
