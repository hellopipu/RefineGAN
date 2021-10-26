from torch.fft import fft2, ifft2
import torch
import math


def undersample(image, mask, norm='ortho'):
    assert image.shape == mask.shape

    k = fft2(image, norm=norm)
    k_und = mask * k
    x_und = ifft2(k_und, norm=norm)

    return x_und, k_und, k


def cal_psnr(pred, gt, maxp=1.):
    pred = pred.abs()
    pred = torch.clamp(pred, min=0., max=maxp)  # some points in pred are larger than maxp
    gt = gt.abs()

    mse = torch.mean((pred - gt) ** 2, dim=(1, 2))

    psnr = -10. * torch.log10(mse)  # + 1e-6
    psnr = psnr + 20. * math.log10(maxp)

    return psnr.sum()


def RF(x_rec, mask, norm='ortho'):
    '''
    RF means R*F(input), F is fft, R is applying mask;
    return the masked k-space of x_rec,
    '''
    x_rec = x_rec.permute(0, 2, 3, 1)
    mask = mask.permute(0, 2, 3, 1)
    k_rec = torch.fft.fft2(torch.view_as_complex(x_rec.contiguous()), norm=norm)
    k_rec = torch.view_as_real(k_rec)
    k_rec *= mask
    k_rec = k_rec.permute(0, 3, 1, 2)
    return k_rec


def revert_scale(im_tensor, a=2., b=-1.):
    '''
    param: im_tensor : [B, 2, W, H]
    '''
    b = b * torch.ones_like(im_tensor)
    im = (im_tensor - b) / a

    return im
