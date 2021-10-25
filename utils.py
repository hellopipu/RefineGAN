import numpy as np
from numpy.fft import fft2, ifft2
import torch


def undersample(image, mask, norm='ortho'):
    assert image.shape == mask.shape

    k = fft2(image, norm=norm)
    k_und = mask * k
    x_und = ifft2(k_und, norm=norm)

    return x_und, k_und, k


def c2r(x, axis=1):
    """Convert complex data to pseudo-complex data (2 real channels)

    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    shape = x.shape
    dtype = np.float32 if x.dtype == np.complex64 else np.float64

    x = np.ascontiguousarray(x).view(dtype=dtype).reshape(shape + (2,))

    n = x.ndim
    if axis < 0: axis = n + axis
    if axis < n:
        newshape = tuple([i for i in range(0, axis)]) + (n - 1,) \
                   + tuple([i for i in range(axis, n - 1)])
        x = x.transpose(newshape)

    return x


def to_tensor_format(x, mask=False):
    if mask:
        x = x * (1 + 1j)
    x = c2r(x, axis=0)

    return x


def cal_psnr(pred, gt, maxp=255):
    pred = pred.abs()
    pred = torch.clamp(pred, min=0., max=255.)  # some points in pred are larger than 255
    gt = gt.abs()

    mse = torch.mean((pred - gt) ** 2)
    if maxp is None:
        pass
        # psnr = tf.multiply(log10(mse), -10., name=name)
    else:
        maxp = torch.Tensor([maxp])
        psnr = -10. * torch.log10(mse + 1e-6)
        psnr = psnr + 20. * torch.log10(maxp)

    return psnr.numpy()[0]
