import torch
import torch.nn as nn

NB_FILTERS = 64  # channel size


class Dataconsistency(nn.Module):
    def __init__(self):
        super(Dataconsistency, self).__init__()

    def forward(self, x_rec, mask, k_un, norm='ortho'):
        x_rec = x_rec.permute(0, 2, 3, 1)
        mask = mask.permute(0, 2, 3, 1)
        k_un = k_un.permute(0, 2, 3, 1)
        k_rec = torch.fft.fft2(torch.view_as_complex(x_rec.contiguous()), norm=norm)
        k_rec = torch.view_as_real(k_rec)
        k_out = k_rec + (k_un - k_rec) * mask
        k_out = torch.view_as_complex(k_out)
        x_out = torch.view_as_real(torch.fft.ifft2(k_out, norm=norm))
        x_out = x_out.permute(0, 3, 1, 2)
        return x_out


class Residual(nn.Module):
    def __init__(self, chan):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(chan, chan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(chan),
            nn.LeakyReLU(0.2),
            nn.Conv2d(chan, chan // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(chan // 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(chan // 2, chan, kernel_size=3, stride=1, padding=1, bias=False),  # nl=tf.identity
            nn.BatchNorm2d(chan),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return x + self.block(x)


class Residual_enc(nn.Module):
    def __init__(self, in_chan, chan):
        super(Residual_enc, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_chan, chan, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(chan),
            nn.LeakyReLU(0.2),
            Residual(chan),
            nn.Conv2d(chan, chan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(chan),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)


class Residual_dec(nn.Module):
    def __init__(self, in_chan, chan):
        super(Residual_dec, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_chan, chan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(chan),
            nn.LeakyReLU(0.2),
            Residual(chan),
            nn.ConvTranspose2d(chan, chan, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(chan),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.e0 = Residual_enc(2, NB_FILTERS * 1)
        self.e1 = Residual_enc(NB_FILTERS * 1, NB_FILTERS * 2)
        self.e2 = Residual_enc(NB_FILTERS * 2, NB_FILTERS * 4)
        self.e3 = Residual_enc(NB_FILTERS * 4, NB_FILTERS * 8)

        self.d3 = Residual_dec(NB_FILTERS * 8, NB_FILTERS * 4)
        self.d2 = Residual_dec(NB_FILTERS * 4, NB_FILTERS * 2)
        self.d1 = Residual_dec(NB_FILTERS * 2, NB_FILTERS * 1)
        self.d0 = Residual_dec(NB_FILTERS * 1, NB_FILTERS * 1)

        self.dd = nn.Sequential(
            nn.Conv2d(NB_FILTERS, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x_e0 = self.e0(x)
        x_e1 = self.e1(x_e0)
        x_e2 = self.e2(x_e1)
        x_e3 = self.e3(x_e2)

        x_d3 = self.d3(x_e3)
        x_d2 = self.d2(x_d3 + x_e2)
        x_d1 = self.d1(x_d2 + x_e1)
        x_d0 = self.d0(x_d1 + x_e0)

        out = self.dd(x_d0)

        return out + x


class Refine_G(nn.Module):

    def __init__(self):
        super(Refine_G, self).__init__()
        self.g0 = Generator()
        self.g1 = Generator()
        self.dc = Dataconsistency()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.trunc_normal_(m.weight, mean=0, std=0.02, a=-0.04, b=0.04)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_un, k_un, mask):
        x_rec0 = self.g0(x_un)
        x_dc0 = self.dc(x_rec0, mask, k_un)
        x_rec1 = self.g1(x_dc0)
        x_dc1 = self.dc(x_rec1, mask, k_un)

        return x_rec0, x_dc0, x_rec1, x_dc1


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.e0 = Residual_enc(2, NB_FILTERS * 1)
        self.e1 = Residual_enc(NB_FILTERS * 1, NB_FILTERS * 2)
        self.e2 = Residual_enc(NB_FILTERS * 2, NB_FILTERS * 4)
        self.e3 = Residual_enc(NB_FILTERS * 4, NB_FILTERS * 8)

        self.ret = nn.Conv2d(NB_FILTERS * 8, 1, kernel_size=4, stride=1, padding=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.trunc_normal_(m.weight, mean=0, std=0.02, a=-0.04, b=0.04)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_e0 = self.e0(x)
        x_e1 = self.e1(x_e0)
        x_e2 = self.e2(x_e1)
        x_e3 = self.e3(x_e2)

        ret = self.ret(x_e3)

        return ret


if __name__ == '__main__':
    netG = Refine_G()
    netD = Discriminator()
    print('    Total params: %.2fMB' % (sum(p.numel() for p in netG.parameters()) / (1024.0 * 1024) * 4))
    print('    Total params: %.2fMB' % (sum(p.numel() for p in netD.parameters()) / (1024.0 * 1024) * 4))
