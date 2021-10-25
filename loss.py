import torch


def total_variant(images):
    '''
    :param images:  [B,C,W,H]
    :return: total_variant
    '''
    pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
    pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]

    tot_var = torch.abs(pixel_dif1).sum([1, 2, 3]) + torch.abs(pixel_dif2).sum([1, 2, 3])

    return tot_var


def RF(x_rec, mask, norm='ortho'):
    '''
    return the masked k-space of x_rec
    '''
    x_rec = x_rec.permute(0, 2, 3, 1)
    mask = mask.permute(0, 2, 3, 1)
    k_rec = torch.fft.fft2(torch.view_as_complex(x_rec.contiguous()), norm=norm)
    k_rec = torch.view_as_real(k_rec)
    k_rec *= mask
    k_rec = k_rec.permute(0, 3, 1, 2)

    return k_rec


def build_loss(dis_real, dis_fake):
    '''
    calculate WGAN loss
    '''
    d_loss = torch.mean(dis_fake - dis_real)
    g_loss = -torch.mean(dis_fake)
    return g_loss, d_loss


def cal_loss(S01, S01_k_un, S02, S02_k_un, mask, Sp1, S1, Tp1, T1, Sp2, S2, Tp2, T2, S1_dis_real, S1_dis_fake,
             T1_dis_fake, S2_dis_real, S2_dis_fake,
             T2_dis_fake, cal_G=True):
    '''
    TODO: input arguments are too much, and some calculation is redundant
    '''
    G_loss_AA, D_loss_AA = build_loss(S1_dis_real, S1_dis_fake)
    G_loss_Aa, D_loss_Aa = build_loss(S1_dis_real, T1_dis_fake)

    G_loss_BB, D_loss_BB = build_loss(S2_dis_real, S2_dis_fake)
    G_loss_Bb, D_loss_Bb = build_loss(S2_dis_real, T2_dis_fake)

    G_loss_AB, D_loss_AB = build_loss(S1_dis_real, S2_dis_fake)
    G_loss_Ab, D_loss_Ab = build_loss(S1_dis_real, T2_dis_fake)

    G_loss_BA, D_loss_BA = build_loss(S2_dis_real, S1_dis_fake)
    G_loss_Ba, D_loss_Ba = build_loss(S2_dis_real, T1_dis_fake)

    if cal_G:
        recon_frq_AA = torch.mean(torch.abs(S01_k_un - RF(Sp1, mask)))
        recon_frq_BB = torch.mean(torch.abs(S02_k_un - RF(Sp2, mask)))

        recon_frq_Aa = torch.mean(torch.abs(S01_k_un - RF(Tp1, mask)))
        recon_frq_Bb = torch.mean(torch.abs(S02_k_un - RF(Tp2, mask)))

        recon_img_AA = torch.mean((torch.abs((S01) - (S1))))
        recon_img_BB = torch.mean((torch.abs((S02) - (S2))))
        error_img_AA = torch.mean(torch.abs((S01) - (Sp1)))
        error_img_BB = torch.mean(torch.abs((S02) - (Sp2)))
        smoothness_AA = torch.mean(total_variant(S1))
        smoothness_BB = torch.mean(total_variant(S2))

        recon_img_Aa = torch.mean(torch.abs((S01) - (T1)))
        recon_img_Bb = torch.mean(torch.abs((S02) - (T2)))
        error_img_Aa = torch.mean(torch.abs((S01) - (Tp1)))
        error_img_Bb = torch.mean(torch.abs((S02) - (Tp2)))
        smoothness_Aa = torch.mean(total_variant(T1))
        smoothness_Bb = torch.mean(total_variant(T2))

        ALPHA = 1e+1
        GAMMA = 1e-0
        DELTA = 1e-4
        RATES = torch.count_nonzero(torch.ones_like(mask)) / 2. / torch.count_nonzero(mask)
        GAMMA = RATES

        g_loss = \
            (G_loss_AA + G_loss_BB + G_loss_AB + G_loss_BA) + \
            (G_loss_Aa + G_loss_Bb + G_loss_Ab + G_loss_Ba) + \
            (recon_img_AA + recon_img_BB) * 1.00 * ALPHA * RATES + \
            (recon_img_Aa + recon_img_Bb) * 1.00 * ALPHA * RATES + \
            (error_img_AA + error_img_BB) * 1e+2 * ALPHA * RATES + \
            (error_img_Aa + error_img_Bb) * 1e+2 * ALPHA * RATES + \
            (recon_frq_AA + recon_frq_BB) * 1.00 * GAMMA * RATES + \
            (recon_frq_Aa + recon_frq_Bb) * 1.00 * GAMMA * RATES + \
            (smoothness_AA + smoothness_BB + smoothness_Aa + smoothness_Bb) * DELTA
        return g_loss, [G_loss_AA, G_loss_Aa, recon_img_AA, recon_img_Aa, error_img_AA, error_img_Aa, recon_frq_AA,
                        recon_frq_Aa, smoothness_AA, smoothness_Aa]
    else:

        d_loss = \
            D_loss_AA + D_loss_BB + D_loss_AB + D_loss_BA + \
            D_loss_Aa + D_loss_Bb + D_loss_Ab + D_loss_Ba

        return d_loss, [D_loss_AA, D_loss_Aa, D_loss_AB, D_loss_Ab]
