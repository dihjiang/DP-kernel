#!/usr/bin/env python
# encoding: utf-8


import torch
import math
from torch.distributions.multivariate_normal import MultivariateNormal as mvn



def rbf_kernel_DP_loss(X, Y, sigma_list, noise_multiplier):
    '''
    Compute Gaussian kernel between dataset X and Y
    :param X: N*d
    :param Y: M*d
    :return:
    '''
    N = X.size(0)
    M = Y.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t() # (N+M)*(N+M)

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K += torch.exp(-gamma * exponent)

    K_XX = K[:N, :N]
    K_XY = K[:N, N:]
    K_YY = K[N:, N:]
    f_Dx = torch.mean(K_XX, dim=0) # (N,)
    f_Dy = torch.mean(K_XY, dim=0) # (M,)
    f_Dxy = torch.cat([f_Dx, f_Dy]) # size [N+M]

    # batch method
    coeff =  math.sqrt(2 * len(sigma_list)) / N * noise_multiplier
    mvn_Dxy = mvn(torch.zeros_like(f_Dxy), K * coeff)
    f_Dxy_tilde = f_Dxy + mvn_Dxy.sample()
    f_Dx_tilde = f_Dxy_tilde[:N] # [N]
    f_Dy_tilde = f_Dxy_tilde[N:] # [M]
    del mvn_Dxy
    mmd_XX = torch.mean(f_Dx_tilde)
    mmd_XY = torch.mean(f_Dy_tilde)
    mmd_YY = torch.mean(K_YY)

    return mmd_XX - 2 * mmd_XY + mmd_YY

def rbf_kernel_DP_loss_with_labels(X, Y, x_label, y_label, sigma_list, noise_multiplier):
    '''
    Compute Gaussian kernel between dataset X and Y, with labels
    :param X: N*d
    :param Y: M*d
    :return:
    '''
    N = X.size(0)
    M = Y.size(0)

    Z = torch.cat((X, Y), 0)
    L = torch.cat((x_label, y_label), 0)
    ZZT = torch.mm(Z, Z.t())
    LLT = torch.mm(L, L.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t() # (N+M)*(N+M)

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K += torch.exp(-gamma * exponent)

    K = K * LLT # new kernel account for labels

    K_XX = K[:N, :N]
    K_XY = K[:N, N:]
    K_YY = K[N:, N:]
    f_Dx = torch.mean(K_XX, dim=0) # (N,)
    f_Dy = torch.mean(K_XY, dim=0) # (M,)
    f_Dxy = torch.cat([f_Dx, f_Dy]) # size [N+M]

    # batch method
    coeff =  math.sqrt(2 * len(sigma_list)) / N * noise_multiplier
    mvn_Dxy = mvn(torch.zeros_like(f_Dxy), K * coeff)
    f_Dxy_tilde = f_Dxy + mvn_Dxy.sample()
    f_Dx_tilde = f_Dxy_tilde[:N] # [N]
    f_Dy_tilde = f_Dxy_tilde[N:] # [M]
    del mvn_Dxy
    mmd_XX = torch.mean(f_Dx_tilde)
    mmd_XY = torch.mean(f_Dy_tilde)
    mmd_YY = torch.mean(K_YY)

    return mmd_XX - 2 * mmd_XY + mmd_YY