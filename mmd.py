#!/usr/bin/env python
# encoding: utf-8


import torch
import math
from torch.distributions.multivariate_normal import MultivariateNormal as mvn
from numpy.random import multivariate_normal as MVN
import numpy as np

min_var_est = 1e-8

### online method
def IncrementMatrices(C, V, history, x_i, sigma_list):
    '''

    :param C: C_{i-1} (i-2)*(i-2)
    :param V: V_{i-1} (i-2)
    :param history: [x_1, x_2, ..., x_{i-1}] size=i-1
    :param x_i: new coming data
    :return: C_i, V_i, all size(0)=i-1
    '''
    # print(C.size(), V.size(), history.size())
    assert C.size(0) == V.size(0) == (history.size(0) - 1), "length of vectors are incorrect!"
    # x_i1 means x_{i-1}
    x_i1 = history[-1] # size[d]


    XX = torch.diagonal(torch.matmul(history, history.t())) #size[40]
    C_XY = 2.0 * torch.matmul(history, x_i1.t()) #size[40]
    C_YY = torch.matmul(x_i1, x_i1.t()) #size[1]


    V_XY = 2.0 * torch.matmul(history, x_i.t())
    V_YY = torch.matmul(x_i, x_i.t()) #size[1]

    exponent_C = XX - C_XY + C_YY
    exponent_V = XX - V_XY + V_YY


    # exponent_C = torch.norm(history - x_i1, dim=1) ** 2
    # exponent_V = torch.norm(history - x_i, dim=1) ** 2

    K_C = 0.0
    K_V = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K_C += torch.exp(-gamma * exponent_C)
        K_V += torch.exp(-gamma * exponent_V)

    V_i = K_V

    C_i = torch.cat([C, torch.unsqueeze(K_C[:-1], dim=1)], dim=1)
    C_i = torch.cat([C_i, torch.unsqueeze(K_C, dim=0)], dim=0)
    # print(C_i.size(), V_i.size(), history.size())

    return C_i, V_i

def OnlineUpdate(C, G, V):
    '''

    :param C: C_i
    :param G: G_i
    :param V: V_i
    :return: \tilde(f)_D(x_i)
    '''
    # C_inv = torch.inverse(C)
    # print('C:', C)
    # print('C_inv:', C_inv)
    # V_matmul_C_inv = torch.matmul(V, C_inv)
    # print('V^T C_inv:', V_matmul_C_inv)
    # print(torch.linalg.det(C), C)

    V_matmul_C_inv = torch.linalg.solve(C, V)
    # print('C_inv by solve:', C_inv) # same as torch.inverse
    # print('V^T C_inv by solve:', V_matmul_C_inv) # different, OK use this one

    # print(V.size(), C_inv.size())
    mean_tilde = torch.matmul(V_matmul_C_inv, G)
    var_tilde =  1.0 - torch.matmul(V_matmul_C_inv, V)
    # print(var_tilde)
    var_tilde = torch.abs(var_tilde) # manually make it non-negative
    f_D_tilde_xi = mean_tilde + torch.sqrt(var_tilde) * torch.randn(1).cuda()
    # print('mean_tilde:', mean_tilde)
    # print('var_tilde:', var_tilde)
    return f_D_tilde_xi

def f_D(x, x_i, sigma_list):
    '''
    for online method
    :param x: the batch, N*d
    :param x_i:          1*d
    :return:
    '''


    # exponent = torch.norm(x - x_i, dim=1) ** 2
    # exponent = torch.unsqueeze(exponent, dim=1)


    XX = torch.unsqueeze(torch.diagonal(torch.matmul(x, x.t())), dim=1) # size[40,1]
    XY = 2.0 * torch.matmul(x, x_i.t()) # size[40,1]
    YY = torch.matmul(x_i, x_i.t()) # size[1,1]

    exponent = XX - XY + YY


    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K += torch.exp(-gamma * exponent)


    return torch.mean(K, dim=0)


### batch method
def rbf_kernel_dataset(X, sigma_list, noise_multiplier):
    '''
    Compute Gaussian kernel within dataset X
    :param X: N*d
    :return:
    '''
    N = X.size(0)

    XXT = torch.mm(X, X.t())
    diag_XXT = torch.diag(XXT).unsqueeze(1)
    X_norm_sqr = diag_XXT.expand_as(XXT)
    exponent = X_norm_sqr - 2 * XXT + X_norm_sqr.t() # (N*N)

    K_XX = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K_XX += torch.exp(-gamma * exponent)

    f_Dx = torch.mean(K_XX, dim=0) # (N,)
    if noise_multiplier > 0:
        coeff = len(sigma_list) * math.sqrt(2) / N * noise_multiplier
        mvn_Dx = mvn(torch.zeros_like(f_Dx), K_XX * coeff)
        f_Dx_tilde = f_Dx + mvn_Dx.sample()
        del mvn_Dx
    else:
        f_Dx_tilde = torch.Tensor([0.0]) # dumb value
    return f_Dx_tilde, f_Dx, K_XX


def rbf_kernel_full(X, Y, sigma_list):
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

    # K_XX = K[:N, :N]
    K_XY = K[:N, N:]
    K_YY = K[N:, N:]
    # f_Dx = torch.mean(K_XX, dim=0) # (N,)
    f_Dy = torch.mean(K_XY, dim=0) # (M,)

    return f_Dy, K_YY, K_XY

def get_DP_mmd_loss(f_Dx, f_Dx_tilde, X, noise_multiplier):
    '''
    Compute the DP mmd loss where f_D(x) and f_D(y) are privatized by Gaussian process.
    :param f_Dx: N*1
    :param f_Dx_tilde: N*1
    :param X: N*d
    :return:  mmd loss where f_D(x) and f_D(y) are privatized by Gaussian process.
    '''
    N = X.size(0)

    def DP_mmd_loss(Y, sigma_list):
        '''
        :param Y: M*d
        :return:
        '''
        M = Y.size(0)
        coeff = len(sigma_list) * math.sqrt(2) / N * noise_multiplier
        f_Dy, K_YY, _ = rbf_kernel_full(X, Y, sigma_list)
        mvn_Dy = mvn(torch.zeros_like(f_Dy), K_YY * coeff)
        # mvn_Dy = mvn(torch.zeros_like(f_Dy), coeff * torch.eye(M).cuda())
        f_Dy_tilde = f_Dy + mvn_Dy.sample()
        del mvn_Dy
        # print(torch.mean(f_Dx), torch.mean(f_Dy), torch.mean(K_YY))
        # print(torch.mean(f_Dx_tilde), torch.mean(f_Dy_tilde), torch.mean(K_YY))
        mmd_XX = torch.mean(f_Dx_tilde)
        mmd_XY = torch.mean(f_Dy_tilde)
        mmd_YY = torch.mean(K_YY)

        return mmd_XX - 2 * mmd_XY + mmd_YY

    def mmd_loss(Y, sigma_list):
        '''
        :param Y: M*d
        :return:
        '''
        f_Dy, K_YY, _ = rbf_kernel_full(X, Y, sigma_list)
        return torch.mean(f_Dx) - 2 * torch.mean(f_Dy) + torch.mean(K_YY)

    return DP_mmd_loss, mmd_loss


def rbf_kernel_DP_loss(X, Y, sigma_list, noise_multiplier):
    '''
    Compute Gaussian kernel between dataset X and Y
    f_D uses Gaussian kernel, compute everything in Gaussian RKHS
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
    f_D uses Gaussian kernel, compute everything in Gaussian RKHS
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

    if noise_multiplier > 0:
        coeff =  math.sqrt(2 * len(sigma_list)) / N * noise_multiplier
        mvn_Dxy = mvn(torch.zeros_like(f_Dxy), K * coeff)
        f_Dxy_tilde = f_Dxy + mvn_Dxy.sample()
        f_Dx_tilde = f_Dxy_tilde[:N] # [N]
        f_Dy_tilde = f_Dxy_tilde[N:] # [M]
        del mvn_Dxy
        mmd_XX = torch.mean(f_Dx_tilde)
        mmd_XY = torch.mean(f_Dy_tilde)
    else:
        mmd_XX = torch.mean(K_XX)
        mmd_XY = torch.mean(K_XY)
    mmd_YY = torch.mean(K_YY)

    return mmd_XX - 2 * mmd_XY + mmd_YY

def sensitivity_phi_LaplacianRKHS(sigma_list, gamma):
    '''
    phi_\mu(x) = 1./n_h \sum_{i=1}^{n_h}\exp{-(x-\mu)^2/(2h_i^2)}
    the square of sensitivity of phi in H_L \leq n_h + 1/(2\gamma)* (\sqrt{\pi}/2 \sum_{i=1}^{n_h}1/h_i +\sqrt{\pi/2}*\gamma^2*\sum_{i=1}^{n_h} h_i)

    :param sigma_list: list of h_i, for function
    :param gamma: for Laplacian kernel
    :return:
    '''
    sigma_inv_list = [1./x for x in sigma_list]
    sensitivity_sq = len(sigma_list) + 1./(2.*gamma) * (math.sqrt(math.pi)/2. * sum(sigma_inv_list) + math.sqrt(math.pi/2.)* (gamma**2) * sum(sigma_list))
    return math.sqrt(sensitivity_sq)

def laplace_kernel_DP_loss_with_labels(X, Y, x_label, y_label, sigma_list, noise_multiplier):
    '''
    Compute Gaussian kernel between dataset X and Y, with labels
    f_D uses Gaussian kernel, compute everything in Laplacian RKHS
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
    mmd_YY = torch.mean(K_YY)
    f_Dx = torch.mean(K_XX, dim=0) # (N,)
    f_Dy = torch.mean(K_XY, dim=0) # (M,)
    f_Dxy = torch.cat([f_Dx, f_Dy]) # size [N+M]

    #### Laplacian RKHS
    gamma_L = len(sigma_list)/sum(sigma_list) # for the Laplacian kernel
    sen = sensitivity_phi_LaplacianRKHS(sigma_list, gamma_L)
    coeff =  2 / N * sen * noise_multiplier

    # recompute K by changing kernel to k_L(x,y)=\exp(-\gamma||x-y||_1)
    Z_repeat = Z.repeat(N+M, 1, 1)
    Z_repeat_T = torch.transpose(Z_repeat, 0, 1)
    exponent_L = torch.sum(torch.abs(Z_repeat_T - Z_repeat), dim=2)
    K_L = 0.0
    K_L += torch.exp(-gamma_L * exponent_L)
    K_L = K_L * LLT
    mvn_Dxy = mvn(torch.zeros_like(f_Dxy), K_L * coeff)
    f_Dxy_tilde = f_Dxy + mvn_Dxy.sample()
    f_Dx_tilde = f_Dxy_tilde[:N] # [N]
    f_Dy_tilde = f_Dxy_tilde[N:] # [M]
    del mvn_Dxy
    mmd_XX = torch.mean(f_Dx_tilde)
    mmd_XY = torch.mean(f_Dy_tilde)

    return mmd_XX - 2 * mmd_XY + mmd_YY


# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss


# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = (alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c)
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = (alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c)
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = (alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c)
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = (alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c)
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def _mix_rbf_kernel(X, Y, sigma_list):
    assert(X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
        # print('batch:', (Kt_XX_sum + sum_diag_X) / (m * m), K_XY_sum / (m * m), (Kt_YY_sum + sum_diag_Y) / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)                     # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X      # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y      # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum  = (K_XY ** 2).sum()                    # \| K_{XY} \|_F^2

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    var_est = (
        2.0 / (m**2 * (m - 1.0)**2) * (2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
        - (4.0*m - 6.0) / (m**3 * (m - 1.0)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0*(m - 2.0) / (m**3 * (m - 1.0)**2) * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0*(m - 3.0) / (m**3 * (m - 1.0)**2) * (K_XY_2_sum) - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0 / (m**3 * (m - 1.0)) * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0))
        )
    return mmd2, var_est