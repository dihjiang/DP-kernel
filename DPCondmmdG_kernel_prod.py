#!/usr/bin/env python
# encoding: utf-8


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import os

from util import *
from datasets import *
import numpy as np

import base_module
from mmd import *
import time
from torch.utils.data import DataLoader
from pyvacy import sampling

start_t = time.time()

# NetG is a decoder
# input: batch_size * nz * 1 * 1
# output: batch_size * nc * image_size * image_size
class CNetG(nn.Module):
    def __init__(self, decoder):
        super(CNetG, self).__init__()
        self.decoder = decoder

    def forward(self, input, label):
        output = self.decoder(input, label)
        return output

# Get argument
parser = argparse.ArgumentParser()
parser = get_args(parser)
args = parser.parse_args()
print(args)

checkpoints_dir = f'./{args.experiment}'
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)


if torch.cuda.is_available():
    args.cuda = True
    torch.cuda.set_device(args.gpu_device)
    print("Using GPU device", torch.cuda.current_device())
else:
    raise EnvironmentError("GPU device not available!")

set_random_seed(args.seed)
# Get data
trn_dataset, _ = set_datasets(args.dataset, None)

hidden_dim = args.nz
G_decoder = base_module.CondDecoder(args.image_size, args.nc, k=args.nz, ngf=64, num_classes=10)
netG = CNetG(G_decoder)
print("netG:", netG)
netG.apply(base_module.weights_init)
if args.cuda:
    netG.cuda()

# bandwidth for Gaussian kernl in MMD
base = 1.0
sigma_list = args.sigma_list
sigma_list = [sigma / base for sigma in sigma_list]
print(sigma_list)

# put variable into cuda device
if args.dataset in ['mnist', 'fmnist']:
    fixed_noise = torch.randn(80, args.nz).cuda()
    fixed_label = torch.arange(10).repeat(8).cuda()
elif args.dataset == 'celeba':
    fixed_noise = torch.randn(20, args.nz).cuda()
    fixed_label = torch.cat([torch.zeros(10), torch.ones(10)]).cuda()
    fixed_label = fixed_label.to(torch.int64)

# setup optimizer
optimizerG = torch.optim.RMSprop(netG.parameters(), lr=args.lr)
noise_multiplier = args.noise

minibatch_loader, _ = sampling.get_data_loaders(
    args.batch_size,
    1,  # mbs, doesn't matter
    args.max_iter) # iterations
iter = 0

for x, label in minibatch_loader(trn_dataset):
    iter_loss = 0
    x = x.cuda()
    if args.dataset == 'celeba':
        label = label[:, 20]
    label = label.cuda()
    batch_size = x.size(0)

    gen_labels = torch.randint(args.n_class, (batch_size,)).cuda()

    optimizerG.zero_grad()

    noise = torch.randn(batch_size, args.nz).cuda()
    y = netG(noise, label=gen_labels)
    label = F.one_hot(label, args.n_class).float()
    gen_labels = F.one_hot(gen_labels, args.n_class).float()

    #### compute mmd loss using my implementation ####
    DP_mmd_loss = rbf_kernel_DP_loss_with_labels(x.view(batch_size, -1), y.view(batch_size, -1), label, gen_labels, sigma_list, noise_multiplier)

    errG = torch.pow(DP_mmd_loss, 2)
    errG.backward()
    optimizerG.step()
    iter_loss += errG.item()

    if iter % args.vis_step == 0:
        # training loss
        print('Current iter: ', iter, 'Total training iters: ', args.max_iter)
        print('Training loss: {}\tLoss:{:.6f}\t'.format(iter, iter_loss))
        y_fixed = netG(fixed_noise, label=fixed_label)
        y_fixed.data = y_fixed.data.mul(0.5).add(0.5)
        grid = vutils.make_grid(y_fixed.data, nrow=10)
        vutils.save_image(grid, os.path.join(checkpoints_dir, f'netG_iter{iter}_noise{args.noise}_lr{args.lr}_bs{args.batch_size}.png'))
        torch.save(netG.state_dict(), os.path.join(checkpoints_dir, f'netG_iter{iter}_noise{args.noise}_lr{args.lr}_bs{args.batch_size}.pkl'))

    iter += 1


print(f'>>> Finished DP-conditional mmd on {args.dataset} by perturbing the loss!')
print("Time cost: ", round(time.time() - start_t, 2), "s.")
