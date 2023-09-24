#!/usr/bin/env python
# encoding: utf-8


import argparse
import torch
import torch.nn as nn
import torchvision.utils as vutils
import os

from util import *
import numpy as np

import base_module
from mmd import *
from datasets import *
import time
from torch.utils.data import DataLoader
from pyvacy import sampling

start_t = time.time()

# NetG is a decoder
# input: batch_size * nz * 1 * 1
# output: batch_size * nc * image_size * image_size
class NetG(nn.Module):
    def __init__(self, decoder):
        super(NetG, self).__init__()
        self.decoder = decoder

    def forward(self, input):
        output = self.decoder(input)
        return output


# Get argument
parser = argparse.ArgumentParser()
parser = get_args(parser)
args = parser.parse_args()
print(args)
CLASS = args.select

checkpoints_dir = f'./{args.experiment}/CLASS{CLASS}'
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
if args.dataset == 'mnist':
    class_dist = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
elif args.dataset == 'fmnist':
    class_dist = [6000] * 10
elif args.dataset == 'celeba':
    class_dist = [6000] * 2
BATCH_SIZE = int(args.batch_size/6000 * class_dist[CLASS])
trn_dataset, _ = set_datasets(args.dataset, CLASS)

# construct decoder modules
hidden_dim = args.nz
G_decoder = base_module.Decoder(args.image_size, args.nc, k=args.nz, ngf=64)
netG = NetG(G_decoder)
print("netG:", netG)
netG.apply(base_module.weights_init)
if args.cuda:
    netG.cuda()

# bandwidth for Gaussian kernel in MMD
base = 1.0
sigma_list = args.sigma_list
sigma_list = [sigma / base for sigma in sigma_list]
print(sigma_list)

# put variable into cuda device
fixed_noise = torch.randn(80, args.nz, 1, 1).cuda()

# setup optimizer
optimizerG = torch.optim.RMSprop(netG.parameters(), lr=args.lr)
noise_multiplier = args.noise

minibatch_loader, _ = sampling.get_data_loaders(
    BATCH_SIZE,
    1,  # mbs, doesn't matter
    args.max_iter) # iterations
iter = 0

for x, _ in minibatch_loader(trn_dataset):
    iter_loss = 0
    x = x.cuda()
    batch_size = x.size(0)

    optimizerG.zero_grad()

    noise = torch.randn(batch_size, args.nz, 1, 1).cuda()
    y = netG(noise)

    #### compute mmd loss using my implementation ####
    DP_mmd_loss = rbf_kernel_DP_loss(x.view(batch_size, -1), y.view(batch_size, -1), sigma_list, noise_multiplier)

    errG = torch.pow(DP_mmd_loss, 2)
    errG.backward()
    optimizerG.step()
    iter_loss += errG.item()

    if iter % args.vis_step == 0:
        # training loss
        print('Current iter: ', iter, 'Total training iters: ', args.max_iter)
        print('Training loss: {}\tLoss:{:.6f}\t'.format(iter, iter_loss))
        y_fixed = netG(fixed_noise)
        y_fixed.data = y_fixed.data.mul(0.5).add(0.5)
        grid = vutils.make_grid(y_fixed.data, nrow=10)
        vutils.save_image(grid, os.path.join(checkpoints_dir, f'netG_iter{iter}_noise{args.noise}_lr{args.lr}_bs{args.batch_size}.png'))
        torch.save(netG.state_dict(), os.path.join(checkpoints_dir, f'netG_iter{iter}_noise{args.noise}_lr{args.lr}_bs{args.batch_size}.pkl'))

    iter += 1


print(f'>>> Finished DP-unconditional mmd on class {CLASS} of {args.dataset} by perturbing the loss!')
print("Time cost: ", round(time.time() - start_t, 2), "s.")
