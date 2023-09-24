'''
Test script
'''
import argparse
from argparse import Namespace
import os
import torch
import numpy as np
import torch.nn as nn
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import base_module
import time
from util import *
from datasets import *
import random

def _init_fn():
    np.random.seed(12)


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

def main():
    # Get argument
    parser = argparse.ArgumentParser()
    parser = get_args(parser)
    args = parser.parse_args()

    device = torch.device('cuda')

    set_random_seed(args.seed)
    print(f"****** Random seed = {args.seed} ******")

    trn_dataset, tst_dataset = set_datasets(args.dataset, None)
    testloader = DataLoader(tst_dataset, batch_size=200, drop_last=False, worker_init_fn=_init_fn)

    # construct decoder modules
    G_decoder = base_module.CondDecoder(args.image_size, args.nc, k=args.nz, ngf=64, num_classes=10)
    netG = CNetG(G_decoder)
    netG = netG.cuda()

    model_path = f'{args.experiment}/netG_iter{args.num_iters}_noise{args.noise}_lr{args.lr}_bs{args.batch_size}.pkl'
    print(model_path)
    netG.load_state_dict(torch.load(model_path), strict=False)
    netG.eval()

    if args.MLACC_flag:
        # Compute ML ACC when training on synthetic data, then test on real data
        real_test = []
        syn_train = []

        for _, (x, y) in enumerate(testloader):
            x = x.view(x.size(0), -1) # flatten
            if args.dataset == 'celeba':
                y = y[:, 20] # only need the gender label
            y = torch.unsqueeze(y, 1)
            real_test.append(torch.cat([x, y], dim=1))

        del y
        for _ in range(int(args.num_samples / 100)):
            if 'mnist' in args.dataset or 'fmnist' in args.dataset:
                y = torch.arange(10).repeat(10)
            elif args.dataset == 'celeba':
                y = torch.cat([torch.zeros(50), torch.ones(50)])
                y = y.to(torch.int64)
            z = torch.randn(100, args.nz).cuda()
            images = netG(z, y.cuda()).detach().cpu()
            images = images.view(images.size(0), -1)
            y = torch.unsqueeze(y, 1).float()
            syn_train.append(torch.cat([images, y], dim=1))

        syn_train_loader = DataLoader(torch.cat(syn_train, dim=0), batch_size=200, shuffle=True, worker_init_fn=_init_fn)

        print('==========CNN===========')
        if args.dataset == 'celeba':
            testloader = DataLoader(torch.cat(real_test, dim=0), batch_size=200, shuffle=True, worker_init_fn=_init_fn)
            model = NNClassifier((args.nc, args.image_size, args.image_size), type='CNN', data_mode='syn', dataset=args.dataset)
            model = NNfit(model, syn_train_loader, lr=0.001, num_epochs=100, data_mode='syn')
            print('==> syn data:')
            evalNN(model, testloader, data_mode='syn', dataset='celeba')
        else:
            model = NNClassifier((args.nc, args.image_size, args.image_size), type='CNN', data_mode='syn', dataset=args.dataset)
            model = NNfit(model, syn_train_loader, lr=0.001, num_epochs=5, data_mode='syn')
            print('==> syn data:')
            evalNN(model, testloader, data_mode='syn')

    elif args.FID_flag:
        saveImgDir = f'./Imgs/{args.dataset}/'
        if not os.path.exists(saveImgDir):
            os.makedirs(f'{saveImgDir}True/')
            print(f'Make Dir:{saveImgDir}True/')

        if not os.path.exists(f'{saveImgDir}Gen/{args.num_samples}/'):
            os.makedirs(f'{saveImgDir}Gen/{args.num_samples}/')
            print(f'Make Dir:{saveImgDir}Gen/{args.num_samples}/')

        if len(os.listdir(f'{saveImgDir}True/')) == 0:
            ii = 0
            for _, (x, y) in enumerate(testloader):
                for xi in x:
                    vutils.save_image(xi, f'{saveImgDir}True/imgs_{ii}.png')
                    ii += 1
            print('Total: ', ii, ' true test images.')

        ii = 0
        for _ in range(int(args.num_samples / 100)):
            if 'mnist' in args.dataset or 'fmnist' in args.dataset:
                y = torch.arange(10).repeat(10)
            elif args.dataset == 'celeba':
                y = torch.cat([torch.zeros(50), torch.ones(50)])
                y = y.to(torch.int64)
            z = torch.randn(100, args.nz).cuda()
            images = netG(z, y.cuda()).detach().cpu()
            for img in images:
                vutils.save_image(img, f'{saveImgDir}Gen/{args.num_samples}/imgs_{ii}.png')
                ii += 1
        print('Generated ', ii, ' images.')
    else:
        if args.dataset in ['mnist', 'fmnist']:
            z = torch.randn(80, args.nz).cuda()
            y = torch.arange(10).repeat(8).cuda()
        elif args.dataset == 'celeba':
            z = torch.randn(20, args.nz).cuda()
            y = torch.cat([torch.zeros(10), torch.ones(10)]).cuda()
            y = y.to(torch.int64)

        images = netG(z, y.cuda()).detach().cpu()
        images = images.mul(0.5).add(0.5)
        grid = vutils.make_grid(images.data, nrow=10)
        vutils.save_image(grid, f'DP-kernel_{args.dataset}_iter{args.num_iters}_noise{args.noise}_lr{args.lr}_bs{args.batch_size}.png')


if __name__ == '__main__':
    """
    entry point.
    """

    start_t = time.time()
    main()
    print("Time cost: ", round(time.time() - start_t, 2), "s.")
