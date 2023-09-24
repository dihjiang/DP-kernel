#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import torch
import torch.nn as nn
import random


def get_args(parser):
    parser.add_argument('--dataset', required=True, help='mnist | fmnist | celeba ')
    parser.add_argument('--batch_size', type=int, default=60, help='input batch size')
    parser.add_argument('--image_size', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=1, help='number of channel')
    parser.add_argument('--nz', type=int, default=10, help='size of the latent z vector')
    parser.add_argument('--max_iter', type=int, default=200002, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate, default=5e-5')
    parser.add_argument('--noise', type=float, default=0.60, help='noise multiplier')
    parser.add_argument('--gpu_device', type=int, default=0, help='using gpu device id')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--sigma_list', type=float, nargs='+')
    parser.add_argument('--vis_step', type=int, default=5000, help='generate imgs every #vis_steps steps')
    parser.add_argument('--num_samples', type=int, default=60000, help='number of generated images for computing FID')
    parser.add_argument('--compute_FID', dest='FID_flag', action='store_true', default=False, help='generate images to calculate FID')
    parser.add_argument('--ML_ACC', dest='MLACC_flag', action='store_true', default=False,
                        help='generate images to calculate ML ACC')
    parser.add_argument('--seed', type=int, default=1126, help='random_seed')
    parser.add_argument('--num_iters', type=int, default=200000, help='for loading differently trained models')
    parser.add_argument('--select', type=int, default=0, help='select a class (0-9)')
    parser.add_argument('--n_class', type=int, default=10, help='number of c')
    return parser


def set_random_seed(seed):
    """
    Sets random seeds.

    :param seed: the seed to be set for all libraries.
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # np.random.shuffle.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NNClassifier(nn.Module):
    def __init__(self, input_shape, type='MLP', data_mode='real', dataset='mnist'):
        super(NNClassifier, self).__init__()

        self.input_shape = input_shape
        self.input_channel, h, w = input_shape
        self.input_dim = self.input_channel * h * w
        self.type = type
        self.data_mode = data_mode
        if 'mnist' in dataset or 'fmnist' in dataset:
            self.output_dim = 10
        elif 'celeba' in dataset:
            self.output_dim = 2

        self.MLP = nn.Sequential(
            nn.Linear(self.input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, self.output_dim),
            nn.Softmax()
        )
        self.flatten_dim = 64 * int(h / 4) * int(w / 4)
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channel, 32, kernel_size=3, stride = 2, padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride = 2, padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(self.flatten_dim, self.output_dim),
            nn.Softmax()
        )

    def forward(self, x):
        if self.type == 'MLP':
            if self.data_mode == 'real':
                x = x.view(-1, self.input_dim)
            x = self.MLP(x)
        elif self.type == 'CNN':
            if self.data_mode == 'syn':
                x = x.view(-1, *self.input_shape)
            x = self.conv(x)
            x = x.view(-1, self.flatten_dim)
            x = self.linear(x)
        return x

def NNfit(model, train_loader, lr=0.01, num_epochs=5, data_mode='real'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_func = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for batch_idx, data_true in enumerate(train_loader):
            if data_mode == 'real':
                X, y_true = data_true
            elif data_mode == 'syn':
                X = data_true[:, :-1]
                y_true = data_true[:, -1]
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_func(y_pred, y_true.to(torch.long))
            loss.backward()
            optimizer.step()
            # Total correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            correct += (predicted == y_true.to(torch.long)).sum()
            total += X.size(0)
            # print(correct)
        print('Epoch : {} \tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
            epoch, loss.item(), float(correct * 100) / total)
        )
    return model

def evalNN(model, test_loader, data_mode='real', dataset='mnist'):
    model.eval()
    correct = 0
    total = 0

    for batch_idx, data in enumerate(test_loader):
        if dataset in ['mnist', 'fmnist']:
            X, y_true = data
            if data_mode == 'syn':
                X = X.view(X.size(0), -1)
        elif dataset in ['celeba']:
            X = data[:, :-1]
            y_true = data[:, -1]

        y_pred = model(X)
        # Total correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        correct += (predicted == y_true.to(torch.long)).sum()
        total += X.size(0)

    print("Test accuracy:{:.3f}% ".format(float(correct * 100) / total))
