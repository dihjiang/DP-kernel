#!/usr/bin/env python
# encoding: utf-8

import torch.nn as nn
import torch



# input: batch_size * k * 1 * 1
# output: batch_size * nc * image_size * image_size
class Decoder(nn.Module):
    def __init__(self, isize, nc, k=100, ngf=64):
        super(Decoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        modules = []
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(k, cngf, 4, 1, 0, bias=False),
                nn.BatchNorm2d(cngf),
                nn.ReLU(True)
            )
        )

        csize = 4
        while csize < isize // 2:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(cngf // 2),
                    nn.ReLU(True)
                )
            )
            cngf = cngf // 2
            csize = csize * 2

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        )

        self.main = nn.Sequential(*modules)

    def forward(self, input):
        output = self.main(input)
        return output


class CondDecoder(nn.Module):
    def __init__(self, isize, nc, k=100, ngf=64, num_classes=10):
        super(CondDecoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        self.k = k
        self.num_class = num_classes
        self.decoder_input = nn.Linear(k + num_classes, k)

        modules = []
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(k, cngf, 4, 1, 0, bias=False),
                nn.BatchNorm2d(cngf),
                nn.ReLU(True)
            )
        )

        csize = 4
        while csize < isize // 2:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(cngf // 2),
                    nn.ReLU(True)
                )
            )
            cngf = cngf // 2
            csize = csize * 2

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        )

        self.main = nn.Sequential(*modules)

    def forward(self, z, label):
        label = torch.nn.functional.one_hot(label, self.num_class).float()
        input = torch.cat([z, label], dim=1)
        output = self.decoder_input(input)
        output = output.view(-1, self.k, 1, 1)
        output = self.main(output)
        return output




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)
