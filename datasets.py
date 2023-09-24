import numpy as np
import torch
from torchvision import datasets, transforms
import os


# Load the dataset
def set_datasets(dataset, CLASS):
    if dataset in ['mnist', 'fmnist']:
        T = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5,)),
        ])
        if dataset == 'mnist':
            train_set = datasets.MNIST('./data', train=True, download=True, transform=T)
            test_set = datasets.MNIST('./data', train=False, download=True, transform=T)
        elif dataset == 'fmnist':
            train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=T)
            test_set = datasets.FashionMNIST('./data', train=False, download=True, transform=T)

        if CLASS is not None:
            idx = train_set.targets == CLASS
            train_set.targets = train_set.targets[idx]
            train_set.data = train_set.data[idx]

    elif dataset in ['celeba']:
        celeba_dist = [30000] * 2
        T = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # change the path of dataset
        train_set = datasets.CelebA('Replace with your path', split='train', download=True, transform=T)
        test_set = datasets.CelebA('Replace with your path', split='test', download=True, transform=T)

        if CLASS is not None:
            # filter male or female, by attribute 20
            indices_full = []
            for i in range(len(train_set)):
                if train_set[i][1][20] == CLASS:
                    indices_full.append(i)
                if len(indices_full) == celeba_dist[CLASS]:
                    break
            indices_full = np.array(indices_full)
            train_set = torch.utils.data.Subset(train_set, indices_full)
        else:
            indices_full = np.arange(len(train_set))
            np.random.shuffle(indices_full)
            train_set = torch.utils.data.Subset(train_set, indices_full[:60000])
            print('len(train_set):', len(train_set))

    return train_set, test_set


