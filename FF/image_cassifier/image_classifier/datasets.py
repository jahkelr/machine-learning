from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import torch


def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose(
        [
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x)),
        ]
    )

    train_loader = DataLoader(
        MNIST("./data/", train=True, download=True, transform=transform),
        batch_size=train_batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        MNIST("./data/", train=False, download=True, transform=transform),
        batch_size=test_batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def CIFAR10_loaders(train_batch_size=50000, test_batch_size=10000, shuffle=True):
    transform = Compose(
        [
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            Lambda(lambda x: torch.flatten(x)),
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        CIFAR10(
            "data",
            train=True,
            download=True,
            transform=transform,
        ),
        batch_size=train_batch_size,
        shuffle=shuffle,
    )

    test_loader = torch.utils.data.DataLoader(
        CIFAR10(
            "data",
            train=False,
            transform=transform,
        ),
        batch_size=test_batch_size,
        shuffle=shuffle,
    )

    return train_loader, test_loader


def Fashion_MNIST_loaders():
    pass
