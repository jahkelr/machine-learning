import argparse

import torch

from datasets import MNIST_loaders, CIFAR10_loaders
from helpers import overlay_y_on_x, visualize_sample
from FF_Arch import Net


def main(args):
    torch.manual_seed(1234)
    if args.data == "cifar10":
        train_loader, test_loader = CIFAR10_loaders()
        net = Net([(3 * 32 * 32), 3072, 3072, 3072])
        shape = (3, 32, 32)
    elif args.data == "mnist":
        train_loader, test_loader = MNIST_loaders()
        net = Net([(28 * 28), 500, 500])
        shape = (28, 28)
    x, y = next(iter(train_loader))
    x, y = x.cpu(), y.cpu()
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])

    for data, name in zip([x, x_pos, x_neg], ["orig", "pos", "neg"]):
        visualize_sample(data, shape, name, 15)

    net.train(x_pos, x_neg)

    print("train error:", 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cpu(), y_te.cpu()

    print("test error:", 1.0 - net.predict(x_te).eq(y_te).float().mean().item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", default="mnist")
    args = parser.parse_args()
    main(args)
