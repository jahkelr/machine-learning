import torch
import torchvision.transforms as transforms
import torchvision
from trainer import Trainer
from evaluator import Evaluator
from cnn_cifar import Net as cifarNet
from cnn_tod import Net
import utils


def init_cifar10_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 8

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    classes = trainset.classes

    return transform, batch_size, trainset, trainloader, testset, testloader, classes


def init_time_data():
    transform = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    batch_size = 8

    classes = ["daytime", "nighttime", "sunrise"]

    trainset, testset = utils.covert_to_pt_Dataset(
        "./data",
        classes=classes,
        transform=transform,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return transform, batch_size, trainset, trainloader, testset, testloader, classes

# Test dataset
'''
(
    transform,
    batch_size,
    trainset,
    trainloader,
    testset,
    testloader,
    classes,
) = init_cifar10_data()
'''

# Replace test dataset
(
    transform,
    batch_size,
    trainset,
    trainloader,
    testset,
    testloader,
    classes,
) = init_time_data()

# Create a CNN
# model = cifarNet() # Test net
model = Net()

# Run train script
trainer = Trainer(model, classes)
trainer.train(trainloader, batch_size, 50)
trainer.save_model_state_dict()

# Run eval script
model.load_state_dict(torch.load("./model.pth"))
model.eval()

evaluator = Evaluator(model, classes)
evaluator.evaluate(testloader)
