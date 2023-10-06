import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision
import torch.optim as optim


class Trainer:
    def __init__(self, model, classes, epochs=2, save_path="./model.pth") -> None:
        self.model = model
        self.classes = classes
        self.epochs = epochs
        self.save_path = save_path

    @staticmethod
    def imshow(img):
        # function to show an image
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def save_model_state_dict(self):
        torch.save(self.model.state_dict(), self.save_path)

    def train(self, trainloader, batch_size, feedback_time=2500):
        # get some random training images
        dataiter = iter(trainloader)
        images, labels = next(dataiter)

        # show images
        # self.imshow(torchvision.utils.make_grid(images))
        # print labels
        if isinstance(labels[0], str):
            print(" ".join(f"{labels[j]:5s}" for j in range(batch_size)))
        else:
            print(" ".join(f"{self.classes[labels[j]]:5s}" for j in range(batch_size)))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(self.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in tqdm(
                enumerate(trainloader, 0),
                desc="Samples per epoch",
                total=len(trainloader),
            ):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                labels = torch.Tensor(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % feedback_time == 0:  # print every feedback_time mini-batches
                    print(
                        f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / feedback_time:.3f}"
                    )
                    running_loss = 0.0
