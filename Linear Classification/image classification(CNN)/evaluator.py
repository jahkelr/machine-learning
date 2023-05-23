import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torchvision


class Evaluator:
    def __init__(self, model, classes) -> None:
        self.model = model
        self.classes = classes

    @staticmethod
    def imshow(img):
        # function to show an image
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def evaluate(self, testloader):
        dataiter = iter(testloader)
        images, labels = next(dataiter)

        # print images
        #self.imshow(torchvision.utils.make_grid(images))
        print(
            "GroundTruth: ", " ".join(f"{self.classes[labels[j]]:5s}" for j in range(4))
        )

        outputs = self.model(images)

        _, predicted = torch.max(outputs, 1)

        print(
            "Predicted: ",
            " ".join(f"{self.classes[predicted[j]]:5s}" for j in range(4)),
        )

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
        )

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}

        # again no gradients needed
        with torch.no_grad():
            for data in tqdm(
                testloader, desc="Samples evaluated", total=len(testloader)
            ):
                images, labels = data
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")
