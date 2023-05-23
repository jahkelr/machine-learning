import os
from torch.utils.data import Dataset, random_split
from torchvision.io import read_image


class ImageDataset(Dataset):
    def __init__(self, img_dir, classes, transform=None) -> None:
        self.img_dir = img_dir
        self.image_files = []
        self.labels = []
        self.classes = []
        for i, c in enumerate(classes):
            self.image_files.append(
                [os.path.join(c, file) for file in os.listdir(os.path.join(img_dir, c))]
            )
            self.labels.append([i] * len(os.listdir(os.path.join(img_dir, c))))
            self.classes.append(c)
        self.image_files = [file for classes in self.image_files for file in classes]
        self.labels = [label for classes in self.labels for label in classes]
        assert len(self.image_files) == len(self.labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # load image and class
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


def covert_to_pt_Dataset(dir_path, classes, transform):
    dataset = ImageDataset(dir_path, classes, transform)

    trainset, testset = random_split(dataset=dataset, lengths=[0.8, 0.2])

    return trainset, testset
