import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.sequences = self.data["sequences"].values
        self.labels = self.data.drop(columns=["sequences", "protein"]).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return sequence, label
