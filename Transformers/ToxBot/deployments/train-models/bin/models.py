import torch.nn as nn


class ToxicityClassifier(nn.Module):
    def __init__(self) -> None:
        pass

    def load_model(self, path: str):
        return path

    def __call__(self, data):
        return data


class ToxicityGenerator(nn.Module):
    def __init__(self) -> None:
        pass

    def load_model(self, path: str):
        return path

    def __call__(self, data):
        return data
