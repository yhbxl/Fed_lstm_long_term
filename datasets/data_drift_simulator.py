#数据扰动模拟器，用于模拟噪声与特征漂移：

import torch
from torch.utils.data import Dataset

class NoisyMNISTDataset(Dataset):
    def __init__(self, dataset, noise_level=0.2, drift_factor=0.1):
        self.dataset = dataset
        self.noise_level = noise_level
        self.drift_factor = drift_factor

    def __getitem__(self, index):
        x, y = self.dataset[index]
        noise = torch.randn_like(x) * self.noise_level
        drift = torch.ones_like(x) * self.drift_factor
        return x + noise + drift, y

    def __len__(self):
        return len(self.dataset)