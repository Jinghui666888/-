# utils/dataloader.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class BotLoader(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # 假设你有 'features' 和 'label' 两列
        self.features = self.data['features'].apply(eval).tolist()  # 字符串转列表
        self.labels = self.data['label'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

    def collate_fn(self, batch):
        features, labels = zip(*batch)
        features = torch.stack(features)
        labels = torch.stack(labels)
        return features, labels


# 可选：你也可以直接这样导出 PyTorch 自带的 DataLoader
DataLoader = torch.utils.data.DataLoader