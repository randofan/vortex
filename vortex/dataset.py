import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 224  # ViT default

_tf = A.Compose([
    A.LongestMaxSize(224),  # Scale so longest side = 224, preserving aspect ratio
    A.PadIfNeeded(224, 224, border_mode=cv2.BORDER_REFLECT_101),  # Pad to exactly 224x224
    ToTensorV2(),
])


class PaintingDataset(Dataset):
    def __init__(self, csv_path: str, base_year: int = 1600):
        self.df = pd.read_csv(csv_path)
        self.tf = _tf
        self.base_year = base_year

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(row.path), cv2.COLOR_BGR2RGB)
        x = self.tf(image=img)["image"]
        y = torch.tensor(int(row.year) - self.base_year, dtype=torch.long)
        return x, y
