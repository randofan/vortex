import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 224  # ViT default

_train_tf = A.Compose(
    [
        A.SmallestMaxSize(IMG_SIZE),
        A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=5, border_mode=cv2.BORDER_REPLICATE),
        ToTensorV2(),
    ]
)
_val_tf = A.Compose(
    [
        A.SmallestMaxSize(IMG_SIZE),
        A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_CONSTANT),
        ToTensorV2(),
    ]
)


class PaintingDataset(Dataset):
    def __init__(self, csv_path: str, train: bool = True, base_year: int = 1600):
        self.df = pd.read_csv(csv_path)
        self.tf = _train_tf if train else _val_tf
        self.base_year = base_year

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(row["path"]), cv2.COLOR_BGR2RGB)
        x = self.tf(image=img)["image"]
        y = torch.tensor(int(row["year"]) - self.base_year, dtype=torch.long)
        return x, y
