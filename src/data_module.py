from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import lightning as L

class ImageFolderDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_names: list[str] = []
        self.samples_per_class: list[int] = []

    def setup(self, stage: Optional[str] = None):
        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        eval_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_dataset = datasets.ImageFolder(self.data_dir / "train", transform=train_transform)
        self.val_dataset = datasets.ImageFolder(self.data_dir / "val", transform=eval_transform)
        self.test_dataset = datasets.ImageFolder(self.data_dir / "test", transform=eval_transform)

        self.class_names = self.train_dataset.classes
        counts = [0 for _ in self.class_names]
        for _, label in self.train_dataset.samples:
            counts[label] += 1
        self.samples_per_class = counts

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def class_weights(self) -> torch.Tensor:
        counts = torch.tensor(self.samples_per_class, dtype=torch.float32)
        weights = counts.sum() / torch.clamp(counts, min=1.0)
        weights = weights / weights.mean()
        return weights

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
