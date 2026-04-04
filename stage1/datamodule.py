from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from stage1.dataset import Stage1Dataset


class Stage1DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "data/stage1_data/parsed_taco_data",
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        seed: int = 42,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None
        self.test_dataset: Optional[torch.utils.data.Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        full_dataset = Stage1Dataset(root_dir=self.data_path)
        if len(full_dataset) == 0:
            raise RuntimeError(f"No valid samples were found under {self.data_path}.")

        if len(full_dataset) == 1:
            self.train_dataset = full_dataset
            self.val_dataset = full_dataset
            self.test_dataset = full_dataset
            return

        val_size = max(1, int(round(len(full_dataset) * self.val_split)))
        val_size = min(val_size, len(full_dataset) - 1)
        train_size = len(full_dataset) - val_size

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        self.test_dataset = self.val_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
