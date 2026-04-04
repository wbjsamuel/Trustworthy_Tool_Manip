import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


class Stage1PoseDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        resolution: int = 224,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
            ]
        )
        self.samples: List[Dict[str, object]] = []

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self._build_index()

    def _load_instruction(self, seq_path: Path) -> str:
        for filename in ["instruction.txt", "instruction.md", "task.txt", "language.txt", "text.txt"]:
            candidate = seq_path / filename
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8").strip()
                if text:
                    return text
        return seq_path.parent.name.replace("_", " ")

    def _build_index(self) -> None:
        for task_dir in sorted(self.root_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for seq_dir in sorted(task_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue

                rgb_dir = seq_dir / "rgb"
                pose_path = seq_dir / "tool_poses.pkl"
                if not rgb_dir.is_dir() or not pose_path.is_file():
                    continue

                with open(pose_path, "rb") as handle:
                    tool_poses = np.asarray(pickle.load(handle), dtype=np.float32)

                if tool_poses.ndim != 2 or tool_poses.shape[0] < 2:
                    continue

                instruction = self._load_instruction(seq_dir)
                num_frames = tool_poses.shape[0]

                for frame_idx in range(num_frames - 1):
                    current_image_path = rgb_dir / f"{frame_idx:06d}.png"
                    next_image_path = rgb_dir / f"{frame_idx + 1:06d}.png"
                    if not current_image_path.is_file():
                        continue
                    if not next_image_path.is_file():
                        continue

                    current_pose = tool_poses[frame_idx].copy()
                    next_pose = tool_poses[frame_idx + 1].copy()
                    self.samples.append(
                        {
                            "current_image_path": current_image_path,
                            "target_image_path": next_image_path,
                            "current_tool_pose": current_pose,
                            "target_pose": next_pose,
                            "pose_delta": (next_pose - current_pose).copy(),
                            "instruction": instruction,
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        current_image = self.transform(Image.open(sample["current_image_path"]).convert("RGB"))
        target_image = self.transform(Image.open(sample["target_image_path"]).convert("RGB"))
        videos = torch.stack([current_image, target_image], dim=0)
        return {
            "videos": videos,
            "task_instruction": sample["instruction"],
            "current_tool_pose": torch.from_numpy(sample["current_tool_pose"]).float(),
            "target_pose": torch.from_numpy(sample["target_pose"]).float(),
            "pose_delta": torch.from_numpy(sample["pose_delta"]).float(),
        }


class LightningStage1PoseDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str = "data/stage1_data/parsed_taco_dataset",
        batch_size: int = 16,
        resolution: int = 224,
        num_workers: int = 4,
        val_split: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.resolution = resolution
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        full_dataset = Stage1PoseDataset(
            root_dir=self.data_root,
            resolution=self.resolution,
        )
        if len(full_dataset) == 0:
            raise RuntimeError(f"No valid samples were found under {self.data_root}.")

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
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
