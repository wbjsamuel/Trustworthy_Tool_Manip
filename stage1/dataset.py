import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class Stage1Dataset(Dataset):
    """Dataset for stage 1 sequence-to-target pose prediction."""

    def __init__(
        self,
        root_dir: str = "data/stage1_data/parsed_taco_data",
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.samples: List[Dict[str, object]] = []

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(
                f"Stage1 dataset directory was not found: {self.root_dir}"
            )

        self._build_index()

    @staticmethod
    def _load_pose_sequence(tool_poses_path: str) -> np.ndarray:
        with open(tool_poses_path, "rb") as handle:
            tool_poses = np.asarray(pickle.load(handle), dtype=np.float32)

        # Stored poses come in as (1, L, 4, 4). Remove the singleton batch axis
        # so we can index the temporal dimension directly.
        tool_poses = np.squeeze(tool_poses)

        if tool_poses.ndim != 3 or tool_poses.shape[1:] != (4, 4):
            raise ValueError(
                f"Expected tool poses with shape (L, 4, 4) after squeeze, got {tool_poses.shape}"
            )

        return tool_poses

    def _build_index(self) -> None:
        for task_dir in sorted(os.listdir(self.root_dir)):
            # breakpoint()
            task_path = os.path.join(self.root_dir, task_dir)
            if not os.path.isdir(task_path):
                continue

            for seq_dir in sorted(os.listdir(task_path)):
                seq_path = os.path.join(task_path, seq_dir)
                if not os.path.isdir(seq_path):
                    continue

                tool_poses_path = os.path.join(seq_path, "tool_poses.pkl")
                rgb_dir = os.path.join(seq_path, "rgb")
                if not os.path.isfile(tool_poses_path) or not os.path.isdir(rgb_dir):
                    continue

                tool_poses = self._load_pose_sequence(tool_poses_path)

                if tool_poses.shape[0] < 2:
                    continue

                instruction = self._load_instruction(seq_path)
                target_pose = tool_poses[-1]
                num_source_frames = tool_poses.shape[0] - 1

                for frame_idx in range(num_source_frames):
                    image_path = os.path.join(rgb_dir, f"{frame_idx:06d}.png")
                    if not os.path.isfile(image_path):
                        continue

                    self.samples.append(
                        {
                            "image_path": image_path,
                            "current_tool_pose": tool_poses[frame_idx].reshape(-1).copy(),
                            "target_pose": target_pose.reshape(-1).copy(),
                            "instruction": instruction,
                            "sequence_path": seq_path,
                            "frame_idx": frame_idx,
                        }
                    )

    def _load_instruction(self, seq_path: str) -> str:
        candidates = [
            "instruction.txt",
            "instruction.md",
            "task.txt",
            "language.txt",
            "text.txt",
        ]
        for filename in candidates:
            path = os.path.join(seq_path, filename)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    text = handle.read().strip()
                if text:
                    return text
        return ""

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "current_tool_pose": torch.from_numpy(sample["current_tool_pose"]).float(),
            "instruction": sample["instruction"],
            "target_pose": torch.from_numpy(sample["target_pose"]).float(),
        }


if __name__ == "__main__":
    dataset = Stage1Dataset()
    print(f"Loaded {len(dataset)} samples.")
    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        print("Image shape:", sample["image"].shape)
        print("Instruction:", sample["instruction"] or "<empty>")
        print("Current tool pose shape:", sample["current_tool_pose"].shape)
        print("Target pose shape:", sample["target_pose"].shape)
