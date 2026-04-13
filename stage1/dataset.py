import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def flatten_pose_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected pose matrix with shape (4, 4), got {pose.shape}")
    return pose.reshape(-1).copy()


def compute_delta_pose(current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    current_pose = np.asarray(current_pose, dtype=np.float32)
    target_pose = np.asarray(target_pose, dtype=np.float32)
    if current_pose.shape != (4, 4) or target_pose.shape != (4, 4):
        raise ValueError(
            "Expected current_pose and target_pose to both have shape (4, 4), "
            f"got {current_pose.shape} and {target_pose.shape}"
        )
    delta_pose = np.linalg.inv(current_pose) @ target_pose
    return delta_pose.astype(np.float32, copy=False)


def apply_delta_pose(current_pose: np.ndarray, delta_pose: np.ndarray) -> np.ndarray:
    current_pose = np.asarray(current_pose, dtype=np.float32)
    delta_pose = np.asarray(delta_pose, dtype=np.float32)
    if current_pose.shape != (4, 4) or delta_pose.shape != (4, 4):
        raise ValueError(
            "Expected current_pose and delta_pose to both have shape (4, 4), "
            f"got {current_pose.shape} and {delta_pose.shape}"
        )
    next_pose = current_pose @ delta_pose
    return next_pose.astype(np.float32, copy=False)


def _format_noun_phrase(text: str, add_article: bool = False) -> str:
    phrase = text.replace("_", " ").strip()
    if not phrase:
        return phrase

    lowered = phrase.lower()
    if lowered.startswith(("a ", "an ", "the ")):
        return phrase
    if add_article:
        return f"the {phrase}"
    return phrase


def format_task_name_as_instruction(task_name: str) -> str:
    normalized = task_name.strip().replace("-", "_")
    if not normalized:
        return ""

    parts = [part for part in normalized.split("_") if part]
    if len(parts) >= 3:
        action = parts[0].replace("_", " ")
        tool = _format_noun_phrase(parts[1])
        target = _format_noun_phrase(" ".join(parts[2:]), add_article=True)
        return f"use {tool} to {action} {target}"

    phrase = normalized.replace("_", " ")
    return phrase


class BaseStage1Dataset(Dataset):
    """Base dataset for stage 1 pose prediction tasks."""

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

                self._build_sequence_samples(seq_path, rgb_dir, tool_poses)

    def _build_sequence_samples(
        self,
        seq_path: str,
        rgb_dir: str,
        tool_poses: np.ndarray,
    ) -> None:
        raise NotImplementedError

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
        task_name = os.path.basename(os.path.dirname(seq_path))
        return format_task_name_as_instruction(task_name)

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


class Stage1Dataset(BaseStage1Dataset):
    """Dataset for stage 1 next-pose prediction."""

    def _build_sequence_samples(
        self,
        seq_path: str,
        rgb_dir: str,
        tool_poses: np.ndarray,
    ) -> None:
        instruction = self._load_instruction(seq_path)
        num_source_frames = tool_poses.shape[0] - 1

        for frame_idx in range(num_source_frames):
            image_path = os.path.join(rgb_dir, f"{frame_idx:06d}.png")
            if not os.path.isfile(image_path):
                continue

            self.samples.append(
                {
                    "image_path": image_path,
                    "current_tool_pose": flatten_pose_matrix(tool_poses[frame_idx]),
                    "target_pose": flatten_pose_matrix(tool_poses[frame_idx + 1]),
                    "instruction": instruction,
                    "sequence_path": seq_path,
                    "frame_idx": frame_idx,
                }
            )


class Stage1TargetPoseDataset(BaseStage1Dataset):
    """Dataset for stage 1 final-target pose prediction."""

    def _build_sequence_samples(
        self,
        seq_path: str,
        rgb_dir: str,
        tool_poses: np.ndarray,
    ) -> None:
        instruction = self._load_instruction(seq_path)
        target_pose = flatten_pose_matrix(tool_poses[-1])
        num_source_frames = tool_poses.shape[0] - 1

        for frame_idx in range(num_source_frames):
            image_path = os.path.join(rgb_dir, f"{frame_idx:06d}.png")
            if not os.path.isfile(image_path):
                continue

            self.samples.append(
                {
                    "image_path": image_path,
                    "current_tool_pose": flatten_pose_matrix(tool_poses[frame_idx]),
                    "target_pose": target_pose,
                    "instruction": instruction,
                    "sequence_path": seq_path,
                    "frame_idx": frame_idx,
                }
            )


class Stage1DeltaPoseDataset(BaseStage1Dataset):
    """Dataset for stage 1 relative next-pose prediction."""

    def _build_sequence_samples(
        self,
        seq_path: str,
        rgb_dir: str,
        tool_poses: np.ndarray,
    ) -> None:
        instruction = self._load_instruction(seq_path)
        num_source_frames = tool_poses.shape[0] - 1

        for frame_idx in range(num_source_frames):
            image_path = os.path.join(rgb_dir, f"{frame_idx:06d}.png")
            if not os.path.isfile(image_path):
                continue

            current_pose = tool_poses[frame_idx]
            next_pose = tool_poses[frame_idx + 1]
            delta_pose = compute_delta_pose(current_pose, next_pose)
            self.samples.append(
                {
                    "image_path": image_path,
                    "current_tool_pose": flatten_pose_matrix(current_pose),
                    "target_pose": flatten_pose_matrix(delta_pose),
                    "instruction": instruction,
                    "sequence_path": seq_path,
                    "frame_idx": frame_idx,
                }
            )


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
