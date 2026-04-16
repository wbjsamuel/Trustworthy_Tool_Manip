from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import contextlib
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from stage1.config_utils import (
    build_stage1_model_kwargs,
    load_stage1_config,
    resolve_torch_dtype,
    resolve_stage1_checkpoint_path,
    supports_autocast,
)
from stage1.model.stage1_transformer import Stage1Transformer


def _as_tensor(data: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    return torch.as_tensor(data, device=device, dtype=dtype)


def _flatten_pose_tensor(pose: torch.Tensor, pose_dim: int) -> torch.Tensor:
    if pose.ndim >= 2 and pose.shape[-2:] == (4, 4):
        pose = pose.reshape(*pose.shape[:-2], 16)
    elif pose.shape[-1] != pose_dim:
        pose = pose.reshape(*pose.shape[:-1], pose_dim)
    return pose


class GroundingSAMDetector(nn.Module):
    """
    Best-effort detector adapter.

    It first consumes any precomputed boxes already attached to `obs_dict`, then
    optionally falls back to a Hugging Face zero-shot detector if configured.
    """

    def __init__(
        self,
        box_key: str = "boxes",
        text_prompt_key: str = "object_prompt",
        model_id: Optional[str] = None,
        threshold: float = 0.2,
    ) -> None:
        super().__init__()
        self.box_key = box_key
        self.text_prompt_key = text_prompt_key
        self.model_id = model_id
        self.threshold = threshold
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if self.model_id is None:
            return None
        try:
            transformers = importlib.import_module("transformers")
            self._pipeline = transformers.pipeline(
                task="zero-shot-object-detection",
                model=self.model_id,
            )
        except Exception:
            self._pipeline = None
        return self._pipeline

    def forward(
        self,
        images: torch.Tensor,
        obs_dict: Dict[str, Any],
        prompts: Sequence[str],
    ) -> Optional[torch.Tensor]:
        if self.box_key in obs_dict:
            boxes = obs_dict[self.box_key]
            boxes = _as_tensor(boxes, device=images.device, dtype=torch.float32)
            if boxes.ndim == 2:
                boxes = boxes.unsqueeze(1)
            return boxes

        pipeline = self._load_pipeline()
        if pipeline is None:
            return None

        pil_images: List[Image.Image] = []
        for image in images.detach().cpu():
            chw = image.clamp(0.0, 1.0)
            pil_images.append(transforms.ToPILImage()(chw))

        detections = []
        for image, prompt in zip(pil_images, prompts):
            result = pipeline(image, candidate_labels=[prompt or "object"], threshold=self.threshold)
            if len(result) == 0:
                detections.append(torch.zeros(1, 4, dtype=torch.float32))
                continue
            box = result[0]["box"]
            detections.append(
                torch.tensor(
                    [[box["xmin"], box["ymin"], box["xmax"], box["ymax"]]],
                    dtype=torch.float32,
                )
            )
        return torch.stack(detections, dim=0).to(images.device)


class SAM3DPointCloudBuilder(nn.Module):
    """
    Converts a 2D box and depth/intrinsics into a masked point cloud.

    If a binary mask is already available, it is used directly. Otherwise the
    box is rasterized into a coarse mask.
    """

    def __init__(
        self,
        depth_key: str = "depth",
        intrinsics_key: str = "camera_intrinsics",
        mask_key: str = "masks",
    ) -> None:
        super().__init__()
        self.depth_key = depth_key
        self.intrinsics_key = intrinsics_key
        self.mask_key = mask_key

    def _select_first_frame(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim >= 5:
            return value[:, 0]
        if value.ndim >= 4:
            return value[:, 0]
        return value

    def forward(
        self,
        obs_dict: Dict[str, Any],
        boxes: Optional[torch.Tensor],
        image_shape: Sequence[int],
        device: torch.device,
    ) -> Optional[List[torch.Tensor]]:
        if self.depth_key not in obs_dict or self.intrinsics_key not in obs_dict:
            return None

        depth = _as_tensor(obs_dict[self.depth_key], device=device, dtype=torch.float32)
        intrinsics = _as_tensor(obs_dict[self.intrinsics_key], device=device, dtype=torch.float32)
        depth = self._select_first_frame(depth)
        intrinsics = self._select_first_frame(intrinsics)

        masks = None
        if self.mask_key in obs_dict:
            masks = _as_tensor(obs_dict[self.mask_key], device=device, dtype=torch.float32)
            masks = self._select_first_frame(masks)

        batch_points: List[torch.Tensor] = []
        _, height, width = image_shape
        ys = torch.arange(height, device=device, dtype=torch.float32)
        xs = torch.arange(width, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        for batch_idx in range(depth.shape[0]):
            depth_map = depth[batch_idx]
            if depth_map.ndim == 3:
                depth_map = depth_map.squeeze(0)

            if masks is not None:
                mask = masks[batch_idx]
                if mask.ndim == 3:
                    mask = mask.squeeze(0)
                mask = mask > 0.5
            elif boxes is not None:
                mask = torch.zeros((height, width), device=device, dtype=torch.bool)
                x0, y0, x1, y1 = boxes[batch_idx, 0].round().long()
                x0 = x0.clamp(0, width - 1)
                x1 = x1.clamp(0, width - 1)
                y0 = y0.clamp(0, height - 1)
                y1 = y1.clamp(0, height - 1)
                mask[y0 : y1 + 1, x0 : x1 + 1] = True
            else:
                batch_points.append(torch.empty(0, 3, device=device, dtype=torch.float32))
                continue

            valid = mask & torch.isfinite(depth_map) & (depth_map > 0)
            if not valid.any():
                batch_points.append(torch.empty(0, 3, device=device, dtype=torch.float32))
                continue

            z = depth_map[valid]
            x = grid_x[valid]
            y = grid_y[valid]
            k = intrinsics[batch_idx]
            fx = k[0, 0]
            fy = k[1, 1]
            cx = k[0, 2]
            cy = k[1, 2]
            points = torch.stack(
                [
                    (x - cx) * z / fx,
                    (y - cy) * z / fy,
                    z,
                ],
                dim=-1,
            )
            batch_points.append(points)

        return batch_points


class FoundationPoseEstimator(nn.Module):
    """
    Lightweight pose initializer.

    If an upstream system already provides `object_pose`, this module is bypassed.
    Otherwise it estimates a rigid pose from the segmented point cloud using the
    centroid and PCA axes as a practical fallback.
    """

    def __init__(
        self,
        pose_key_candidates: Optional[Sequence[str]] = None,
        pose_dim: int = 16,
    ) -> None:
        super().__init__()
        self.pose_key_candidates = list(
            pose_key_candidates
            or [
                "initial_object_pose",
                "object_pose",
                "current_tool_pose",
                "tool_pose",
                "pose",
            ]
        )
        self.pose_dim = pose_dim

    def _extract_pose_from_obs(
        self,
        obs_dict: Dict[str, Any],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        for key in self.pose_key_candidates:
            if key not in obs_dict:
                continue
            pose = _as_tensor(obs_dict[key], device=device, dtype=torch.float32)
            if pose.ndim >= 3 and pose.shape[-2:] != (4, 4):
                pose = pose[:, 0]
            pose = _flatten_pose_tensor(pose, self.pose_dim)
            return pose[:, : self.pose_dim]
        return None

    def forward(
        self,
        obs_dict: Dict[str, Any],
        point_clouds: Optional[List[torch.Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        pose = self._extract_pose_from_obs(obs_dict, batch_size, device)
        if pose is not None:
            return pose

        identity = torch.eye(4, device=device, dtype=torch.float32).reshape(1, 16).repeat(batch_size, 1)
        if point_clouds is None:
            return identity

        output = []
        for points in point_clouds:
            if points.numel() == 0:
                output.append(torch.eye(4, device=device, dtype=torch.float32))
                continue
            center = points.mean(dim=0)
            centered = points - center
            try:
                _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                rotation = vh.transpose(0, 1)
            except RuntimeError:
                rotation = torch.eye(3, device=device, dtype=torch.float32)

            transform = torch.eye(4, device=device, dtype=torch.float32)
            transform[:3, :3] = rotation
            transform[:3, 3] = center
            output.append(transform)

        return torch.stack(output, dim=0).reshape(batch_size, 16)


class InitialPosePipeline(nn.Module):
    def __init__(
        self,
        detector: Optional[nn.Module] = None,
        sam3d: Optional[nn.Module] = None,
        foundationpose: Optional[nn.Module] = None,
        default_prompt: str = "object",
        instruction_key: str = "instruction",
        object_prompt_key: str = "object_prompt",
    ) -> None:
        super().__init__()
        self.detector = detector or GroundingSAMDetector()
        self.sam3d = sam3d or SAM3DPointCloudBuilder()
        self.foundationpose = foundationpose or FoundationPoseEstimator()
        self.default_prompt = default_prompt
        self.instruction_key = instruction_key
        self.object_prompt_key = object_prompt_key

    def _get_prompts(self, obs_dict: Dict[str, Any], batch_size: int) -> List[str]:
        prompt_value = obs_dict.get(self.object_prompt_key, obs_dict.get(self.instruction_key, self.default_prompt))
        if isinstance(prompt_value, str):
            return [prompt_value] * batch_size
        if isinstance(prompt_value, (list, tuple)):
            prompts = list(prompt_value)
            if len(prompts) == batch_size:
                return [str(prompt) for prompt in prompts]
        return [self.default_prompt] * batch_size

    def forward(self, obs_dict: Dict[str, Any], images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        prompts = self._get_prompts(obs_dict, batch_size)
        boxes = self.detector(images, obs_dict, prompts) if self.detector is not None else None
        point_clouds = (
            self.sam3d(obs_dict, boxes, images.shape[1:], images.device)
            if self.sam3d is not None
            else None
        )
        return self.foundationpose(obs_dict, point_clouds, batch_size, images.device)


class Stage1PoseConditioner(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        camera_key: Optional[str] = None,
        pose_dim: int = 16,
        instruction_key: str = "instruction",
        default_instruction: str = "",
        image_is_normalized: bool = True,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.256, 0.225),
        initial_pose_pipeline: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.camera_key = camera_key
        self.pose_dim = pose_dim
        self.instruction_key = instruction_key
        self.default_instruction = default_instruction
        self.image_is_normalized = image_is_normalized
        self.register_buffer("image_mean", torch.tensor(image_mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std, dtype=torch.float32).view(1, -1, 1, 1))
        self.initial_pose_pipeline = initial_pose_pipeline or InitialPosePipeline()
        self._config: Optional[dict] = None
        # Keep the frozen stage-1 checkpoint out of the module tree so it does not
        # unexpectedly appear in Lightning checkpoints or EMA parameter snapshots.
        self._runtime_cache: Dict[str, Any] = {"stage1_model": None}

    def _resolve_config_path(self) -> Path:
        if self.config_path is not None:
            return Path(self.config_path)
        return Path(__file__).resolve().parents[2] / "stage1" / "config" / "stage1.yaml"

    def _load_config(self) -> dict:
        if self._config is None:
            config_path = self._resolve_config_path()
            self._config = load_stage1_config(str(config_path))
        return self._config

    def _load_stage1_model(self, device: torch.device) -> Stage1Transformer:
        cached_model = self._runtime_cache["stage1_model"]
        if cached_model is not None:
            inference_dtype = resolve_torch_dtype(self._load_config().get("inference", {}).get("model_dtype"))
            if inference_dtype is not None and device.type == "cuda":
                cached_model.to(device=device, dtype=inference_dtype)
            else:
                cached_model.to(device)
            return cached_model

        config = self._load_config()
        checkpoint_path = resolve_stage1_checkpoint_path(self.checkpoint_path)
        model = Stage1Transformer.load_from_checkpoint(
            str(checkpoint_path),
            map_location=device,
            **build_stage1_model_kwargs(config),
        )
        inference_dtype = resolve_torch_dtype(config.get("inference", {}).get("model_dtype"))
        model.eval()
        if inference_dtype is not None and device.type == "cuda":
            model.to(device=device, dtype=inference_dtype)
        else:
            model.to(device)
        model.requires_grad_(False)
        self._runtime_cache["stage1_model"] = model
        return model

    def _choose_camera_key(self, obs_dict: Dict[str, Any]) -> str:
        if self.camera_key is not None and self.camera_key in obs_dict:
            return self.camera_key
        for key, value in obs_dict.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 5 and value.shape[2] == 3:
                return key
        raise KeyError("No RGB observation key found for Stage1 pose conditioning.")

    def _prepare_images(self, obs_dict: Dict[str, Any]) -> torch.Tensor:
        camera_key = self._choose_camera_key(obs_dict)
        images = obs_dict[camera_key]
        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(images)
        images = images.to(device=self.image_mean.device, dtype=torch.float32)
        if images.ndim != 5:
            raise ValueError(f"Expected image history with shape [B, T, C, H, W], got {tuple(images.shape)}")
        if self.image_is_normalized:
            images = images * self.image_std + self.image_mean
        return images.clamp(0.0, 1.0)

    def _get_instructions(self, obs_dict: Dict[str, Any], batch_size: int) -> List[str]:
        raw_value = obs_dict.get(self.instruction_key, self.default_instruction)
        if isinstance(raw_value, str):
            return [raw_value] * batch_size
        if isinstance(raw_value, (list, tuple)):
            values = [str(item) for item in raw_value]
            if len(values) == batch_size:
                return values
        return [self.default_instruction] * batch_size

    @torch.no_grad()
    def forward(self, obs_dict: Dict[str, Any], n_obs_steps: int) -> torch.Tensor:
        device = self.image_mean.device
        images = self._prepare_images(obs_dict).to(device)
        batch_size, time_steps = images.shape[:2]
        steps = min(n_obs_steps, time_steps)
        stage1_model = self._load_stage1_model(device)
        autocast_dtype = resolve_torch_dtype(self._load_config().get("inference", {}).get("autocast_dtype"))
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if supports_autocast(device, autocast_dtype)
            else contextlib.nullcontext()
        )
        instructions = self._get_instructions(obs_dict, batch_size)

        initial_pose = self.initial_pose_pipeline(obs_dict, images[:, 0]).to(device=device, dtype=torch.float32)
        current_pose = _flatten_pose_tensor(initial_pose, self.pose_dim)
        pose_history = [current_pose]

        with autocast_context:
            for step_idx in range(steps - 1):
                predicted_pose = stage1_model(images[:, step_idx], current_pose, instructions)
                current_pose = _flatten_pose_tensor(predicted_pose, self.pose_dim).to(torch.float32)
                pose_history.append(current_pose)

        pose_history_tensor = torch.stack(pose_history, dim=1)
        if pose_history_tensor.shape[1] < n_obs_steps:
            pad = pose_history_tensor[:, -1:].repeat(1, n_obs_steps - pose_history_tensor.shape[1], 1)
            pose_history_tensor = torch.cat([pose_history_tensor, pad], dim=1)
        return pose_history_tensor[:, :n_obs_steps]
