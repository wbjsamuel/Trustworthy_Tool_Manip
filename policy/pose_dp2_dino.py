from typing import Any, Dict, Optional, Sequence

from omegaconf import DictConfig
import torch
import torch.nn as nn

from policy.dp2_dino import DP2 as DP2DINO
from model.pose.stage1_pose_conditioner import Stage1PoseConditioner
from common.pytorch_util import dict_apply


class DP2PoseDINO(DP2DINO):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler,
        obs_encoder,
        optimazer_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        pose_dim: int = 16,
        pose_feature_dim: int = 128,
        pose_keys: Optional[Sequence[str]] = None,
        stage1_pose_conditioner: Optional[nn.Module] = None,
        allow_pose_condition_dropout: bool = False,
        **kwargs,
    ):
        self.pose_dim = pose_dim
        self.pose_feature_dim = pose_feature_dim
        self.pose_keys = list(
            pose_keys
            or [
                "pose_history",
                "object_pose_history",
                "object_pose",
                "initial_object_pose",
                "current_tool_pose",
                "tool_pose",
                "pose",
            ]
        )
        stage1_pose_conditioner_module = stage1_pose_conditioner
        self.allow_pose_condition_dropout = allow_pose_condition_dropout

        super().__init__(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            optimazer_cfg=optimazer_cfg,
            scheduler_cfg=scheduler_cfg,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            obs_as_global_cond=obs_as_global_cond,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            **kwargs,
        )

        self.stage1_pose_conditioner = stage1_pose_conditioner_module
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, pose_feature_dim),
            nn.GELU(),
            nn.Linear(pose_feature_dim, pose_feature_dim),
        )

        obs_feature_dim = self.obs_feature_dim + pose_feature_dim
        action_dim = self.action_dim
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        self.model = self.model.__class__(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.obs_feature_dim = obs_feature_dim

    def _extract_encoder_obs(self, obs_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        encoder_keys = list(self.obs_encoder.rgb_keys) + list(self.obs_encoder.low_dim_keys)
        return {key: obs_dict[key] for key in encoder_keys if key in obs_dict}

    def _maybe_normalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        for key in ["object_pose", "pose", "current_tool_pose", "tool_pose"]:
            if key in self.normalizer.params_dict:
                return self.normalizer[key].normalize(pose)
        return pose

    def _expand_pose_history(self, pose: torch.Tensor, steps: int) -> torch.Tensor:
        if pose.ndim == 2:
            pose = pose.unsqueeze(1)
        if pose.shape[-2:] == (4, 4):
            pose = pose.reshape(*pose.shape[:-2], 16)
        if pose.shape[1] >= steps:
            return pose[:, :steps]
        pad = pose[:, -1:].repeat(1, steps - pose.shape[1], 1)
        return torch.cat([pose, pad], dim=1)

    def _resolve_pose_history(self, obs_dict: Dict[str, Any], n_obs_steps: int) -> torch.Tensor:
        for key in self.pose_keys:
            if key not in obs_dict:
                continue
            value = obs_dict[key]
            if isinstance(value, torch.Tensor):
                pose = value.to(device=self.device, dtype=torch.float32)
            else:
                pose = torch.as_tensor(value, device=self.device, dtype=torch.float32)
            return self._expand_pose_history(pose, n_obs_steps)

        if self.stage1_pose_conditioner is not None:
            if isinstance(self.stage1_pose_conditioner, Stage1PoseConditioner):
                self.stage1_pose_conditioner.to(self.device)
            return self.stage1_pose_conditioner(obs_dict, n_obs_steps).to(self.device)

        batch_size = next(
            value.shape[0] for value in obs_dict.values() if isinstance(value, torch.Tensor)
        )
        return torch.zeros(batch_size, n_obs_steps, self.pose_dim, device=self.device, dtype=self.dtype)

    def _encode_condition(
        self,
        obs_dict: Dict[str, Any],
        batch_size: int,
        horizon: int,
        n_obs_steps: int,
    ):
        nobs = dict(obs_dict)
        if "qpos" in nobs and "qpos" in self.normalizer.params_dict:
            nobs["qpos"] = self.normalizer["qpos"].normalize(nobs["qpos"])

        encoder_obs = self._extract_encoder_obs(nobs)
        this_nobs = dict_apply(
            encoder_obs,
            lambda x: x[:, :n_obs_steps, ...].reshape(-1, *x.shape[2:]),
        )
        nobs_features = self.obs_encoder(this_nobs).reshape(batch_size, n_obs_steps, -1)

        pose_history = self._resolve_pose_history(obs_dict, n_obs_steps)
        pose_history = self._maybe_normalize_pose(pose_history)
        pose_features = self.pose_encoder(pose_history.reshape(-1, pose_history.shape[-1]))
        pose_features = pose_features.reshape(batch_size, n_obs_steps, -1)

        combined_features = torch.cat([nobs_features, pose_features], dim=-1)

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            global_cond = combined_features.reshape(batch_size, -1)
            cond_data = torch.zeros(
                size=(batch_size, horizon, self.action_dim),
                device=self.device,
                dtype=self.dtype,
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            cond_data = torch.zeros(
                size=(batch_size, horizon, self.action_dim + self.obs_feature_dim),
                device=self.device,
                dtype=self.dtype,
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :n_obs_steps, self.action_dim :] = combined_features
            cond_mask[:, :n_obs_steps, self.action_dim :] = True

        return cond_data, cond_mask, local_cond, global_cond

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        value = next(value for value in obs_dict.values() if isinstance(value, torch.Tensor))
        batch_size = value.shape[0]
        cond_data, cond_mask, local_cond, global_cond = self._encode_condition(
            obs_dict=obs_dict,
            batch_size=batch_size,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
        )

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        naction_pred = nsample[..., : self.action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {"action": action, "action_pred": action_pred}

    def compute_loss(self, batch, **kwargs):
        assert "valid_mask" not in batch
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size, horizon = nactions.shape[:2]

        cond_data, _, local_cond, global_cond = self._encode_condition(
            obs_dict=batch["obs"],
            batch_size=batch_size,
            horizon=horizon,
            n_obs_steps=self.n_obs_steps,
        )

        trajectory = nactions
        if self.obs_as_global_cond:
            condition_mask = self.mask_generator(trajectory)
        else:
            cond_data = torch.cat([nactions, cond_data[..., self.action_dim :]], dim=-1)
            trajectory = cond_data.detach()
            condition_mask = self.mask_generator(trajectory)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = nn.functional.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = loss.reshape(loss.shape[0], -1).mean(dim=-1).mean()
        if "output_pred" in kwargs:
            return loss, pred
        return loss
