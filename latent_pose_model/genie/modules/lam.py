from typing import Dict, List

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torchvision import transforms
from transformers import AutoTokenizer, T5EncoderModel

from latent_pose_model.genie.modules.blocks import (
    SpatioTemporalTransformer,
    SpatioTransformer,
    VectorQuantizer,
)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class PoseTokenEncoder(nn.Module):
    def __init__(self, pose_dim: int, dino_dim: int, num_pose_tokens: int) -> None:
        super().__init__()
        self.num_pose_tokens = num_pose_tokens
        self.token_mlp = nn.Sequential(
            nn.Linear(pose_dim * 3, dino_dim * num_pose_tokens),
            nn.GELU(),
            nn.Linear(dino_dim * num_pose_tokens, dino_dim * num_pose_tokens),
        )

    def forward(
        self,
        current_tool_pose: Tensor,
        target_pose: Tensor,
        pose_delta: Tensor,
    ) -> Tensor:
        pose_features = torch.cat([current_tool_pose, target_pose, pose_delta], dim=-1)
        tokens = self.token_mlp(pose_features)
        return tokens.view(current_tool_pose.shape[0], self.num_pose_tokens, -1)


class UncontrolledDINOLatentPoseModel(nn.Module):
    """
    Stage-1 latent pose VQ-VAE operating in DINO feature space.
    """

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        latent_dim: int,
        num_latents: int,
        patch_size: int,
        enc_blocks: int,
        dec_blocks: int,
        num_heads: int,
        pose_dim: int = 7,
        num_pose_tokens: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.num_pose_tokens = num_pose_tokens

        self.dino_transform = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        )
        self.dino_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768
        self.pose_token_encoder = PoseTokenEncoder(
            pose_dim=pose_dim,
            dino_dim=dino_dim,
            num_pose_tokens=num_pose_tokens,
        )

        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )
        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.pose_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.text_encoder = T5EncoderModel.from_pretrained("t5-base")
        self.text_encoder.requires_grad_(False)
        self.lang_proj = nn.Linear(768, model_dim)
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")

    def encode_text(self, lang: List[str]):
        encoding = self.tokenizer(lang, return_tensors="pt", padding=True, truncation=True)
        encoding = {key: value.to(self.device) for key, value in encoding.items()}
        with torch.no_grad():
            encoder_outputs = self.text_encoder(**encoding)
        return encoder_outputs.last_hidden_state, encoding["attention_mask"]

    def vq_encode(self, batch: Dict, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        videos = batch["videos"]
        current_tool_pose = batch["current_tool_pose"]
        target_pose = batch["target_pose"]
        pose_delta = batch["pose_delta"]

        bsz, num_frames = videos.shape[:2]
        videos = rearrange(videos, "b t c h w -> (b t) c h w")
        videos = self.dino_transform(videos)
        dino_features = self.dino_encoder.forward_features(videos)["x_norm_patchtokens"]
        dino_features = rearrange(dino_features, "(b t) l d -> b t l d", b=bsz, t=num_frames)

        pose_tokens = self.pose_token_encoder(current_tool_pose, target_pose, pose_delta)
        pose_tokens = pose_tokens.unsqueeze(1).expand(-1, num_frames, -1, -1)
        padded_patches = torch.cat([pose_tokens, dino_features], dim=2)

        z = self.encoder(padded_patches, lang_embed, attention_mask)
        z = self.to_codebook(z[:, 1:, : self.num_pose_tokens])
        z = z.reshape(bsz * (num_frames - 1), self.num_pose_tokens, self.latent_dim)
        z_q, z, emb, indices = self.vq(z)
        z_q = z_q.reshape(bsz, num_frames - 1, self.num_pose_tokens, self.latent_dim)

        return {
            "patches": dino_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "indices": indices,
        }

    def forward(self, batch: Dict) -> Dict:
        bsz, num_frames = batch["videos"].shape[:2]
        height, width = batch["videos"].shape[3:5]

        lang_embed, attention_mask = self.encode_text(batch["task_instruction"])
        lang_embed = self.lang_proj(lang_embed)
        attention_mask = torch.cat(
            [
                torch.ones(
                    (bsz, self.num_pose_tokens + (height // self.patch_size) ** 2),
                    device=self.device,
                    dtype=attention_mask.dtype,
                ),
                attention_mask,
            ],
            dim=-1,
        )

        outputs = self.vq_encode(
            batch,
            repeat(lang_embed, "b l d -> b t l d", t=num_frames),
            attention_mask.repeat(num_frames, 1),
        )
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        pose_patches = self.pose_up(outputs["z_q"])
        video_pose_patches = torch.cat([pose_patches, video_patches], dim=2)

        video_recon = self.decoder(video_pose_patches, lang_embed.unsqueeze(1), attention_mask)
        video_recon = video_recon[:, :, self.num_pose_tokens : self.num_pose_tokens + video_patches.shape[2]]

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, [-1]],
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device


class ControllableDINOLatentPoseModel(nn.Module):
    """
    Stage-2 latent pose model with frozen base codebook and controllable pose tokens.
    """

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        latent_dim: int,
        num_latents: int,
        patch_size: int,
        enc_blocks: int,
        dec_blocks: int,
        num_heads: int,
        pose_dim: int = 7,
        num_pose_tokens: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.num_pose_tokens = num_pose_tokens

        self.dino_transform = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        )
        self.dino_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.dino_encoder.requires_grad_(False)
        dino_dim = 768

        self.pose_token_encoder = PoseTokenEncoder(
            pose_dim=pose_dim,
            dino_dim=dino_dim,
            num_pose_tokens=num_pose_tokens,
        )
        self.controllable_pose_token = nn.Parameter(
            torch.empty(1, 1, self.num_pose_tokens, dino_dim)
        )
        nn.init.uniform_(self.controllable_pose_token, a=-1, b=1)

        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )
        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.to_codebook_base = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=16,
            latent_dim=latent_dim,
            code_restart=True,
        )
        self.vq_pose = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        self.vq.requires_grad_(False)

        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.pose_up = nn.Linear(latent_dim, model_dim)
        self.pose_up_base = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

    def vq_encode(self, batch: Dict) -> Dict:
        videos = batch["videos"]
        current_tool_pose = batch["current_tool_pose"]
        target_pose = batch["target_pose"]
        pose_delta = batch["pose_delta"]

        bsz, num_frames = videos.shape[:2]
        videos = rearrange(videos, "b t c h w -> (b t) c h w")
        videos = self.dino_transform(videos)
        dino_features = self.dino_encoder.forward_features(videos)["x_norm_patchtokens"]
        dino_features = rearrange(dino_features, "(b t) l d -> b t l d", b=bsz, t=num_frames)

        base_pose_tokens = self.pose_token_encoder(current_tool_pose, target_pose, pose_delta)
        base_pose_tokens = base_pose_tokens.unsqueeze(1).expand(-1, num_frames, -1, -1)
        controllable_tokens = self.controllable_pose_token.expand(bsz, num_frames, -1, -1)
        padded_patches = torch.cat([controllable_tokens, base_pose_tokens, dino_features], dim=2)

        z = self.encoder(padded_patches)

        z_base = self.to_codebook_base(z[:, 1:, self.num_pose_tokens : self.num_pose_tokens * 2])
        z_base = z_base.reshape(bsz * (num_frames - 1), self.num_pose_tokens, self.latent_dim)
        z_q_base, z_base, emb_base, indices_base = self.vq(z_base)
        z_q_base = z_q_base.reshape(bsz, num_frames - 1, self.num_pose_tokens, self.latent_dim)

        z_pose = self.to_codebook(z[:, 1:, : self.num_pose_tokens])
        z_pose = z_pose.reshape(bsz * (num_frames - 1), self.num_pose_tokens, self.latent_dim)
        z_q, z, emb, indices = self.vq_pose(z_pose)
        z_q = z_q.reshape(bsz, num_frames - 1, self.num_pose_tokens, self.latent_dim)

        return {
            "patches": dino_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "z_q_uncontrol": z_q_base,
            "z_uncontrol": z_base,
            "emb_uncontrol": emb_base,
            "indices": indices,
            "indices_uncontrol": indices_base,
        }

    def forward(self, batch: Dict) -> Dict:
        outputs = self.vq_encode(batch)
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        video_pose_patches = torch.cat(
            [
                self.pose_up(outputs["z_q"]),
                self.pose_up_base(outputs["z_q_uncontrol"]),
                video_patches,
            ],
            dim=2,
        )
        video_recon = self.decoder(video_pose_patches)
        video_recon = video_recon[:, :, -video_patches.shape[2] :]

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, [-1]],
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
