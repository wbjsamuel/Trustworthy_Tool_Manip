import time
from typing import Any, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoProcessor, AutoTokenizer, SiglipVisionModel, T5EncoderModel

from model.common.lr_scheduler import get_scheduler

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        dino_repo: str = "facebookresearch/dinov2",
        dino_model_name: str = "dinov2_vitb14_reg",
        siglip_model_name: str = "google/siglip-base-patch16-224",
        dino_feature_dim: int = 768,
        siglip_feature_dim: int = 768,
        freeze_backbones: bool = True,
    ) -> None:
        super().__init__()
        self.dino_feature_dim = dino_feature_dim
        self.siglip_feature_dim = siglip_feature_dim
        self.freeze_backbones = freeze_backbones

        self.dino_transform = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        self.dinov2 = torch.hub.load(dino_repo, dino_model_name, pretrained=True)
        self.siglip_processor = AutoProcessor.from_pretrained(siglip_model_name)
        self.siglip = SiglipVisionModel.from_pretrained(siglip_model_name)

        if freeze_backbones:
            self.dinov2.requires_grad_(False)
            self.siglip.requires_grad_(False)
            self.dinov2.eval()
            self.siglip.eval()

    def _encode_dino(self, image: torch.Tensor) -> torch.Tensor:
        dino_input = self.dino_transform(image)
        dino_features = self.dinov2.forward_features(dino_input)
        return dino_features["x_norm_patchtokens"].mean(dim=1)

    def _encode_siglip(self, image: torch.Tensor) -> torch.Tensor:
        # The processor handles model-specific normalization. We skip resize/rescale
        # because the dataset already emits 224x224 tensors in [0, 1].
        processor_inputs = self.siglip_processor(
            images=[img.detach().cpu() for img in image],
            return_tensors="pt",
            do_resize=False,
            do_rescale=False,
        )
        pixel_values = processor_inputs["pixel_values"].to(image.device)
        siglip_outputs = self.siglip(pixel_values=pixel_values)
        return siglip_outputs.pooler_output

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbones:
            with torch.no_grad():
                dino_features = self._encode_dino(image)
                siglip_features = self._encode_siglip(image)
        else:
            dino_features = self._encode_dino(image)
            siglip_features = self._encode_siglip(image)
        return torch.cat([dino_features, siglip_features], dim=1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbones:
            self.dinov2.eval()
            self.siglip.eval()
        return self


class LanguageEncoder(nn.Module):
    def __init__(
        self,
        text_model_name: str = "t5-base",
        freeze_backbone: bool = True,
        cache_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.model = T5EncoderModel.from_pretrained(text_model_name)
        self.freeze_backbone = freeze_backbone
        self.cache_embeddings = cache_embeddings and freeze_backbone
        self._embedding_cache: dict[str, torch.Tensor] = {}

        if freeze_backbone:
            self.model.requires_grad_(False)
            self.model.eval()

    def _encode_text_batch(self, texts: List[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.model(**encoded)
        else:
            outputs = self.model(**encoded)

        hidden_states = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked_hidden = hidden_states * attention_mask
        token_counts = attention_mask.sum(dim=1).clamp(min=1)
        return masked_hidden.sum(dim=1) / token_counts

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        normalized_texts = [text if text else "" for text in texts]
        if not self.cache_embeddings:
            return self._encode_text_batch(normalized_texts, device)

        missing_texts: List[str] = []
        for text in normalized_texts:
            if text not in self._embedding_cache:
                missing_texts.append(text)

        if missing_texts:
            unique_missing_texts = list(dict.fromkeys(missing_texts))
            missing_embeddings = self._encode_text_batch(unique_missing_texts, device)
            for text, embedding in zip(unique_missing_texts, missing_embeddings):
                self._embedding_cache[text] = embedding.detach().cpu()

        return torch.stack(
            [self._embedding_cache[text].to(device=device) for text in normalized_texts],
            dim=0,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
        return self


class Stage1Transformer(pl.LightningModule):
    def __init__(
        self,
        image_feature_dim: int = 1536,
        language_feature_dim: int = 768,
        pose_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 8,
        embed_dim: int = 2048,
        feedforward_mult: int = 8,
        num_fusion_tokens: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        dino_repo: str = "facebookresearch/dinov2",
        dino_model_name: str = "dinov2_vitb14_reg",
        siglip_model_name: str = "google/siglip-base-patch16-224",
        text_model_name: str = "t5-base",
        freeze_backbones: bool = True,
        cache_language_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._last_train_step_start_time: Optional[float] = None
        self._cumulative_estimated_flops: float = 0.0

        dino_feature_dim = image_feature_dim // 2
        siglip_feature_dim = image_feature_dim - dino_feature_dim

        self.image_encoder = ImageEncoder(
            dino_repo=dino_repo,
            dino_model_name=dino_model_name,
            siglip_model_name=siglip_model_name,
            dino_feature_dim=dino_feature_dim,
            siglip_feature_dim=siglip_feature_dim,
            freeze_backbones=freeze_backbones,
        )
        self.language_encoder = LanguageEncoder(
            text_model_name=text_model_name,
            freeze_backbone=freeze_backbones,
            cache_embeddings=cache_language_embeddings,
        )

        self.image_proj = nn.Linear(image_feature_dim, embed_dim)
        self.language_proj = nn.Linear(language_feature_dim, embed_dim)
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.modality_embeddings = nn.Parameter(torch.randn(3, embed_dim))
        self.fusion_tokens = nn.Parameter(torch.randn(1, num_fusion_tokens, embed_dim))

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * feedforward_mult,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, pose_dim),
        )
        self.criterion = nn.MSELoss()

    def _linear_flops(self, in_features: int, out_features: int, batch_tokens: int) -> int:
        return 2 * batch_tokens * in_features * out_features

    def _transformer_layer_flops(self, seq_len: int) -> int:
        embed_dim = int(self.hparams.embed_dim)
        ff_dim = embed_dim * int(self.hparams.feedforward_mult)

        attention_qkv = 3 * self._linear_flops(embed_dim, embed_dim, seq_len)
        attention_scores = 2 * int(self.hparams.num_heads) * seq_len * seq_len * (embed_dim // int(self.hparams.num_heads))
        attention_reduce = 2 * int(self.hparams.num_heads) * seq_len * seq_len * (embed_dim // int(self.hparams.num_heads))
        attention_out = self._linear_flops(embed_dim, embed_dim, seq_len)
        ffn = self._linear_flops(embed_dim, ff_dim, seq_len) + self._linear_flops(ff_dim, embed_dim, seq_len)
        return attention_qkv + attention_scores + attention_reduce + attention_out + ffn

    def estimate_forward_flops(self, batch_size: int) -> int:
        embed_dim = int(self.hparams.embed_dim)
        seq_len = int(self.hparams.num_fusion_tokens) + 3
        image_proj = self._linear_flops(int(self.hparams.image_feature_dim), embed_dim, batch_size)
        language_proj = self._linear_flops(int(self.hparams.language_feature_dim), embed_dim, batch_size)
        pose_proj = (
            self._linear_flops(int(self.hparams.pose_dim), embed_dim, batch_size)
            + self._linear_flops(embed_dim, embed_dim, batch_size)
        )
        transformer = batch_size * int(self.hparams.num_layers) * self._transformer_layer_flops(seq_len)
        output_head = (
            self._linear_flops(embed_dim, embed_dim, batch_size)
            + self._linear_flops(embed_dim, embed_dim, batch_size)
            + self._linear_flops(embed_dim, int(self.hparams.pose_dim), batch_size)
        )
        return image_proj + language_proj + pose_proj + transformer + output_head

    def floating_point_ops(self, batch: Optional[dict[str, Any]] = None) -> int:
        if batch is None:
            batch_size = 1
        elif isinstance(batch, dict) and "image" in batch:
            batch_size = int(batch["image"].shape[0])
        else:
            batch_size = 1
        return self.estimate_forward_flops(batch_size)

    def forward(
        self,
        image: torch.Tensor,
        current_tool_pose: torch.Tensor,
        instruction: List[str],
    ) -> torch.Tensor:
        image_embedding = self.image_proj(self.image_encoder(image))
        language_features = self.language_encoder(instruction, image.device)
        language_embedding = self.language_proj(language_features)
        pose_embedding = self.pose_proj(current_tool_pose)

        combined_input = torch.stack(
            [
                image_embedding + self.modality_embeddings[0],
                language_embedding + self.modality_embeddings[1],
                pose_embedding + self.modality_embeddings[2],
            ],
            dim=1,
        )
        fusion_tokens = self.fusion_tokens.expand(image.shape[0], -1, -1)
        combined_input = torch.cat([fusion_tokens, combined_input], dim=1)

        transformer_output = self.transformer(combined_input)
        fused_embedding = transformer_output[:, : self.hparams.num_fusion_tokens].mean(dim=1)
        return self.output_head(fused_embedding)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        predicted_pose = self(
            batch["image"], batch["current_tool_pose"], batch["instruction"]
        )
        loss = self.criterion(predicted_pose, batch["target_pose"])
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["image"].size(0),
        )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        self._last_train_step_start_time = time.perf_counter()
        loss = self._shared_step(batch, "train")

        batch_size = int(batch["image"].size(0))
        estimated_forward_flops = float(self.estimate_forward_flops(batch_size))
        estimated_train_step_flops = estimated_forward_flops * 3.0
        self._cumulative_estimated_flops += estimated_train_step_flops

        self.log(
            "train_estimated_fusion_flops",
            estimated_train_step_flops,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "train_cumulative_estimated_fusion_flops",
            self._cumulative_estimated_flops,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def on_train_batch_end(self, outputs, batch: dict, batch_idx: int) -> None:
        if self._last_train_step_start_time is None:
            return

        elapsed = max(time.perf_counter() - self._last_train_step_start_time, 1e-6)
        batch_size = int(batch["image"].size(0))
        self.log(
            "train_step_time",
            elapsed,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "train_samples_per_sec",
            batch_size / elapsed,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
        )

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler_name = self.hparams.lr_scheduler.lower()
        if scheduler_name == "constant":
            return optimizer

        if scheduler_name == "cosine":
            scheduler = get_scheduler(
                name="cosine",
                optimizer=optimizer,
                num_warmup_steps=self.hparams.lr_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        raise ValueError(
            f"Unsupported lr_scheduler '{self.hparams.lr_scheduler}'. "
            "Expected 'constant' or 'cosine'."
        )


if __name__ == "__main__":
    model = Stage1Transformer()
    image = torch.randn(2, 3, 224, 224)
    tool_pose = torch.randn(2, 16)
    instruction = ["pick up the tool", "place the block on the red square"]
    predicted_pose = model(image, tool_pose, instruction)
    print("Predicted Pose shape:", predicted_pose.shape)
