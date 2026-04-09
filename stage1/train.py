from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from stage1.datamodule import Stage1DataModule
from stage1.model.stage1_transformer import Stage1Transformer

try:
    from pytorch_lightning.loggers import WandbLogger
except ImportError:  # pragma: no cover - optional dependency
    WandbLogger = None


def load_config() -> dict:
    with open("stage1/config/stage1.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_logger(config: dict):
    logging_config = config.get("logging", {})
    if logging_config.get("use_wandb", False) and WandbLogger is not None:
        logger = WandbLogger(
            project=logging_config["project"],
            name=logging_config["run_name"],
            save_dir=logging_config.get("save_dir", "logs"),
            mode=logging_config.get("mode", "online"),
        )
        logger.experiment.config.update(config, allow_val_change=True)
        return logger
    save_dir = logging_config.get("save_dir", "logs")
    return CSVLogger(save_dir=save_dir, name=logging_config.get("run_name", "stage1"))


def main() -> None:
    config = load_config()
    torch.set_float32_matmul_precision(config["training"].get("matmul_precision", "high"))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = config["training"].get("cudnn_benchmark", True)

    dm = Stage1DataModule(
        data_path=config["data"]["path"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 4),
        val_split=config["data"].get("val_split", 0.1),
        seed=config["training"].get("seed", 42),
        prefetch_factor=config["training"].get("prefetch_factor", 4),
        persistent_workers=config["training"].get("persistent_workers", True),
        prediction_target=config["data"].get("prediction_target", "next_pose"),
    )

    model = Stage1Transformer(
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        feedforward_mult=config["model"].get("feedforward_mult", 8),
        num_fusion_tokens=config["model"].get("num_fusion_tokens", 32),
        image_feature_dim=config["model"]["image_feature_dim"],
        language_feature_dim=config["model"]["language_feature_dim"],
        pose_dim=config["model"]["pose_dim"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
        dino_repo=config["model"].get("dino_repo", "facebookresearch/dinov2"),
        dino_model_name=config["model"].get("dino_model_name", "dinov2_vitb14_reg"),
        siglip_model_name=config["model"].get("siglip_model_name", "google/siglip-base-patch16-224"),
        text_model_name=config["model"].get("text_model_name", "t5-base"),
        freeze_backbones=config["model"].get("freeze_backbones", True),
    )

    checkpoint_dir = Path(config["training"].get("checkpoint_dir", "checkpoints/stage1"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="stage1-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_last=True,
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=build_logger(config),
        callbacks=callbacks,
        deterministic=config["training"].get("deterministic", True),
        log_every_n_steps=config["training"].get("log_every_n_steps", 10),
        precision=config["training"].get("precision", "bf16-mixed"),
        check_val_every_n_epoch=config["training"].get("check_val_every_n_epoch", 5),
    )

    trainer.fit(model, datamodule=dm)

    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.finish()


if __name__ == "__main__":
    main()
