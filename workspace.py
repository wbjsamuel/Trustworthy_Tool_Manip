import os
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import DictConfig, OmegaConf
import pathlib
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, random_split
from torch.optim.swa_utils import get_ema_avg_fn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch import LightningModule

from model.common.callbacks import ModelAveragingCallback, SaveConfigCallback

OmegaConf.register_new_resolver("eval", eval, replace=True)

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def resolve_resume_checkpoint(cfg: DictConfig, output_dir: pathlib.Path) -> str | None:
    resume_cfg = cfg.get("resume")
    if resume_cfg is None or not resume_cfg.get("enabled", False):
        return None

    checkpoint_name = resume_cfg.get("checkpoint_name", "last.ckpt")
    candidates: list[pathlib.Path] = []

    checkpoint_path = resume_cfg.get("checkpoint_path")
    if checkpoint_path:
        candidates.append(pathlib.Path(checkpoint_path))

    checkpoint_dir = resume_cfg.get("checkpoint_dir")
    if checkpoint_dir:
        candidates.append(pathlib.Path(checkpoint_dir))
    else:
        candidates.append(output_dir / "checkpoints")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
        if candidate.is_dir():
            matches = sorted(
                candidate.rglob(checkpoint_name),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if matches:
                return str(matches[0])

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Resume is enabled, but checkpoint '{checkpoint_name}' was not found under: {searched}"
    )


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent / "config"),
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    output_dir = pathlib.Path(HydraConfig.get().run.dir)
    # set seed
    seed = cfg.seed
    seed_everything(seed)
    # configure model
    model: LightningModule = hydra.utils.instantiate(cfg.policy)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*0.95), len(dataset) - int(len(dataset)*0.95)])
    train_dataloader = DataLoaderX(train_dataset, **cfg.dataloader.train)
    val_dataloader = DataLoaderX(val_dataset, **cfg.dataloader.val)

    model.set_normalizer(dataset.get_normalizer())
    resume_checkpoint = resolve_resume_checkpoint(cfg, output_dir)
    if resume_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        hydra.utils.instantiate(cfg.checkpoint, dirpath=output_dir / 'checkpoints'),
        ModelAveragingCallback(None, get_ema_avg_fn(0.9), cfg.ema.update_after_steps),
        SaveConfigCallback(OmegaConf.to_container(cfg, resolve=True))
    ]

    logger = WandbLogger(
        save_dir=output_dir,
        **cfg.logging,
    )

    trainer = Trainer(
        **cfg.trainer,
        strategy="ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto",
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
        ckpt_path=resume_checkpoint,
    )

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
