from lightning.pytorch.cli import LightningCLI
from model.dataset import LightningStage1PoseDataModule
from model.model import DINO_LAM

cli = LightningCLI(
    DINO_LAM,
    LightningStage1PoseDataModule,
    seed_everything_default=42,
)
