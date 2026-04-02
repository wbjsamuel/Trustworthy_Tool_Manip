from lightning.pytorch.cli import LightningCLI
from genie.dataset import LightningStage1PoseDataModule
from genie.model import DINO_LAM

cli = LightningCLI(
    DINO_LAM,
    LightningStage1PoseDataModule,
    seed_everything_default=42,
)
