import torch
import yaml
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from stage1.model.stage1_transformer import Stage1Transformer
from stage1.datamodule import Stage1DataModule

def main():
    # Load configuration
    with open('stage1/config/stage1.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- Setup ---
    # DataModule
    dm = Stage1DataModule(
        data_path=config['data']['path'], 
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4)
    )

    # Model
    model = Stage1Transformer(
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        learning_rate=config['training']['learning_rate']
    )
    
    # Logger
    wandb_logger = WandbLogger(
        project=config['logging']['project'],
        name=config['logging']['run_name']
    )
    wandb_logger.watch(model)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        logger=wandb_logger,
        gpus=1 if torch.cuda.is_available() else 0
    )

    # --- Training ---
    trainer.fit(model, dm)


if __name__ == '__main__':
    # Setup dummy data and config for testing
    if not os.path.exists("stage1/config/stage1.yaml"):
        os.makedirs("stage1/config", exist_ok=True)
        dummy_config = {
            'data': {'path': 'data/stage1_data/parsed_taco_data'},
            'model': {'embed_dim': 512, 'num_heads': 8, 'num_layers': 6},
            'training': {'batch_size': 2, 'epochs': 2, 'learning_rate': 1e-4, 'num_workers': 0},
            'logging': {'project': 'trustworthy_tool_manip', 'run_name': 'stage1_test_run'}
        }
        with open("stage1/config/stage1.yaml", "w") as f:
            yaml.dump(dummy_config, f)
    
    # Create dummy data files if they don't exist
    import pickle
    import numpy as np
    from PIL import Image

    dummy_data_dir = "data/stage1_data/parsed_taco_data/dummy_task/seq_0"
    os.makedirs(os.path.join(dummy_data_dir, "rgb"), exist_ok=True)

    # Create a dummy image
    image_path = os.path.join(dummy_data_dir, "rgb", "000000.png")
    if not os.path.exists(image_path):
        Image.new('RGB', (224, 224)).save(image_path)
    
    # Create another dummy image for a different frame
    image_path_1 = os.path.join(dummy_data_dir, "rgb", "000001.png")
    if not os.path.exists(image_path_1):
        Image.new('RGB', (224, 224)).save(image_path_1)

    # Create dummy tool poses
    poses_path = os.path.join(dummy_data_dir, "tool_poses.pkl")
    if not os.path.exists(poses_path):
        tool_poses = np.random.rand(2, 7) # 2 frames, 7-d pose
        with open(poses_path, 'wb') as f:
            pickle.dump(tool_poses, f)
            
    main()
