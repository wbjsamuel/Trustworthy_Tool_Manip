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
        batch_size=config['training']['batch_size']
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
            'data': {'path': 'data/stage1_data/parsed_taco_data.pkl'},
            'model': {'embed_dim': 512, 'num_heads': 8, 'num_layers': 6},
            'training': {'batch_size': 2, 'epochs': 2, 'learning_rate': 1e-4},
            'logging': {'project': 'trustworthy_tool_manip', 'run_name': 'stage1_test_run'}
        }
        with open("stage1/config/stage1.yaml", "w") as f:
            yaml.dump(dummy_config, f)
    
    # Create dummy pickle file if it doesn't exist
    import pickle
    import numpy as np
    from PIL import Image

    pickle_path = "data/stage1_data/parsed_taco_data.pkl"
    if not os.path.exists(pickle_path):
        dummy_data = []
        data_dir = "data/stage1_data/parsed_taco_data/episode_001"
        os.makedirs(data_dir, exist_ok=True)
        image_path = os.path.join(data_dir, "image_000.png")
        Image.new('RGB', (224, 224)).save(image_path)

        episode_data = {
            'images': [image_path],
            'tool_poses': np.random.rand(5, 7),
            'instruction': 'pick up the tool',
            'target_pose': np.random.rand(7)
        }
        dummy_data.append(episode_data)
        
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        with open(pickle_path, 'wb') as f:
            pickle.dump(dummy_data, f)
            
    main()
