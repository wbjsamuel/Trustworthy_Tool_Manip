import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from stage1.dataset import Stage1Dataset

class Stage1DataModule(pl.LightningDataModule):
    def __init__(self, data_path: str = "data/stage1_data/parsed_taco_data", batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = Stage1Dataset(root_dir=self.data_path)
            # You might want to split your data into train and validation sets
            # For now, we use the same dataset for both.
            self.val_dataset = Stage1Dataset(root_dir=self.data_path)

        if stage == 'test' or stage is None:
            self.test_dataset = Stage1Dataset(root_dir=self.data_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        
        images = torch.stack([item['image'] for item in batch])
        instructions = [item['instruction'] for item in batch]
        current_tool_poses = torch.stack([item['current_tool_pose'] for item in batch])
        target_poses = torch.stack([item['target_pose'] for item in batch])

        return {
            'image': images,
            'instruction': instructions,
            'current_tool_pose': current_tool_poses,
            'target_pose': target_poses
        }

if __name__ == '__main__':
    # Create dummy data if it doesn't exist
    import os
    import pickle
    import numpy as np
    from PIL import Image

    dummy_data_dir = "data/stage1_data/parsed_taco_data/dummy_task/seq_0"
    os.makedirs(os.path.join(dummy_data_dir, "rgb"), exist_ok=True)

    # Create a dummy image
    image_path = os.path.join(dummy_data_dir, "rgb", "000000.png")
    Image.new('RGB', (224, 224)).save(image_path)
    
    # Create another dummy image for a different frame
    image_path_1 = os.path.join(dummy_data_dir, "rgb", "000001.png")
    Image.new('RGB', (224, 224)).save(image_path_1)

    # Create dummy tool poses
    tool_poses = np.random.rand(2, 7) # 2 frames, 7-d pose
    poses_path = os.path.join(dummy_data_dir, "tool_poses.pkl")
    with open(poses_path, 'wb') as f:
        pickle.dump(tool_poses, f)

    dm = Stage1DataModule(data_path="data/stage1_data/parsed_taco_data")
    dm.setup()
    train_loader = dm.train_dataloader()
    
    print(f"Dataset has {len(dm.train_dataset)} samples.")

    for batch in train_loader:
        if batch:
            print("Batch keys:", batch.keys())
            print("Image batch shape:", batch['image'].shape)
            print("Current tool pose batch shape:", batch['current_tool_pose'].shape)
            print("Target pose batch shape:", batch['target_pose'].shape)
            break
        else:
            print("Received an empty batch.")
