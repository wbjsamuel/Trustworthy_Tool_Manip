import pytorch_lightning as pl
from torch.utils.data import DataLoader
from stage1.dataset import Stage1Dataset

class Stage1DataModule(pl.LightningDataModule):
    def __init__(self, data_path: str = "data/stage1_data/parsed_taco_data.pkl", batch_size: int = 32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = Stage1Dataset(self.data_path)
            # You might want to split your data into train and validation sets
            # For now, we use the same dataset for both.
            self.val_dataset = Stage1Dataset(self.data_path)

        if stage == 'test' or stage is None:
            self.test_dataset = Stage1Dataset(self.data_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
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
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(dummy_data, f)


    dm = Stage1DataModule(data_path=pickle_path)
    dm.setup()
    for batch in dm.train_dataloader():
        print("Batch keys:", batch.keys())
        print("Image batch shape:", batch['image'].shape)
        break
