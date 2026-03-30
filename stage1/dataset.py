import torch
from torch.utils.data import Dataset
import os
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class Stage1Dataset(Dataset):
    """
    Dataset for Stage 1, loading data from a directory structure.
    """
    def __init__(self, root_dir="data/stage1_data/parsed_taco_data", transform=None):
        self.root_dir = root_dir
        self.samples = []
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

        for task_dir in sorted(os.listdir(self.root_dir)):
            task_path = os.path.join(self.root_dir, task_dir)
            if not os.path.isdir(task_path):
                continue
            for seq_dir in sorted(os.listdir(task_path)):
                seq_path = os.path.join(task_path, seq_dir)
                if not os.path.isdir(seq_path):
                    continue
                
                tool_poses_path = os.path.join(seq_path, 'tool_poses.pkl')
                if os.path.exists(tool_poses_path):
                    with open(tool_poses_path, 'rb') as f:
                        tool_poses = pickle.load(f)
                    num_frames = len(tool_poses)
                    for i in range(num_frames):
                        self.samples.append((seq_path, i))

    def __len__(self):
        return len(self.samples) - 1

    def __getitem__(self, idx):
        seq_path, frame_idx = self.samples[idx]
        
        # Load tool poses
        with open(os.path.join(seq_path, 'tool_poses.pkl'), 'rb') as f:
            tool_poses = pickle.load(f)

        # The last pose is the target pose.
        target_pose = torch.from_numpy(tool_poses[-1]).float()
        
        # The pose at the current frame_idx is the current tool pose.
        current_tool_pose = torch.from_numpy(tool_poses[frame_idx]).float()

        # Load the image for the current frame.
        image_path = os.path.join(seq_path, 'rgb', f'{frame_idx:06d}.png')
        if not os.path.exists(image_path):
            # This case should ideally not be hit if data is consistent
            return None

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Instruction is not available, so use a placeholder
        instruction = "No instruction available"

        return {
            "image": image,
            "current_tool_pose": current_tool_pose,
            "instruction": instruction,
            "target_pose": target_pose
        }

if __name__ == '__main__':
    # This assumes the data is in the expected directory structure.
    dataset = Stage1Dataset()
    if len(dataset) > 0:
        sample = dataset[0]
        if sample:
            print("Sample keys:", sample.keys())
            print("Image shape:", sample['image'].shape)
            print("Instruction:", sample['instruction'])
            print("Current tool pose shape:", sample['current_tool_pose'].shape)
            print("Target pose shape:", sample['target_pose'].shape)
            
        # Check another sample
        sample = dataset[3]
        if sample:
            print("\nSample 101 keys:", sample.keys())
            print("Image shape:", sample['image'].shape)
            print("Instruction:", sample['instruction'])
            print("Current tool pose shape:", sample['current_tool_pose'].shape)
            print("Target pose shape:", sample['target_pose'].shape)
    else:
        print("Dataset is empty. Check the data path and structure.")
