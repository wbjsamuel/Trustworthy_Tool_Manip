import argparse
import os
from pathlib import Path
from tqdm import tqdm
import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d

def clip(qpos, velocity_threshold=0.001, sigma=1.0):
    """
    Clips static parts using joint velocity magnitude.
    Since qpos is now our source of truth for movement, 
    we calculate velocity directly from it.
    """
    # Calculate velocity of joint positions (ignoring gripper for clipping usually)
    velocity = np.diff(qpos[:, :6], axis=0)
    velocity_magnitude = np.linalg.norm(velocity, axis=1)
    
    smoothed_velocity = gaussian_filter1d(velocity_magnitude, sigma=sigma)
    is_moving = smoothed_velocity > velocity_threshold
    
    moving_indices = np.where(is_moving)[0]
    if len(moving_indices) == 0:
        return 0, len(qpos)
        
    start_idx = max(0, moving_indices[0] - 2)
    end_idx = min(len(qpos), moving_indices[-1] + 3)
    return start_idx, end_idx

class ROSDiffusionDataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.data_files = sorted(list(self.dataset_dir.rglob('*.hdf5')))
        if not self.data_files:
            raise FileNotFoundError(f"No HDF5 files found in {dataset_dir}")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        return self.extract_hdf5_data(idx)

    def extract_hdf5_data(self, idx):
        hdf5_file = self.data_files[idx]
        with h5py.File(hdf5_file, 'r') as f:
            root = f['data']
            obs = root['obs']
            
            # 1. Load raw components
            raw_joints = obs['joint_pose'][()]    # (N, 6)
            raw_gripper = obs['gripper_state'][()] # (N, 1)
            raw_rgb = obs['rgb'][()]               # (N, H, W, 3)
            
            # 2. Concatenate Gripper to Joint Poses to form 7D qpos
            # Final vector: [j1, j2, j3, j4, j5, j6, gripper]
            full_qpos = np.concatenate([raw_joints, raw_gripper], axis=-1)
            
            # 3. Define Action as the NEXT qpos
            # action[t] = qpos[t+1]
            # This makes qpos (0 to N-2) and action (1 to N-1)
            qpos_seq = full_qpos[:-1]
            action_seq = full_qpos[1:]
            rgb_seq = raw_rgb[:-1]
            
            # 4. Clipping based on the new qpos sequence
            start_idx, end_idx = clip(qpos_seq, velocity_threshold=0.0008)
            
        return {
            'obs': {
                'cam_front': rgb_seq[start_idx:end_idx],
                'qpos': qpos_seq[start_idx:end_idx]
            },
            'action': action_seq[start_idx:end_idx]
        }

def main(dataset_dir: str, output_path: str) -> None:
    dataset = ROSDiffusionDataset(dataset_dir)
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 4}
    
    episode_ends = []
    episode_idx = []
    total_steps = 0

    with h5py.File(output_path, "w") as f:
        initialized = False

        for i, data in tqdm(enumerate(dataset), desc="Processing", total=len(dataset)):
            obs = data["obs"]
            action = data["action"]
            rgb = obs["cam_front"]
            qpos = obs["qpos"]

            # Save Episode Group (for visualization/debugging)
            ep_group = f.create_group(f"episode_{i}")
            ep_group.create_dataset("cam_front", data=rgb, dtype="uint8", chunks=True, **comp_kwargs)
            ep_group.create_dataset("qpos", data=qpos, dtype="float32", **comp_kwargs)
            ep_group.create_dataset("action", data=action, dtype="float32", **comp_kwargs)

            # Flat dataset for training
            current_len = action.shape[0]
            if not initialized:
                f.create_dataset("qpos", data=qpos, shape=qpos.shape, maxshape=(None, 7), dtype="float32", **comp_kwargs)
                f.create_dataset("action", data=action, shape=action.shape, maxshape=(None, 7), dtype="float32", **comp_kwargs)
                initialized = True
            else:
                for key, val in [("qpos", qpos), ("action", action)]:
                    f[key].resize((f[key].shape[0] + val.shape[0]), axis=0)
                    f[key][-val.shape[0]:] = val

            total_steps += current_len
            episode_ends.append(total_steps)
            episode_idx += [(i, j) for j in range(current_len)]

        f.create_dataset("episode_ends", data=np.array(episode_ends), **comp_kwargs)
        f.create_dataset("episode_idx", data=np.array(episode_idx), **comp_kwargs)

    print(f"\n[DONE] Extraction complete.")
    print(f"Structure: qpos (7D), action (Next 7D qpos), episode_ends ({len(episode_ends)} episodes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.dataset_dir, args.output_path)