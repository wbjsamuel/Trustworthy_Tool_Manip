import argparse
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Optional
from tqdm import tqdm
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
        return extract_hdf5_data(self.data_files[idx])


def extract_hdf5_data(hdf5_file):
    hdf5_file = Path(hdf5_file)
    with h5py.File(hdf5_file, 'r') as f:
        # root = f['data']
        obs = f['observations']

        # 1. Load raw components
        raw_joints = obs['qpos'][()]    # (N, 6)
        # raw_gripper = obs['gripper_state'][()] # (N, 1)
        raw_rgb = obs['rgb'][()]               # (N, H, W, 3)

        # 2. Concatenate Gripper to Joint Poses if needed.
        # Final vector would be: [j1, j2, j3, j4, j5, j6, gripper]
        # full_qpos = np.concatenate([raw_joints, raw_gripper], axis=-1)
        full_qpos = raw_joints

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


def _extract_episode(task):
    idx, hdf5_file = task
    return idx, extract_hdf5_data(hdf5_file)


def iter_extracted_episodes(data_files, num_workers):
    tasks = list(enumerate(data_files))
    if num_workers <= 1:
        for task in tqdm(tasks, desc="Processing", total=len(tasks)):
            yield _extract_episode(task)
        return

    next_to_submit = 0
    next_to_yield = 0
    pending = set()
    completed = {}
    max_pending = min(len(tasks), num_workers * 2)

    try:
        executor = ProcessPoolExecutor(max_workers=num_workers)
    except OSError as exc:
        print(f"[WARN] Multiprocessing unavailable ({exc}); falling back to single-process extraction.")
        for task in tqdm(tasks, desc="Processing", total=len(tasks)):
            yield _extract_episode(task)
        return

    with executor:
        with tqdm(total=len(tasks), desc=f"Processing ({num_workers} workers)") as pbar:
            while next_to_submit < len(tasks) and len(pending) < max_pending:
                pending.add(executor.submit(_extract_episode, tasks[next_to_submit]))
                next_to_submit += 1

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    idx, data = future.result()
                    completed[idx] = data
                    pbar.update(1)

                while next_to_submit < len(tasks) and len(pending) < max_pending:
                    pending.add(executor.submit(_extract_episode, tasks[next_to_submit]))
                    next_to_submit += 1

                while next_to_yield in completed:
                    yield next_to_yield, completed.pop(next_to_yield)
                    next_to_yield += 1


def main(dataset_dir: str, output_path: str, num_workers: Optional[int] = None) -> None:
    dataset = ROSDiffusionDataset(dataset_dir)
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8, len(dataset))
    num_workers = max(1, min(num_workers, len(dataset)))

    comp_kwargs = {'compression': 'gzip', 'compression_opts': 4}
    
    episode_ends = []
    episode_idx = []
    total_steps = 0
    qpos_dim = None

    with h5py.File(output_path, "w") as f:
        initialized = False

        for i, data in iter_extracted_episodes(dataset.data_files, num_workers):
            obs = data["obs"]
            action = data["action"]
            rgb = obs["cam_front"]
            qpos = obs["qpos"]
            qpos_dim = qpos.shape[1]

            # Save Episode Group (for visualization/debugging)
            ep_group = f.create_group(f"episode_{i}")
            ep_group.create_dataset("cam_front", data=rgb, dtype="uint8", chunks=True, **comp_kwargs)
            ep_group.create_dataset("qpos", data=qpos, dtype="float32", **comp_kwargs)
            ep_group.create_dataset("action", data=action, dtype="float32", **comp_kwargs)

            # Flat dataset for training
            current_len = action.shape[0]
            if not initialized:
                f.create_dataset("qpos", data=qpos, shape=qpos.shape, maxshape=(None, *qpos.shape[1:]), dtype="float32", **comp_kwargs)
                f.create_dataset("action", data=action, shape=action.shape, maxshape=(None, *action.shape[1:]), dtype="float32", **comp_kwargs)
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
    print(f"Structure: qpos/action ({qpos_dim}D), episode_ends ({len(episode_ends)} episodes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes for per-episode extraction. Defaults to min(cpu_count, 8, num_episodes).",
    )
    args = parser.parse_args()
    main(args.dataset_dir, args.output_path, args.num_workers)
