import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


def make_transform(resize_size: tuple = (256, 256)):
    to_tensor = v2.ToImage()
    resize = v2.Resize(resize_size, antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.256, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def clip(seq, velocity_threshold=0.001, sigma=1.0):
    """
    Clip the still part of the sequence from both ends.
    
    Args:
        seq: Action sequence representing joint angles
        velocity_threshold: Threshold for determining if the robot is moving
        sigma: Standard deviation for Gaussian smoothing filter
        min_motion_duration: Minimum number of frames for valid motion segment
    
    Returns:
        Clipped sequence and the start/end indices
    """
    
    # Calculate velocity (difference between consecutive frames)
    velocity = np.diff(seq, axis=0)
    
    # Calculate the magnitude of velocity (L2 norm across joint dimensions)
    velocity_magnitude = np.linalg.norm(velocity, axis=1)  # Shape: (T-1,)
    
    # Apply Gaussian smoothing to make the detection robust to noise
    smoothed_velocity = gaussian_filter1d(velocity_magnitude, sigma=sigma)
    
    # Find motion segments (where smoothed velocity exceeds threshold)
    is_moving = smoothed_velocity > velocity_threshold
    
    # Find the first and last indices where the robot is moving
    moving_indices = np.where(is_moving)[0]
    
    start_idx = moving_indices[0]
    end_idx = moving_indices[-1] + 1  # +1 because velocity has one less frame
    
    # Add small buffer to avoid cutting too aggressively
    buffer = 3  # 5% of sequence length as buffer
    start_idx = max(0, start_idx - buffer)
    end_idx = min(len(seq), end_idx + buffer)
    
    return start_idx, end_idx


class RealTrajectoryDataset(Dataset):
    def __init__(self, dataset_dir: str, resize_size: tuple = (256, 256)):
        self.dataset_dir = Path(dataset_dir)
        self.data_files = list(self.dataset_dir.glob('*.hdf5'))
        self.data_files.sort()
        self.resize_size = resize_size
        self.transform = make_transform(resize_size)

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        return self.extract_hdf5_data(idx)
    

    def decode_per_frame_if_needed(self, v):
        """Return frames as np.ndarray [T, H, W, C].
        Inputs either:
        - already-decoded array of shape [T, H, W, C]
        - compressed buffer array of shape [T, L] uint8 (each row is one image)
        """

        arr = np.asarray(v)
        # Already decoded
        if arr.ndim == 4:
            res = np.empty((arr.shape[0], 3, self.resize_size[0], self.resize_size[1]), dtype=np.float32)
            for i in range(arr.shape[0]):
                res[i] = self.transform(arr[i])
            return arr

        # Compressed buffers: decode each row separately
        if arr.ndim == 2 and arr.dtype == np.uint8:
            frames = []
            for i in range(arr.shape[0]):
                buf = arr[i]
                img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img = self.transform(img_rgb).numpy()

                # plt.figure(figsize=(8, 8))
                # plt.subplot(3, 1, 1); plt.imshow(img_bgr); plt.title("Assume BGR→RGB swap"); plt.axis("off")
                # plt.subplot(3, 1, 2); plt.imshow(img_rgb); plt.title("After cvtColor"); plt.axis("off")
                # plt.subplot(3, 1, 3); plt.imshow(np.transpose(img, (1, 2, 0))); plt.title("After transform"); plt.axis("off")
                # plt.savefig("debug_image_decoding.png")
                
                frames.append(img)
            return np.stack(frames, axis=0)

        raise ValueError(f"Unsupported video tensor shape: {arr.shape}, dtype: {arr.dtype}")


    def extract_hdf5_data(self, idx):
        """
        /action:        (525, 14)       float32
        /action_base:   (525, 6)        float32
        /action_eef:    (525, 14)       float32
        /action_velocity:       (525, 4)        float32
        /observations
        /observations/base_velocity:    (525, 4)        float32
        /observations/eef:      (525, 14)       float32
        /observations/effort:   (525, 14)       float32
        /observations/images
        /observations/images/head:      (525, 14354)    uint8
        /observations/images/left_wrist:        (525, 14354)    uint8
        /observations/images/right_wrist:       (525, 14354)    uint8
        /observations/qpos:     (525, 14)       float32
        /observations/qvel:     (525, 14)       float32
        /observations/robot_base:       (525, 6)        float32
        """
        hdf5_file = self.data_files[idx]
        with h5py.File(hdf5_file, 'r') as f:
            qpos = f['/observations/qpos'][()]
            action = f['/action'][()]
            start_idx, end_idx = clip(action, velocity_threshold=0.01)
            print(f"Clipping episode {hdf5_file.name}: {start_idx} to {end_idx} / {len(action)}")
            image_dict = dict()
            for cam_name in f[f'/observations/images/'].keys():
                image_dict[cam_name] = f[f'/observations/images/{cam_name}'][()]

        for cam_name in image_dict.keys():
            image_dict[cam_name] = self.decode_per_frame_if_needed(image_dict[cam_name])

        return {
            'obs': {
                'cam_right': image_dict['right_wrist'],
                'cam_left': image_dict['left_wrist'],
                'cam_head': image_dict['head'],
                'qpos': qpos
            },
            'action': action
        }


def main(dataset_dir: str, output_path: str, resize_size: tuple) -> None:
    dataset = RealTrajectoryDataset(dataset_dir, resize_size)
    device = 'cuda:0'

    comp_kwaegs = {'compression': 'gzip', 'compression_opts': 4}
    episode_ends = []
    end = 0
    with h5py.File(output_path, "w") as f:
        f.attrs['resize'] = True
        for i, data in tqdm(enumerate(dataset), desc="Loading data", total=len(dataset)):
            obs = data["obs"]
            action = data["action"]
            if data is not None:
                cam_right = obs["cam_right"]
                cam_left = obs["cam_left"]
                cam_head = obs["cam_head"]
                qpos = obs["qpos"]

                ntime, dt = np.linspace(0.0, 1.0, num=action.shape[0], endpoint=True, retstep=True)
                ntime = ntime.astype(np.float32)
                dt = np.full((action.shape[0],), dt, dtype=np.float32)

                end += len(action)
                episode_ends.append(end)

                if i == 0:
                    f.create_dataset(
                        f"cam_head",
                        data=cam_head,
                        shape=cam_head.shape,
                        maxshape=(None, *cam_head.shape[1:]),
                        dtype="float32",
                        **comp_kwaegs
                    )
                    f.create_dataset(
                        f"cam_left",
                        data=cam_left,
                        shape=cam_left.shape,
                        maxshape=(None, *cam_left.shape[1:]),
                        dtype="float32",
                        **comp_kwaegs
                    )
                    f.create_dataset(
                        f"cam_right",
                        data=cam_right,
                        shape=cam_right.shape,
                        maxshape=(None, *cam_right.shape[1:]),
                        dtype="float32",
                        **comp_kwaegs
                    )
                    f.create_dataset(
                        f"qpos",
                        data=qpos,
                        shape=qpos.shape,
                        maxshape=(None, *qpos.shape[1:]),
                        dtype="float32",
                        **comp_kwaegs
                    )
                    f.create_dataset(
                        f"action",
                        data=action,
                        shape=action.shape,
                        maxshape=(None, *action.shape[1:]),
                        dtype="float32",
                        **comp_kwaegs
                    )
                    f.create_dataset(
                        f"ntime",
                        data=ntime,
                        shape=ntime.shape,
                        maxshape=(None,),
                        dtype="float32",
                        **comp_kwaegs
                    )
                    f.create_dataset(
                        f"dt",
                        data=dt,
                        shape=dt.shape,
                        maxshape=(None,),
                        dtype="float32",
                        **comp_kwaegs
                    )
                else:
                    f["cam_right"].resize((f["cam_right"].shape[0] + cam_right.shape[0]), axis=0)
                    f["cam_right"][-cam_right.shape[0]:] = cam_right
                    f["cam_left"].resize((f["cam_left"].shape[0] + cam_left.shape[0]), axis=0)
                    f["cam_left"][-cam_left.shape[0]:] = cam_left
                    f["cam_head"].resize((f["cam_head"].shape[0] + cam_head.shape[0]), axis=0)
                    f["cam_head"][-cam_head.shape[0]:] = cam_head
                    f["qpos"].resize((f["qpos"].shape[0] + qpos.shape[0]), axis=0)
                    f["qpos"][-qpos.shape[0]:] = qpos
                    f["action"].resize((f["action"].shape[0] + action.shape[0]), axis=0)
                    f["action"][-action.shape[0]:] = action
                    f["ntime"].resize((f["ntime"].shape[0] + ntime.shape[0]), axis=0)
                    f["ntime"][-ntime.shape[0]:] = ntime
                    f["dt"].resize((f["dt"].shape[0] + dt.shape[0]), axis=0)
                    f["dt"][-dt.shape[0]:] = dt
        f.create_dataset(
            "episode_ends",
            data=np.array(episode_ends),
            **comp_kwaegs
        )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output")
    parser.add_argument("--resize_size", type=int, nargs=2, default=(224, 384), help="Resize size for images")
    args = parser.parse_args()

    main(args.dataset_dir, args.output_path, args.resize_size)
