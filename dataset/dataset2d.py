from typing import Union, Dict, Any
import cv2
import h5py
import random
import numpy as np
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset

from common.sampler import create_indices
from model.common.normalizer import LinearNormalizer
from common.normalize_util import get_image_range_normalizer


def make_transform(resize_size: tuple = (256, 256)):
    to_tensor = v2.ToImage()
    resize = v2.Resize(resize_size, antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.256, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def generate_skip_flags(episode_idx: np.ndarray, skip_interval: int) -> np.ndarray:
    skip_flags = np.zeros(episode_idx[-1], dtype=bool)
    for start in episode_idx[:-1]:
        for i in range(skip_interval):
            skip_flags[start+i] = True
    return skip_flags


class Dataset2D(Dataset):
    def __init__(self, 
                 dataset_path: str, 
                 horizon: int=1,
                 n_obs_steps: int=1,
                 pad_before: int=0,
                 pad_after: int=0,
                 input_meta: Union[Dict[str, Any], None]=None,
                 episode_mask: Union[np.ndarray, None]=None,
                 mask_head: bool=False,
                 use_mem: bool=False,
                 **kwargs
                 ) -> None:
        
        if use_mem:
            self.data = {}
            with h5py.File(dataset_path, 'r') as f:
                for key in f.keys():
                    if isinstance(f[key], h5py.Group):
                        for episode_key in f[key].keys():
                            self.data[f"{key}/{episode_key}"] = f[key][episode_key][:]
                    else:
                        self.data[key] = f[key][:]           
        else:
            self.data = h5py.File(dataset_path, "r")

        episode_idx = self.data["episode_idx"][:]
        episode_ends = self.data["episode_ends"][:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)
        self.indices = create_indices(episode_ends, 
                sequence_length=horizon, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        self.episode_idx = episode_idx
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.input_meta = input_meta
        self.mask_head = mask_head
        self.skip_flags = generate_skip_flags(np.concatenate(([0], episode_ends)), horizon)
        self.transform = make_transform((224, 384))
        self.noise_mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
        self.noise_std = np.array((0.229, 0.256, 0.225), dtype=np.float32)

    def __len__(self):
        return len(self.indices)
    
    def padding(self, data: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
        if start_idx > 0:
            data[:start_idx] = data[start_idx]
        if end_idx < self.horizon:
            data[end_idx:] = data[end_idx - 1]

        return data
    
    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.data['action'][:],
            'qpos': self.data['qpos'][:]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        for key in self.input_meta["obs"].keys():
            if key.startswith("cam"):
                normalizer[key] = get_image_range_normalizer()
                
        return normalizer
    
    def decode_per_frame_if_needed(self, v):
        """Return frames as torch.Tensor [T, C, H, W] or [T, H, W, C] depending on transform.
        Accepts either:
        - already-decoded: np.ndarray shape [T, H, W, C]  (usually uint8 or float32)
        - compressed:     np.ndarray shape [T, L]        (uint8, each row = encoded image bytes)
        """
        arr = np.asarray(v)

        if arr.ndim == 4:
            # Already decoded: [T, H, W, C]
            frames = []
            for i in range(arr.shape[0]):
                img_bgr = arr[i]                     # assume it's already RGB or we'll convert
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img = self.transform(img_rgb)
                frames.append(img)
            return torch.stack(frames, dim=0)

        elif arr.ndim == 2 and arr.dtype == np.uint8:
            # Compressed buffers: [T, L]
            frames = []
            for i in range(arr.shape[0]):
                buf = arr[i]
                if len(buf) == 0:
                    # You may want to skip / raise / use placeholder
                    raise ValueError(f"Empty buffer at frame {i}")

                img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise ValueError(f"Failed to decode frame {i} (corrupted or invalid image data)")

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img = self.transform(img_rgb)
                frames.append(img)

            return torch.stack(frames, dim=0)

        else:
            raise ValueError(
                f"Unsupported input shape/dtype: {arr.shape} {arr.dtype}\n"
                "Expected either [T, H, W, C] or [T, L] uint8"
            )
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        res = {'obs': {},}
        obs_keys = list(self.input_meta["obs"].keys())
        for key in obs_keys:
            start_idx = buffer_start_idx
            end_idx = buffer_start_idx + self.n_obs_steps - sample_start_idx
            if key.startswith("cam"):
                obs = []
                start_group_idx, start_frame_idx = self.episode_idx[start_idx]
                end_group_idx, end_frame_idx = self.episode_idx[end_idx]
                if start_group_idx == end_group_idx:
                    images = self.data[f"episode_{start_group_idx}/{key}"][start_frame_idx:end_frame_idx]
                else:
                    raise NotImplementedError("Cross-episode sampling is not implemented yet.")
                obs = self.decode_per_frame_if_needed(images).numpy()
                # obs = images
            else:
                obs = self.data[key][start_idx:end_idx]
                obs = np.array(obs).astype(np.float32)

            data = np.zeros((self.n_obs_steps, *obs.shape[1:]), dtype=np.float32)
            try:
                data[sample_start_idx:] = obs
            except ValueError:
                print(f"ValueError at index {idx} for key {key}: data shape {data.shape}, obs shape {obs.shape}, indices {buffer_start_idx}-{buffer_end_idx}, sample indices {sample_start_idx}-{sample_end_idx}")
                raise
            # data[sample_start_idx:] = obs
            data[:sample_start_idx] = obs[0]
            res['obs'][key] = data
            
        action = self.data['action'][buffer_start_idx:buffer_end_idx]
        action = np.array(action).astype(np.float32)
        data = np.zeros((self.horizon, *action.shape[1:]), dtype=np.float32)
        data[sample_start_idx:sample_end_idx] = action
        data = self.padding(data, sample_start_idx, sample_end_idx)
        res['action'] = data

        if self.mask_head:
            if not self.skip_flags[buffer_start_idx] and random.uniform(0, 1) < 0.8:
                origin_img = res['obs']['cam_head']
                noise = np.random.rand(*origin_img.shape).astype(np.float32)
                weight = random.uniform(0.0, 0.5)
                mean = self.noise_mean[None, :, None, None]
                std = self.noise_std[None, :, None, None]
                noise = (noise - mean) / std
                mixed = origin_img * (1.0 - weight) + noise * weight
                res['obs']['cam_head'] = mixed
        return res


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    args = parser.parse_args()
    dataset = Dataset2D(args.dataset_path, horizon=16, n_obs_steps=3,
                        input_meta={
                            "obs": {
                                "cam_left": [270, 480, 3],
                                "cam_right": [270, 480, 3],
                                "cam_head": [270, 480, 3],
                                "qpos": [14],
                            },
                            "action": [14],
                        },
                        mask_head=True,
                        use_mem=True)
    norms = dataset.get_normalizer()
    data = dataset[0]
    data = dataset[-1]
    for i in tqdm(range(0, len(dataset))):
        data = dataset[i]
        # print(data)
        #