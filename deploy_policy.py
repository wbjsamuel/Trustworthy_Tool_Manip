import os
import torch
import dill
import hydra
import numpy as np
from collections import deque
from typing import Any, Dict
from omegaconf import OmegaConf
from lightning import LightningModule
from common.pytorch_util import dict_apply

class DeployPolicy:
    def __init__(self, ckpt_path: str):
        """
        Loads cfg from the .hydra folder located in the checkpoint's parent directory.
        """
        # 1. Resolve Paths
        # Expected: outputs/DP2-DINO/RUN_ID/checkpoints/last.ckpt
        # .hydra is at: outputs/DP2-DINO/RUN_ID/.hydra/
        ckpt_dir = os.path.dirname(ckpt_path)
        run_root = os.path.dirname(ckpt_dir)
        hydra_cfg_path = os.path.join(run_root, '.hydra', 'config.yaml')
        
        if not os.path.exists(hydra_cfg_path):
            raise FileNotFoundError(f"Could not find Hydra config at {hydra_cfg_path}")

        print(f"[*] Loading config from: {hydra_cfg_path}")
        cfg = OmegaConf.load(hydra_cfg_path)

        # 2. Load Checkpoint Payload
        print(f"[*] Loading weights from: {ckpt_path}")
        # Using dill because Diffusion Policy often pickles complex objects
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 3. Instantiate Model via Hydra
        # hydra.utils.instantiate uses the '_target_' field in your config.yaml
        print("[*] Instantiating policy model...")
        model: LightningModule = hydra.utils.instantiate(cfg.policy)
        
        # 4. Load State Dict
        # Lightning checkpoints usually wrap the model in 'state_dict'
        model.load_state_dict(payload['state_dict'])
        self.policy = model.to(self.device)
        self.policy.eval()
        
        # 5. Set Parameters from Config
        self.n_obs_steps = cfg.n_obs_steps if hasattr(cfg, 'n_obs_steps') else 2
        print(f"[+] Model Ready. Device: {self.device} | History: {self.n_obs_steps}")
        
        # Buffers
        self.obs_deque = deque(maxlen=self.n_obs_steps)
        self.action_deque = deque(maxlen=8)

    def get_model_input(self, observation, agent_pos, agent_num):
        """Processes raw socket data into model-ready tensors."""
        head_cam_dict = {}
        agent_pos_list = []
        
        for agent_id in range(agent_num):
            # RGB normalization and axis swap (HWC -> CHW)
            camera_key = f'head_camera_agent{agent_id}'
            rgb = observation['sensor_data'][camera_key]['rgb']
            
            if torch.is_tensor(rgb):
                rgb = rgb.cpu().numpy()
            
            # Normalize to [0, 1] as expected by DINO/ViT backbones
            head_cam = rgb.squeeze().astype(np.float32) / 255.0
            head_cam = np.moveaxis(head_cam, -1, 0) 
            head_cam_dict[f'head_cam_{agent_id}'] = head_cam
            
            # Agent state (Joints + Gripper)
            pos_i = agent_pos[agent_id * 8 : (agent_id + 1) * 8]
            agent_pos_list.append(pos_i)
            
        head_cam_dict['agent_pos'] = np.concatenate(agent_pos_list, axis=-1).astype(np.float32)
        return head_cam_dict

    def update_obs(self, obs: Dict[str, Any]):
        """Updates the history buffer with new data from the ROS client."""
        agent_num = len(obs['agent'])
        initial_qpos_list = []
        
        for i in range(agent_num):
            qpos = obs['agent'][f'panda-{i}']['qpos']
            if torch.is_tensor(qpos):
                qpos = qpos.cpu().numpy()
            qpos = qpos.flatten()
            
            # If no actions taken yet, assume gripper is open (1.0)
            # otherwise use the last commanded gripper state
            gripper_state = 1.0 if len(self.action_deque) == 0 else self.action_deque[-1][i * 8 + 7]
            current_qpos = np.append(qpos[:7], gripper_state)
            initial_qpos_list.append(current_qpos)

        combined_qpos = np.concatenate(initial_qpos_list)
        formatted_obs = self.get_model_input(obs, combined_qpos, agent_num)
        self.obs_deque.append(formatted_obs)

    def get_action(self) -> np.ndarray:
        """Runs inference on the accumulated history."""
        # Ensure we have enough history to run the model
        if len(self.obs_deque) < self.n_obs_steps:
            while len(self.obs_deque) < self.n_obs_steps:
                self.obs_deque.appendleft(self.obs_deque[0])

        # Batch and move to GPU
        batch_obs = {}
        for key in self.obs_deque[0].keys():
            batch_obs[key] = np.stack([x[key] for x in self.obs_deque])

        device_obs = dict_apply(batch_obs, lambda x: torch.from_numpy(x).to(self.device).unsqueeze(0))

        with torch.no_grad():
            result = self.policy.predict_action(device_obs)
            # Handle different return keys based on DP2 version
            action_key = 'action_pred' if 'action_pred' in result else 'action'
            action_np = result[action_key].detach().cpu().numpy()[0, 0]
            
        self.action_deque.append(action_np)
        return action_np