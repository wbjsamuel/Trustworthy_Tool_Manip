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

# FIX: Register the eval resolver directly. 
# We use use_existing_locally=True to prevent errors if it's already registered.
try:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
except Exception:
    pass 

class DeployPolicy:
    def __init__(self, ckpt_path: str):
        """
        Loads cfg from .hydra folder and handles 'eval' interpolations.
        """
        # 1. Path Resolution
        ckpt_path = os.path.abspath(ckpt_path)
        # Assuming path: .../RUN_ID/checkpoints/last.ckpt
        run_root = os.path.dirname(os.path.dirname(ckpt_path))
        hydra_cfg_path = os.path.join(run_root, '.hydra', 'config.yaml')
        
        if not os.path.exists(hydra_cfg_path):
            raise FileNotFoundError(f"Missing Hydra config at {hydra_cfg_path}")

        print(f"[*] Loading config: {hydra_cfg_path}")
        
        # 2. Load and Resolve
        # resolve=True ensures ${eval: ...} and other variables are computed now
        cfg = OmegaConf.load(hydra_cfg_path)
        OmegaConf.resolve(cfg) 

        # 3. Load Weights
        print(f"[*] Loading weights: {ckpt_path}")
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 4. Instantiate Model
        print("[*] Instantiating policy model via Hydra...")
        # This will now work because 'eval' is resolved
        model: LightningModule = hydra.utils.instantiate(cfg.policy)
        
        # 5. Load state dict
        model.load_state_dict(payload['state_dict'])
        self.policy = model.to(self.device)
        self.policy.eval()
        
        # 6. Parameters & History
        self.n_obs_steps = getattr(cfg, 'n_obs_steps', 2)
        print(f"[+] Model Loaded on {self.device}. History steps: {self.n_obs_steps}")
        
        self.obs_deque = deque(maxlen=self.n_obs_steps)
        self.action_deque = deque(maxlen=8)

    def get_model_input(self, observation, agent_pos, agent_num):
        head_cam_dict = {}
        agent_pos_list = []
        
        for agent_id in range(agent_num):
            camera_key = f'head_camera_agent{agent_id}'
            rgb = observation['sensor_data'][camera_key]['rgb']
            
            if torch.is_tensor(rgb):
                rgb = rgb.cpu().numpy()
            
            # Normalize to [0, 1] and HWC -> CHW
            head_cam = rgb.squeeze().astype(np.float32) / 255.0
            if head_cam.shape[-1] == 3: # If HWC
                head_cam = np.moveaxis(head_cam, -1, 0) 
                
            head_cam_dict[f'head_cam_{agent_id}'] = head_cam
            
            pos_i = agent_pos[agent_id * 8 : (agent_id + 1) * 8]
            agent_pos_list.append(pos_i)
            
        head_cam_dict['agent_pos'] = np.concatenate(agent_pos_list, axis=-1).astype(np.float32)
        return head_cam_dict

    def update_obs(self, obs: Dict[str, Any]):
        agent_num = len(obs['agent'])
        initial_qpos_list = []
        
        for i in range(agent_num):
            qpos = obs['agent'][f'panda-{i}']['qpos']
            if torch.is_tensor(qpos):
                qpos = qpos.cpu().numpy()
            qpos = qpos.flatten()
            
            # Use last gripper action if available, else default to 1.0 (open)
            gripper_val = 1.0 if len(self.action_deque) == 0 else self.action_deque[-1][i * 8 + 7]
            current_qpos = np.append(qpos[:7], gripper_val)
            initial_qpos_list.append(current_qpos)

        combined_qpos = np.concatenate(initial_qpos_list)
        formatted_obs = self.get_model_input(obs, combined_qpos, agent_num)
        self.obs_deque.append(formatted_obs)

    def get_action(self) -> np.ndarray:
        if len(self.obs_deque) < self.n_obs_steps:
            while len(self.obs_deque) < self.n_obs_steps:
                self.obs_deque.appendleft(self.obs_deque[0])

        batch_obs = {}
        for key in self.obs_deque[0].keys():
            batch_obs[key] = np.stack([x[key] for x in self.obs_deque])

        device_obs = dict_apply(batch_obs, lambda x: torch.from_numpy(x).to(self.device).unsqueeze(0))

        with torch.no_grad():
            result = self.policy.predict_action(device_obs)
            action_key = 'action_pred' if 'action_pred' in result else 'action'
            # DP2 usually returns [Batch, Horizon, Action_Dim]
            action_np = result[action_key].detach().cpu().numpy()[0, 0]
            
        self.action_deque.append(action_np)
        return action_np