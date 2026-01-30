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

# Register the eval resolver so OmegaConf knows how to handle ${eval:...}
from omegaconf import INTERPOLATION_RESOLVERS
if 'eval' not in INTERPOLATION_RESOLVERS:
    OmegaConf.register_new_resolver("eval", eval)

class DeployPolicy:
    def __init__(self, ckpt_path: str):
        """
        Loads cfg from .hydra folder and handles 'eval' interpolations.
        """
        # 1. Resolve Path to .hydra/config.yaml
        ckpt_path = os.path.abspath(ckpt_path)
        ckpt_dir = os.path.dirname(ckpt_path)
        run_root = os.path.dirname(ckpt_dir)
        hydra_cfg_path = os.path.join(run_root, '.hydra', 'config.yaml')
        
        if not os.path.exists(hydra_cfg_path):
            # Fallback for different folder depths
            run_root = os.path.dirname(run_root)
            hydra_cfg_path = os.path.join(run_root, '.hydra', 'config.yaml')
            if not os.path.exists(hydra_cfg_path):
                raise FileNotFoundError(f"Could not find .hydra/config.yaml relative to {ckpt_path}")

        print(f"[*] Loading config from: {hydra_cfg_path}")
        
        # 2. Load and Resolve Config
        # We use resolve=True to process all ${...} interpolations immediately
        cfg = OmegaConf.load(hydra_cfg_path)
        OmegaConf.resolve(cfg) 

        # 3. Load Checkpoint Payload
        print(f"[*] Loading weights from: {ckpt_path}")
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 4. Instantiate Model
        print("[*] Instantiating policy model...")
        # DP2-DINO models often rely on the hydra.utils.instantiate logic
        model: LightningModule = hydra.utils.instantiate(cfg.policy)
        
        # 5. Load State Dict
        # Lightning usually stores weights under 'state_dict'
        model.load_state_dict(payload['state_dict'])
        self.policy = model.to(self.device)
        self.policy.eval()
        
        # 6. Setup History Buffer
        # Use config for obs steps, default to 2 if not explicitly set
        self.n_obs_steps = getattr(cfg, 'n_obs_steps', 2)
        print(f"[+] Model Ready. Device: {self.device} | History steps: {self.n_obs_steps}")
        
        self.obs_deque = deque(maxlen=self.n_obs_steps)
        self.action_deque = deque(maxlen=8)

    # ... keep get_model_input, update_obs, and get_action from previous versions ...