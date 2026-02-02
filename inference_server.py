import socket
import pickle
import struct
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from lightning import LightningModule
import dill
import hydra

# Explicit imports from the repository structure
from policy.dp2_dino import DP2

# Register the eval resolver for OmegaConf as required by the config
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

def get_args():
    parser = argparse.ArgumentParser(description="DP2 Inference Server (RTX 5090)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--port", type=int, default=9999, help="Socket port")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    return parser.parse_args()

class DP2InferenceEngine:
    def __init__(self, ckpt_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ckpt_path = Path(ckpt_path)
        
        # 1. Path resolution for Hydra config
        config_path = ckpt_path.parent / ".hydra" / "config.yaml"
        if not config_path.exists():
            config_path = ckpt_path.parents[1] / ".hydra" / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Could not find config.yaml at {config_path}")

        self.cfg = OmegaConf.load(config_path)
        print(f"Loaded config from: {config_path}")

        # 2. Load Checkpoint using dill
        print(f"Loading checkpoint from {ckpt_path}...")
        with open(ckpt_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill, map_location=self.device)
        
        # 3. Instantiate Model via Hydra
        self.model: LightningModule = hydra.utils.instantiate(self.cfg.policy)
        
        # Load state dict and prepare for inference
        self.model.load_state_dict(payload['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.horizon = self.cfg.get('n_obs_steps', 1)
        print(f"Model initialized. Observation horizon: {self.horizon}")

        # --- 4. RTX 5090 Warmup Phase ---
        print("Starting GPU warmup (CUDA kernel pre-loading)...")
        self.warmup()
        print("Warmup complete. GPU is ready for low-latency inference.")

    @torch.no_grad()
    def warmup(self):
        """Runs a dummy inference pass to initialize CUDA kernels."""
        # Create dummy tensors matching expected shapes
        # Assuming 224x224 RGB and 7-DoF QPOS
        dummy_rgb = torch.randn(1, 1, 3, 224, 224).to(self.device)
        dummy_qpos = torch.randn(1, 1, 7).to(self.device)
        
        obs = {
            'rgb': dummy_rgb,
            'qpos': dummy_qpos
        }
        
        # Run three passes to ensure the scheduler and Dino backbones are fully cached
        for _ in range(3):
            _ = self.model.predict_action(obs)
        torch.cuda.synchronize()

    @torch.no_grad()
    def infer(self, img, arm_joints, gripper_state):
        # Image Preprocessing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # QPOS Construction
        qpos = np.concatenate([arm_joints, gripper_state], axis=-1)
        qpos_tensor = torch.from_numpy(qpos).float().unsqueeze(0).unsqueeze(0).to(self.device)

        obs = {
            'rgb': img_tensor,
            'qpos': qpos_tensor
        }

        # DP2 Inference
        action_dict = self.model.predict_action(obs)
        
        # Extract first action [B, T, D] -> [D]
        action = action_dict[0, 0, :].cpu().numpy().tolist()
        return action

def main():
    args = get_args()
    engine = DP2InferenceEngine(args.ckpt)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    
    print(f"RTX 5090 Server listening on {args.host}:{args.port}...")
    header_struct = struct.Struct("Q")

    while True:
        conn, addr = server.accept()
        print(f"Robot connected: {addr}")
        buffer = b""
        
        try:
            while True:
                # 1. Read size header
                while len(buffer) < header_struct.size:
                    chunk = conn.recv(16384)
                    if not chunk: raise ConnectionError
                    buffer += chunk
                
                packed_size = buffer[:header_struct.size]
                buffer = buffer[header_struct.size:]
                msg_size = header_struct.unpack(packed_size)[0]

                # 2. Read message body
                while len(buffer) < msg_size:
                    chunk = conn.recv(16384)
                    if not chunk: raise ConnectionError
                    buffer += chunk
                
                payload = buffer[:msg_size]
                buffer = buffer[msg_size:]
                
                # 3. Unpack and Infer
                data = pickle.loads(payload)
                img = cv2.imdecode(data['img'], cv2.IMREAD_COLOR)
                
                # Inference using separate joint and gripper lists from client
                action = engine.infer(img, data['arm_joints'], data['gripper_state'])
                
                # 4. Return result
                resp_payload = pickle.dumps(action)
                conn.sendall(header_struct.pack(len(resp_payload)) + resp_payload)
                
        except (ConnectionError, EOFError, socket.error):
            print(f"Robot {addr} disconnected.")
        finally:
            conn.close()

if __name__ == "__main__":
    main()