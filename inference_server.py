import socket
import pickle
import struct
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from omegaconf import OmegaConf

# Explicit imports from the repository structure
from policy.dp2 import DP2

def get_args():
    parser = argparse.ArgumentParser(description="DP2 Inference Server (RTX 5090)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--port", type=int, default=9999, help="Socket port")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    return parser.parse_args()

class DP2InferenceEngine:
    def __init__(self, ckpt_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Config (Assumes config.yaml is in the same dir as the ckpt)
        ckpt_path = Path(ckpt_path)
        config_path = ckpt_path.parent / ".hydra" / "config.yaml"
        if not config_path.exists():
            config_path = ckpt_path.parents[1] / ".hydra" / "config.yaml"
        
        self.cfg = OmegaConf.load(config_path)
        print(f"Loaded config from: {config_path}")

        # 2. Initialize Model from dp2.py
        # DP2 in this repo usually takes (cfg, dataset_stats)
        # Here we load via Lightning's load_from_checkpoint
        self.model = DP2.load_from_checkpoint(
            str(ckpt_path), 
            cfg=self.cfg, 
            map_location=self.device
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Determine observation horizon (default to 1 if not in config)
        self.horizon = self.cfg.model.get('n_obs_steps', 1)
        print(f"Model initialized. Observation horizon: {self.horizon}")

    @torch.no_grad()
    def infer(self, img, joints):
        """
        DP2 expects:
        - rgb: [B, T, C, H, W]
        - joint_pos: [B, T, D]
        """
        # Image Preprocessing (BGR to RGB and Resize to 224x224)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        
        # Add Batch and Time dimensions: [1, 1, 3, 224, 224]
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Joints Preprocessing: [1, 1, 7]
        joint_tensor = torch.from_numpy(np.array(joints)).float()
        joint_tensor = joint_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        obs = {
            'rgb': img_tensor,
            'joint_pos': joint_tensor
        }

        # The 'predict_action' or forward call in DP2
        # Note: Depending on your exact version, you might use self.model(obs) 
        # or a specific inference method provided in the class.
        action_dict = self.model(obs)
        
        # DP2 returns a sequence. We take the first action [B, T, D] -> [D]
        # Adjust index based on whether your model outputs normalized or raw actions
        action = action_dict[0, 0, :].cpu().numpy().tolist()
        return action

def main():
    args = get_args()
    engine = DP2InferenceEngine(args.ckpt)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    
    print(f"RTX 5090 Server listening on {args.port}...")

    header_struct = struct.Struct("Q") # 8-byte unsigned long long

    while True:
        conn, addr = server.accept()
        print(f"Client connected: {addr}")
        buffer = b""
        
        try:
            while True:
                # Read header
                while len(buffer) < header_struct.size:
                    chunk = conn.recv(8192)
                    if not chunk: raise ConnectionError
                    buffer += chunk
                
                packed_size = buffer[:header_struct.size]
                buffer = buffer[header_struct.size:]
                msg_size = header_struct.unpack(packed_size)[0]

                # Read payload
                while len(buffer) < msg_size:
                    chunk = conn.recv(8192)
                    if not chunk: raise ConnectionError
                    buffer += chunk
                
                payload = buffer[:msg_size]
                buffer = buffer[msg_size:]
                
                data = pickle.loads(payload)
                img = cv2.imdecode(data['img'], cv2.IMREAD_COLOR)
                
                # Perform DP2 Inference
                action = engine.infer(img, data['joints'])
                
                # Return result
                resp_payload = pickle.dumps(action)
                conn.sendall(header_struct.pack(len(resp_payload)) + resp_payload)
                
        except (ConnectionError, EOFError):
            print("Client disconnected.")
        finally:
            conn.close()

if __name__ == "__main__":
    main()