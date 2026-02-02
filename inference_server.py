import socket
import pickle
import struct
import torch
import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from omegaconf import OmegaConf
from lightning import LightningModule
import dill
import hydra

# Explicit imports from the repository structure
from policy.dp2_dino import DP2

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
        
        config_path = ckpt_path.parent / ".hydra" / "config.yaml"
        if not config_path.exists():
            config_path = ckpt_path.parents[1] / ".hydra" / "config.yaml"
        
        self.cfg = OmegaConf.load(config_path)
        print(f"Loaded config from: {config_path}")

        with open(ckpt_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill, map_location=self.device)
        
        self.model: LightningModule = hydra.utils.instantiate(self.cfg.policy)
        self.model.load_state_dict(payload['state_dict'])
        self.model.to(self.device).eval()
        
        print("Starting GPU warmup with T=3 history...")
        self.warmup()
        print("RTX 5090 is hot and ready.")

    @torch.no_grad()
    def warmup(self):
        # Shape: [Batch=1, Time=3, Channels=3, H=224, W=384]
        dummy_rgb = torch.randn(1, 3, 3, 224, 384).to(self.device)
        dummy_qpos = torch.randn(1, 3, 7).to(self.device)
        obs = {'cam_front': dummy_rgb, 'qpos': dummy_qpos}
        for _ in range(5):
            _ = self.model.predict_action(obs)
        torch.cuda.synchronize()

    @torch.no_grad()
    def infer(self, obs_dict):
        """
        obs_dict contains:
        - 'cam_front': List of 3 CV2 images
        - 'qpos': List of 3 joint+gripper arrays
        """
        t0 = time.perf_counter()
        
        # 1. Process Images: [T, C, H, W]
        img_tensors = []
        for img in obs_dict['cam_front']:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_res = cv2.resize(img_rgb, (384, 224)) # Width, Height matching warmup
            img_t = torch.from_numpy(img_res).permute(2, 0, 1).float() / 255.0
            img_tensors.append(img_t)
        
        # Stack to [1, 3, 3, 224, 384]
        combined_img = torch.stack(img_tensors).unsqueeze(0).to(self.device)
        
        # 2. Process Qpos: [1, 3, 7]
        combined_qpos = torch.from_numpy(np.array(obs_dict['qpos'])).float().unsqueeze(0).to(self.device)

        obs = {'cam_front': combined_img, 'qpos': combined_qpos}
        action_dict = self.model.predict_action(obs)
        
        dt = (time.perf_counter() - t0) * 1000
        print(f"GPU Compute (T=3): {dt:.2f}ms")
        
        # Return first action in the horizon
        return action_dict[0, 0, :].cpu().numpy().tolist()

def main():
    args = get_args()
    engine = DP2InferenceEngine(args.ckpt)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    
    header_struct = struct.Struct("Q")
    while True:
        conn, addr = server.accept()
        print(f"Robot {addr} Connected.")
        buffer = b""
        try:
            while True:
                while len(buffer) < header_struct.size:
                    chunk = conn.recv(32768) # Increased for multi-image payload
                    if not chunk: raise ConnectionError
                    buffer += chunk
                msg_size = header_struct.unpack(buffer[:header_struct.size])[0]
                buffer = buffer[header_struct.size:]
                while len(buffer) < msg_size:
                    chunk = conn.recv(32768)
                    if not chunk: raise ConnectionError
                    buffer += chunk
                
                # Unpack the history observations
                data = pickle.loads(buffer[:msg_size])
                buffer = buffer[msg_size:]
                
                # Decode all images in the history list
                decoded_imgs = [cv2.imdecode(img_buf, cv2.IMREAD_COLOR) for img_buf in data['img_history']]
                
                obs_payload = {
                    'cam_front': decoded_imgs,
                    'qpos': data['qpos_history']
                }
                
                action = engine.infer(obs_payload)
                
                resp = pickle.dumps(action)
                conn.sendall(header_struct.pack(len(resp)) + resp)
        except Exception as e:
            print(f"Disconnect: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    main()