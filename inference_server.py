import socket
import pickle
import struct
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from omegaconf import OmegaConf

# Policy-Lightning specific imports
from policy_lightning.models.diffusion_policy import DiffusionPolicy
from policy_lightning.utils.setup_utils import build_model

def get_args():
    parser = argparse.ArgumentParser(description="DP2 Inference Server (RTX 5090)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--port", type=int, default=9999, help="Socket port")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    return parser.parse_args()

class DP2Server:
    def __init__(self, ckpt_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Config from checkpoint directory
        ckpt_path = Path(ckpt_path)
        config_path = ckpt_path.parent / ".hydra" / "config.yaml" 
        if not config_path.exists():
            # Fallback for some Lightning structures
            config_path = ckpt_path.parents[1] / ".hydra" / "config.yaml"
        
        print(f"Loading config from: {config_path}")
        self.cfg = OmegaConf.load(config_path)
        
        # 2. Build Model (DP2)
        # We load the weights into the model architecture defined by the config
        self.model = DiffusionPolicy.load_from_checkpoint(
            str(ckpt_path), 
            cfg=self.cfg, 
            map_location=self.device
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Warmup for RTX 5090
        print("Warming up GPU...")
        self.warmup()

    @torch.no_grad()
    def warmup(self):
        # Create dummy observation based on config
        obs = {
            'rgb': torch.randn(1, 1, 3, 224, 224).to(self.device), # [B, T, C, H, W]
            'joint_pos': torch.randn(1, 1, 7).to(self.device)
        }
        _ = self.model(obs)

    def process_inference(self, img, joints):
        # DP2 expects observation horizons. 
        # Usually, this is 1 (current) or 2 (current + previous).
        # Adjust preprocessing to match your training config
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device) # [1, 1, 3, 224, 224]
        
        joint_tensor = torch.from_numpy(np.array(joints)).float().unsqueeze(0).unsqueeze(0).to(self.device)

        obs = {
            'rgb': img_tensor,
            'joint_pos': joint_tensor
        }

        # The model __call__ in DP2 typically runs the diffusion reverse process
        # and returns a sequence of actions [Batch, Horizon, Action_Dim]
        action_pred = self.model(obs) 
        
        # We take the first action of the predicted horizon
        # Action is usually [joints (6) + gripper (1)]
        action = action_pred[0, 0, :].cpu().numpy().tolist()
        return action

def run_server():
    args = get_args()
    policy_server = DP2Server(args.ckpt)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((args.host, args.port))
    server_socket.listen(1)
    
    print(f"RTX 5090 Server Online at {args.host}:{args.port}")

    payload_size = struct.calcsize("Q")

    while True:
        conn, addr = server_socket.accept()
        print(f"Connection from {addr}")
        data = b""
        
        try:
            while True:
                # Receive size
                while len(data) < payload_size:
                    packet = conn.recv(8192)
                    if not packet: break
                    data += packet
                if not data: break
                
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]

                # Receive payload
                while len(data) < msg_size:
                    data += conn.recv(8192)
                
                msg_data = data[:msg_size]
                data = data[msg_size:]
                
                # Unpickle
                obs_data = pickle.loads(msg_data)
                img = cv2.imdecode(obs_data['img'], cv2.IMREAD_COLOR)
                
                # Inference
                action = policy_server.process_inference(img, obs_data['joints'])
                
                # Send Back
                response = pickle.dumps(action)
                conn.sendall(struct.pack("Q", len(response)) + response)
                
        except Exception as e:
            print(f"Inference Error: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    run_server()