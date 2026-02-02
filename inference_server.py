import socket, pickle, struct, torch, cv2, time, dill, hydra, argparse
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from lightning import LightningModule
from policy.dp2_dino import DP2

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

class DP2InferenceEngine:
    def __init__(self, ckpt_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ckpt_path = Path(ckpt_path)
        config_path = ckpt_path.parent / ".hydra" / "config.yaml"
        if not config_path.exists(): 
            config_path = ckpt_path.parents[1] / ".hydra" / "config.yaml"
        
        self.cfg = OmegaConf.load(config_path)
        with open(ckpt_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill, map_location=self.device)
        
        self.model: LightningModule = hydra.utils.instantiate(self.cfg.policy)
        self.model.load_state_dict(payload['state_dict'])
        self.model.to(self.device).eval()
        
        print("--- RTX 5090: Warming Up ---")
        self.warmup()

    @torch.no_grad()
    def warmup(self):
        d_rgb = torch.randn(1, 3, 3, 224, 384).to(self.device)
        d_qpos = torch.randn(1, 3, 7).to(self.device)
        for _ in range(5): 
            _ = self.model.predict_action({'cam_front': d_rgb, 'qpos': d_qpos})
        torch.cuda.synchronize()

    @torch.no_grad()
    def infer(self, obs_dict):
        try:
            img_tensors = [torch.from_numpy(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (384, 224))).permute(2, 0, 1).float() / 255.0 for img in obs_dict['cam_front']]
            rgb = torch.stack(img_tensors).unsqueeze(0).to(self.device)
            qpos = torch.from_numpy(np.array(obs_dict['qpos'])).float().unsqueeze(0).to(self.device)

            out = self.model.predict_action({'cam_front': rgb, 'qpos': qpos})
            torch.cuda.synchronize() 
            
            actions = out['action'] if isinstance(out, dict) else out
            return actions[0, 0, :].cpu().numpy().tolist() if len(actions.shape) == 3 else actions[0, :].cpu().numpy().tolist()
        except Exception as e:
            print(f"Inference Error: {e}")
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    
    engine = DP2InferenceEngine(args.ckpt)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server.bind(('0.0.0.0', 9999))
    server.listen(1)
    print(f"Server Online. Port: 9999")

    header_struct = struct.Struct("Q")
    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Inference Active: {addr}")
        buffer = b""
        try:
            while True:
                while len(buffer) < header_struct.size:
                    chunk = conn.recv(65536)
                    if not chunk: raise ConnectionError
                    buffer += chunk
                msg_size = header_struct.unpack(buffer[:header_struct.size])[0]
                buffer = buffer[header_struct.size:]
                while len(buffer) < msg_size: buffer += conn.recv(65536)
                data = pickle.loads(buffer[:msg_size]); buffer = buffer[msg_size:]

                obs = {'cam_front': [cv2.imdecode(i, 1) for i in data['img_history']], 'qpos': data['qpos_history']}
                action = engine.infer(obs)
                if action is None: break 
                
                resp = pickle.dumps(action)
                conn.sendall(header_struct.pack(len(resp)) + resp)
        except Exception: pass
        finally: conn.close()

if __name__ == "__main__": main()