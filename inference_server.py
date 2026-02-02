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
        if not config_path.exists(): config_path = ckpt_path.parents[1] / ".hydra" / "config.yaml"
        
        self.cfg = OmegaConf.load(config_path)
        with open(ckpt_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill, map_location=self.device)
        
        self.model: LightningModule = hydra.utils.instantiate(self.cfg.policy)
        self.model.load_state_dict(payload['state_dict'])
        self.model.to(self.device).eval()
        
        print("Starting GPU warmup (T=3, 384x224)...")
        self.warmup()

    @torch.no_grad()
    def warmup(self):
        d_rgb = torch.randn(1, 3, 3, 224, 384).to(self.device)
        d_qpos = torch.randn(1, 3, 7).to(self.device)
        for _ in range(5): _ = self.model.predict_action({'cam_front': d_rgb, 'qpos': d_qpos})
        torch.cuda.synchronize()

    @torch.no_grad()
    def infer(self, obs_dict):
        t0 = time.perf_counter()
        try:
            img_tensors = []
            for img in obs_dict['cam_front']:
                img_res = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (384, 224))
                img_tensors.append(torch.from_numpy(img_res).permute(2, 0, 1).float() / 255.0)
            
            rgb = torch.stack(img_tensors).unsqueeze(0).to(self.device)
            qpos = torch.from_numpy(np.array(obs_dict['qpos'])).float().unsqueeze(0).to(self.device)

            action_dict = self.model.predict_action({'cam_front': rgb, 'qpos': qpos})
            torch.cuda.synchronize() 
            
            actions = action_dict['action'] if isinstance(action_dict, dict) else action_dict
            
            if len(actions.shape) == 3: res_action = actions[0, 0, :].cpu().numpy().tolist()
            elif len(actions.shape) == 2: res_action = actions[0, :].cpu().numpy().tolist()
            else: res_action = actions.flatten().cpu().numpy().tolist()

            # Log if we exceed our 30Hz budget (33.3ms)
            dt = (time.perf_counter() - t0) * 1000
            if dt > 33.3: print(f"⚠️ SLOW INFERENCE: {dt:.1f}ms")
            
            return res_action
        except Exception as e:
            print(f"CRITICAL INFER ERROR: {e}")
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--hz", type=int, default=30)
    args = parser.parse_args()
    
    engine = DP2InferenceEngine(args.ckpt)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Enable TCP_NODELAY to reduce latency for the return action
    server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server.bind(('0.0.0.0', 9999))
    server.listen(1)
    
    header_struct = struct.Struct("Q")
    print(f"Server ready. Target: {args.hz}Hz")

    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Robot connected: {addr}")
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

                obs = {
                    'cam_front': [cv2.imdecode(i, 1) for i in data['img_history']],
                    'qpos': data['qpos_history']
                }
                
                action = engine.infer(obs)
                if action is None: break 
                
                resp = pickle.dumps(action)
                conn.sendall(header_struct.pack(len(resp)) + resp)
        except Exception as e: print(f"Connection ended: {e}")
        finally: conn.close()

if __name__ == "__main__": main()