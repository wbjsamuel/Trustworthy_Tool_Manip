import socket, pickle, struct, torch, cv2, time, dill, hydra, argparse
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from lightning import LightningModule

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
        self.rgb_keys = [
            key
            for key, attr in self.cfg.shape_meta.obs.items()
            if attr.get("type", "low_dim") == "rgb"
        ]
        self.low_dim_keys = [
            key
            for key, attr in self.cfg.shape_meta.obs.items()
            if attr.get("type", "low_dim") == "low_dim"
        ]
        
        print("--- RTX 5090: Warming Up ---")
        self.warmup()

    @torch.no_grad()
    def warmup(self):
        warmup_obs = {}
        for key in self.rgb_keys:
            shape = tuple(self.cfg.shape_meta.obs[key].shape)
            warmup_obs[key] = torch.randn(1, 3, *shape).to(self.device)
        for key in self.low_dim_keys:
            shape = tuple(self.cfg.shape_meta.obs[key].shape)
            warmup_obs[key] = torch.randn(1, 3, *shape).to(self.device)
        for _ in range(5): 
            _ = self.model.predict_action(warmup_obs)
        torch.cuda.synchronize()

    @torch.no_grad()
    def infer(self, obs_dict):
        try:
            model_obs = {}
            for key in self.rgb_keys:
                if key not in obs_dict:
                    continue
                _, height, width = tuple(self.cfg.shape_meta.obs[key].shape)
                img_tensors = [
                    torch.from_numpy(
                        cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (width, height))
                    ).permute(2, 0, 1).float() / 255.0
                    for img in obs_dict[key]
                ]
                model_obs[key] = torch.stack(img_tensors).unsqueeze(0).to(self.device)
            for key in self.low_dim_keys:
                if key not in obs_dict:
                    continue
                model_obs[key] = torch.from_numpy(np.array(obs_dict[key])).float().unsqueeze(0).to(self.device)

            for passthrough_key in ["instruction", "object_prompt", "initial_object_pose", "object_pose", "pose"]:
                if passthrough_key in obs_dict:
                    model_obs[passthrough_key] = obs_dict[passthrough_key]

            out = self.model.predict_action(model_obs)
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

                obs = {}
                image_history = [cv2.imdecode(i, 1) for i in data['img_history']]
                if len(engine.rgb_keys) > 0:
                    obs[engine.rgb_keys[0]] = image_history
                if 'qpos_history' in data and 'qpos' in engine.low_dim_keys:
                    obs['qpos'] = data['qpos_history']
                for passthrough_key in ["instruction", "object_prompt", "initial_object_pose", "object_pose", "pose"]:
                    if passthrough_key in data:
                        obs[passthrough_key] = data[passthrough_key]
                action = engine.infer(obs)
                if action is None: break 
                
                resp = pickle.dumps(action)
                conn.sendall(header_struct.pack(len(resp)) + resp)
        except Exception: pass
        finally: conn.close()

if __name__ == "__main__": main()
