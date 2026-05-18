import socket, pickle, struct, torch, cv2, time, dill, hydra, argparse
import numpy as np
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf
from lightning import LightningModule

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

UR10E_ARM_DOF = 6
UR10E_DEPLOY_ACTION_DIM = 7
UR10E_LEGACY_ACTION_DIM = 8


def _last_dim_shape(shape) -> Optional[int]:
    if shape is None:
        return None
    shape = tuple(shape)
    return int(shape[-1]) if len(shape) > 0 else None


def _format_ur10e_qpos(qpos, expected_dim: Optional[int]) -> np.ndarray:
    """Format qpos history for the checkpoint while keeping deployment 7D canonical.

    Deployment observations are expected to be [joint_0..joint_5, gripper].
    Some older checkpoints may have been trained with an extra compatibility
    column before gripper. For those, insert a blank column and keep gripper as
    the final dimension.
    """
    qpos_arr = np.asarray(qpos, dtype=np.float32)
    if qpos_arr.shape[-1] == UR10E_DEPLOY_ACTION_DIM:
        canonical = qpos_arr
    elif qpos_arr.shape[-1] == UR10E_LEGACY_ACTION_DIM:
        canonical = np.concatenate(
            [qpos_arr[..., :UR10E_ARM_DOF], qpos_arr[..., -1:]],
            axis=-1,
        )
    elif qpos_arr.shape[-1] == UR10E_ARM_DOF:
        gripper = np.zeros((*qpos_arr.shape[:-1], 1), dtype=np.float32)
        canonical = np.concatenate([qpos_arr, gripper], axis=-1)
    else:
        raise ValueError(f"Unsupported UR10e qpos shape {qpos_arr.shape}")

    if expected_dim is None or expected_dim == canonical.shape[-1]:
        return canonical

    if expected_dim == UR10E_LEGACY_ACTION_DIM:
        blank = np.zeros((*canonical.shape[:-1], 1), dtype=np.float32)
        return np.concatenate(
            [canonical[..., :UR10E_ARM_DOF], blank, canonical[..., -1:]],
            axis=-1,
        )

    if expected_dim == UR10E_ARM_DOF:
        return canonical[..., :UR10E_ARM_DOF]

    raise ValueError(
        f"Checkpoint expects qpos dim {expected_dim}, cannot adapt deployment qpos shape {qpos_arr.shape}"
    )


def _format_ur10e_action_for_deployment(action) -> np.ndarray:
    """Return [6 joint targets, gripper] regardless of old/new model width."""
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_arr.size < UR10E_DEPLOY_ACTION_DIM:
        raise ValueError(f"Expected at least 7 UR10e action values, got {action_arr.size}")

    return np.concatenate(
        [action_arr[:UR10E_ARM_DOF], action_arr[-1:]],
        axis=0,
    ).astype(np.float32, copy=False)


def _shape_meta_dim(cfg, key: str) -> Optional[int]:
    try:
        if key == "action":
            return _last_dim_shape(cfg.shape_meta.action.shape)
        return _last_dim_shape(cfg.shape_meta.obs[key].shape)
    except Exception:
        return None


def _clone_param_dict(value):
    if isinstance(value, torch.nn.ParameterDict):
        return torch.nn.ParameterDict(
            {key: _clone_param_dict(item) for key, item in value.items()}
        )
    return value.detach().clone()


def _adapt_vector_dim(value, target_dim: Optional[int], key_name: str):
    cloned = value.detach().clone()
    if target_dim is None or cloned.ndim != 1 or cloned.shape[0] == target_dim:
        return cloned

    current_dim = cloned.shape[0]
    if current_dim == UR10E_LEGACY_ACTION_DIM and target_dim == UR10E_DEPLOY_ACTION_DIM:
        return torch.cat([cloned[:UR10E_ARM_DOF], cloned[-1:]], dim=0)

    if current_dim == UR10E_DEPLOY_ACTION_DIM and target_dim == UR10E_LEGACY_ACTION_DIM:
        neutral = 1.0 if key_name in {"scale", "std"} else 0.0
        blank = torch.full_like(cloned[:1], neutral)
        return torch.cat([cloned[:UR10E_ARM_DOF], blank, cloned[-1:]], dim=0)

    print(
        f"[WARN] Keeping normalizer tensor '{key_name}' with dim {current_dim}; "
        f"cannot adapt to expected dim {target_dim}."
    )
    return cloned


def _adapt_normalizer_field(field, target_dim: Optional[int]):
    adapted = torch.nn.ParameterDict()
    for key, value in field.items():
        if isinstance(value, torch.nn.ParameterDict):
            adapted[key] = _adapt_normalizer_field(value, target_dim)
        else:
            adapted[key] = _adapt_vector_dim(value, target_dim, key)
    return adapted


def _restore_normalizer(model: LightningModule, payload, cfg) -> None:
    if not hasattr(model, "normalizer"):
        return

    state_dict = payload.get("state_dict", payload)
    normalizer_state = {
        key.removeprefix("normalizer."): value
        for key, value in state_dict.items()
        if key.startswith("normalizer.")
    }
    if not normalizer_state:
        print("[WARN] Checkpoint has no normalizer state.")
        return

    model.normalizer.load_state_dict(normalizer_state, strict=False)

    params = model.normalizer.params_dict
    if "action" in params:
        action_dim = _shape_meta_dim(cfg, "action")
        params["action"] = _adapt_normalizer_field(params["action"], action_dim)
    if "qpos" in params:
        qpos_dim = _shape_meta_dim(cfg, "qpos")
        params["qpos"] = _adapt_normalizer_field(params["qpos"], qpos_dim)
    elif "action" in params:
        qpos_dim = _shape_meta_dim(cfg, "qpos")
        params["qpos"] = _adapt_normalizer_field(params["action"], qpos_dim)
        print("[WARN] Checkpoint normalizer had no 'qpos'; copied/adapted it from 'action'.")

    print("Loaded normalizer fields:", ", ".join(params.keys()))


def _load_compatible_state_dict(model: LightningModule, payload, cfg) -> None:
    """Load checkpoint tensors that match the instantiated model.

    Some DP2-DINO checkpoints were trained with the torch-hub DINOv3 module
    whose keys look like `blocks.*`, while a freshly instantiated encoder may
    expose HF-style keys such as `model.encoder.layer.*`. The encoder can still
    initialize from its pretrained weights, so inference should load the policy
    and normalizer weights that match and skip incompatible backbone-key drift.
    """
    state_dict = payload.get("state_dict", payload)
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []

    for key, value in state_dict.items():
        if key.startswith("normalizer."):
            continue
        target = model_state.get(key)
        if target is not None and tuple(target.shape) == tuple(value.shape):
            compatible_state[key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    print(
        "Loaded compatible checkpoint tensors: "
        f"{len(compatible_state)}/{len(state_dict)} "
        f"(skipped {len(skipped)} incompatible/unexpected tensors)."
    )
    if skipped:
        print("Skipped checkpoint key examples:", ", ".join(skipped[:5]))
    if missing:
        print("Missing model key examples:", ", ".join(missing[:5]))
    if unexpected:
        print("Unexpected loaded key examples:", ", ".join(unexpected[:5]))
    _restore_normalizer(model, payload, cfg)


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
        _load_compatible_state_dict(self.model, payload, self.cfg)
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
        self.low_dim_shapes = {
            key: tuple(self.cfg.shape_meta.obs[key].shape)
            for key in self.low_dim_keys
        }
        
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
                expected_dim = _last_dim_shape(self.low_dim_shapes.get(key))
                low_dim_obs = obs_dict[key]
                if key == "qpos":
                    low_dim_obs = _format_ur10e_qpos(low_dim_obs, expected_dim)
                model_obs[key] = torch.from_numpy(np.array(low_dim_obs)).float().unsqueeze(0).to(self.device)

            for passthrough_key in ["instruction", "object_prompt", "initial_object_pose", "object_pose", "pose"]:
                if passthrough_key in obs_dict:
                    model_obs[passthrough_key] = obs_dict[passthrough_key]

            out = self.model.predict_action(model_obs)
            torch.cuda.synchronize() 
            
            actions = out['action'] if isinstance(out, dict) else out
            action = actions[0, 0, :].cpu().numpy() if len(actions.shape) == 3 else actions[0, :].cpu().numpy()
            return _format_ur10e_action_for_deployment(action).tolist()
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
