import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2Model

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.parent))
else:
    sys.path.append(str(Path(__file__).parent))
from model.vision.dinov3.hub.backbones import Weights, dinov3_vits16, dinov3_vitb16


class MultiImageObsEncoder(nn.Module):
    def __init__(self,
            shape_meta: dict,
            weights_path: str = "weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            backbone_source: str = "auto",
            huggingface_model_name: str = "facebook/dinov2-small",
            # weights_path: str = "weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        low_dim_shapes = dict()
        key_model_map = nn.ModuleDict()

        # vision backbone
        key_model_map['rgb'] = self._build_rgb_backbone(
            weights_path=weights_path,
            backbone_source=backbone_source,
            huggingface_model_name=huggingface_model_name,
        )
        key_model_map['rgb'].eval()  # set to eval mode
        key_model_map['rgb'].requires_grad_(False)  # freeze weights

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                low_dim_keys.append(key)
                low_dim_shapes[key] = shape
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
            
        # low dim encoders
        for key in low_dim_keys:
            shape = low_dim_shapes[key]
            model = nn.Sequential(
                nn.Linear(shape[0], 32),
                nn.Dropout(0.1),
                nn.GELU(),
                nn.Linear(32, 32),
            )
            key_model_map[key] = model

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_model_map = key_model_map

    @staticmethod
    def _build_rgb_backbone(weights_path, backbone_source: str, huggingface_model_name: str):
        source = backbone_source.strip().lower() if isinstance(backbone_source, str) else "auto"
        if source == "huggingface":
            return _HFDinoV2Backbone(huggingface_model_name)

        if source == "dinov3":
            resolved_weights = MultiImageObsEncoder._resolve_backbone_weights(weights_path)
            return dinov3_vits16(pretrained=True, weights=resolved_weights)

        if source != "auto":
            raise ValueError(
                f"Unsupported backbone_source '{backbone_source}'. Expected one of: auto, dinov3, huggingface."
            )

        try:
            resolved_weights = MultiImageObsEncoder._resolve_backbone_weights(weights_path)
            return dinov3_vits16(pretrained=True, weights=resolved_weights)
        except Exception:
            return _HFDinoV2Backbone(huggingface_model_name)

    @staticmethod
    def _resolve_backbone_weights(weights_path):
        if isinstance(weights_path, str):
            normalized = weights_path.strip()
            if normalized:
                candidate = Path(normalized).expanduser()
                if candidate.exists():
                    return str(candidate)

                lowered = normalized.lower()
                if "lvd1689m" in lowered:
                    return Weights.LVD1689M
                if "sat493m" in lowered:
                    return Weights.SAT493M
        if weights_path is None:
            return Weights.LVD1689M
        return weights_path

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def forward(self, obs_dict):
        features = list()
        # process rgb input
        with torch.no_grad():
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                imgs.append(img)

            batch_size = img.shape[0]
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            features.append(feature)
        
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            features.append(self.key_model_map[key](data))
            # features.append(data)
        
        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape


class _HFDinoV2Backbone(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name)
        image_mean = getattr(self.image_processor, "image_mean", [0.485, 0.456, 0.406])
        image_std = getattr(self.image_processor, "image_std", [0.229, 0.224, 0.225])
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pixel_values = (image - self.image_mean) / self.image_std
        outputs = self.model(pixel_values=pixel_values)
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]


if __name__ == "__main__":
    shape_meta = {
        'obs': {
            'cam_head': {'shape': (3, 224, 224), 'type': 'rgb'},
            'cam_left': {'shape': (3, 224, 224), 'type': 'rgb'},
            'cam_right': {'shape': (3, 224, 224), 'type': 'rgb'},
            'qpos': {'shape': (14,), 'type': 'low_dim'}
        }
    }
    model = MultiImageObsEncoder(shape_meta)
    image = {
        'cam_head': torch.randn(2, 3, 224, 224),
        'cam_left': torch.randn(2, 3, 224, 224),
        'cam_right': torch.randn(2, 3,  224, 224),
        'qpos': torch.randn(2, 14)
    }
    output = model(image)
    print(output.shape)
    print(model.output_shape())
