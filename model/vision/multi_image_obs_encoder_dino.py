import sys
from pathlib import Path
import torch
import torch.nn as nn

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.parent))
else:
    sys.path.append(str(Path(__file__).parent))
from model.vision.dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16


class MultiImageObsEncoder(nn.Module):
    def __init__(self,
            shape_meta: dict,
            weights_path: str = "weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
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
        key_model_map['rgb'] = dinov3_vits16(pretrained=True, weights=weights_path)
        # key_model_map['rgb'] = dinov3_vitb16(pretrained=True, weights=weights_path)
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