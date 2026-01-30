import sys
from pathlib import Path
from typing import List, Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.parent))
else:
    sys.path.append(str(Path(__file__).parent))
from model.vision.dinov3.hub.backbones import dinov3_vits16


class ObsTokenizer(nn.Module):
    def __init__(self,
            shape_meta: dict,
            weights_path: str = "weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            n_layer: int = 4,
            p_drop_attn: float = 0.1,
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B, T, D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        for key, attr in shape_meta['obs'].items():
            shape = tuple(attr['shape'])
            type = attr['type']
            if type == 'low_dim':
                D = shape[-1]
                low_dim_keys.append(key)
            elif type == 'rgb':
                rgb_keys.append(key)

        self.image_tokenizer = dinov3_vits16(pretrained=True, weights=weights_path)
        self.image_tokenizer.eval()  # set to eval mode
        self.image_tokenizer.requires_grad_(False)  # freeze weights
        embed_dim = self.image_tokenizer.embed_dim
        self.state_tokenizer = nn.Sequential(
            nn.Linear(D, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # obs encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=4*embed_dim,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True # important for stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layer
        )

        self.shape_meta = shape_meta

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def forward(self, images: List[torch.Tensor], state: torch.Tensor):
        tokens = []
        batch_size = state.shape[0]
        with torch.no_grad():
            for img in images:
                img_token = self.image_tokenizer(img.reshape(-1, *img.shape[-3:]))  # B, embed_dim
                tokens.append(img_token.reshape(batch_size, -1, img_token.shape[-1]))
        state_token = self.state_tokenizer(state)
        tokens.append(state_token)
        tokens = self.encoder(torch.cat(tokens, dim=1))
        return tokens
    
if __name__ == "__main__":
    shape_meta = {
        'obs': {
            'cam_head': {'shape': (3, 224, 224), 'type': 'rgb'},
            'cam_left': {'shape': (3, 224, 224), 'type': 'rgb'},
            'cam_right': {'shape': (3, 224, 224), 'type': 'rgb'},
            'state': {'shape': (14,), 'type': 'low_dim'}
        }
    }
    model = ObsTokenizer(shape_meta)
    # image = {
    #     'cam_head': torch.randn(2, 3, 224, 224),
    #     'cam_left': torch.randn(2, 3, 224, 224),
    #     'cam_right': torch.randn(2, 3, 224, 224),
    # }
    image = [
        torch.randn(2, 3, 224, 224),
        torch.randn(2, 3, 224, 224),
        torch.randn(2, 3, 224, 224),
    ]
    state = torch.randn(2, 3, 14)
    out = model(image, state)
    print(out.shape)
