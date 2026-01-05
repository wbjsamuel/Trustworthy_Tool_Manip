from typing import Dict, Tuple, Union, Optional
import copy
import torch
import torch.nn as nn
import torchvision
from model.vision.crop_randomizer import CropRandomizer
from common.pytorch_util import dict_apply, replace_submodules


class PoseEstimator(nn.Module):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            use_group_norm: bool=False,
            share_rgb_model: bool=False,
            imagenet_norm: bool=False,
            # Transformer parameters
            n_emb: int = 256,
            n_head: int = 8,
            n_layer: int = 4,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            # Pose output dimension (x, y, z, rx, ry, rz)
            pose_output_dim: int = 6,
    ):
        """
        Transformer-based Pose Estimator
        Input: RGB observation(s) + current object pose (x,y,z,rx,ry,rz)
        Output: Target object pose (x,y,z,rx,ry,rz)
        
        Assumes rgb input: B,C,H,W
        Assumes pose input: B,6 (x,y,z,rx,ry,rz)
        """
        super().__init__()

        rgb_keys = list()
        pose_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        
        # Handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # Configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # Configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                
                # Configure crop
                this_cropper = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_cropper = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_cropper = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                
                # Combine transforms
                this_transform = nn.Sequential(this_resizer, this_cropper)
                key_transform_map[key] = this_transform
                
            elif type == 'low_dim':
                pose_keys.append(key)
        
        # ImageNet normalization
        self.imagenet_norm = imagenet_norm
        if imagenet_norm:
            self.normalize = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        
        self.rgb_keys = rgb_keys
        self.pose_keys = pose_keys
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.key_shape_map = key_shape_map
        self.share_rgb_model = share_rgb_model
        
        # Determine RGB feature dimension
        # Assume ResNet-style backbone outputs a 1D feature vector
        rgb_feature_dim = self._get_rgb_feature_dim()
        
        # Pose embedding
        pose_input_dim = 6  # (x, y, z, rx, ry, rz)
        self.pose_embed = nn.Sequential(
            nn.Linear(pose_input_dim, n_emb),
            nn.ReLU(),
            nn.Linear(n_emb, n_emb)
        )
        
        # RGB feature projection
        self.rgb_proj = nn.Linear(rgb_feature_dim, n_emb)
        
        # Positional embeddings for tokens
        # Token 0: pose embedding, Token 1+: RGB features (one per camera if multiple)
        max_tokens = 1 + len(rgb_keys)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layer
        )
        
        # Output head for target pose
        self.output_head = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.ReLU(),
            nn.Linear(n_emb, pose_output_dim)
        )
        
        self.pose_output_dim = pose_output_dim
        
    def _get_rgb_feature_dim(self) -> int:
        """
        Determine the output dimension of the RGB encoder.
        """
        # Create a dummy input to infer the output dimension
        if len(self.rgb_keys) == 0:
            return 0
        
        key = self.rgb_keys[0]
        shape = self.key_shape_map[key]
        dummy_input = torch.zeros(1, *shape)
        
        # Apply transforms
        if key in self.key_transform_map:
            dummy_input = self.key_transform_map[key](dummy_input)
        
        # Apply RGB model
        if self.share_rgb_model:
            model = self.key_model_map['rgb']
        else:
            model = self.key_model_map[key]
        
        with torch.no_grad():
            dummy_output = model(dummy_input)
        
        # Flatten spatial dimensions if present
        if len(dummy_output.shape) > 2:
            dummy_output = dummy_output.reshape(dummy_output.shape[0], -1)
        
        return dummy_output.shape[1]
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs: Dictionary containing:
                - RGB observations (one or more cameras): B,C,H,W
                - Current pose: B,6 (x,y,z,rx,ry,rz)
        
        Returns:
            Target pose: B,6 (x,y,z,rx,ry,rz)
        """
        B = obs[next(iter(obs))].shape[0]
        device = self.device
        
        # Process RGB features
        rgb_features = []
        for key in self.rgb_keys:
            x = obs[key]
            
            # Apply transforms
            if key in self.key_transform_map:
                x = self.key_transform_map[key](x)
            
            # ImageNet normalization
            if self.imagenet_norm:
                x = self.normalize(x)
            
            # Apply RGB encoder
            if self.share_rgb_model:
                model = self.key_model_map['rgb']
            else:
                model = self.key_model_map[key]
            
            features = model(x)
            
            # Flatten if needed
            if len(features.shape) > 2:
                features = features.reshape(B, -1)
            
            rgb_features.append(features)
        
        # Process pose input
        pose_input = None
        for key in self.pose_keys:
            if 'pose' in key.lower() or key in ['object_pose', 'current_pose', 'pose']:
                pose_input = obs[key]
                break
        
        if pose_input is None and len(self.pose_keys) > 0:
            # Use the first available low_dim key
            pose_input = obs[self.pose_keys[0]]
        
        assert pose_input is not None, "No pose input found in observations"
        assert pose_input.shape[-1] == 6, f"Expected pose dimension 6, got {pose_input.shape[-1]}"
        
        # Embed pose and RGB features
        pose_token = self.pose_embed(pose_input).unsqueeze(1)  # B,1,n_emb
        rgb_tokens = [self.rgb_proj(feat).unsqueeze(1) for feat in rgb_features]  # Each: B,1,n_emb
        
        # Concatenate all tokens
        tokens = torch.cat([pose_token] + rgb_tokens, dim=1)  # B,n_tokens,n_emb
        
        # Add positional embeddings
        n_tokens = tokens.shape[1]
        tokens = tokens + self.pos_emb[:, :n_tokens, :]
        tokens = self.drop(tokens)
        
        # Apply transformer
        transformer_out = self.transformer(tokens)  # B,n_tokens,n_emb
        
        # Use the first token (pose token) for prediction
        pose_features = transformer_out[:, 0, :]  # B,n_emb
        
        # Predict target pose
        target_pose = self.output_head(pose_features)  # B,6
        
        return target_pose
