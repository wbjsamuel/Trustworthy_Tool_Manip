import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, SiglipVisionModel, AutoProcessor
import pytorch_lightning as pl
import wandb

class ImageEncoder(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        # DINOv2
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True)
        self.dinov2.to(self.device)
        self.dinov2.eval()

        # SigLIP
        self.siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.siglip = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        self.siglip.to(self.device)
        self.siglip.eval()

    def forward(self, image):
        with torch.no_grad():
            # DINOv2 features
            dino_features = self.dinov2.forward_features(image.to(self.device))
            
            # SigLIP features
            inputs = self.siglip_processor(images=image, return_tensors="pt").to(self.device)
            siglip_features = self.siglip(**inputs).pooler_output

        # Concatenate features
        return torch.cat([dino_features['x_norm_patchtokens'].mean(dim=1), siglip_features], dim=1)

class LanguageEncoder(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.model = T5EncoderModel.from_pretrained("t5-base")
        self.model.to(self.device)
        self.model.eval()

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

class Stage1Transformer(pl.LightningModule):
    def __init__(self, image_feature_dim=1536, language_feature_dim=768, pose_dim=7, num_heads=8, num_layers=6, embed_dim=512, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.image_encoder = ImageEncoder()
        self.language_encoder = LanguageEncoder()

        self.image_proj = nn.Linear(image_feature_dim, embed_dim)
        self.language_proj = nn.Linear(language_feature_dim, embed_dim)
        self.pose_proj = nn.Linear(pose_dim, embed_dim)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        self.output_head = nn.Linear(embed_dim, pose_dim)
        
        self.criterion = nn.MSELoss()

    def forward(self, image, current_tool_pose, instruction):
        image_features = self.image_encoder(image)
        language_features = self.language_encoder(instruction)

        image_embedding = self.image_proj(image_features)
        language_embedding = self.language_proj(language_features)
        pose_embedding = self.pose_proj(current_tool_pose)
        
        combined_input = torch.stack([image_embedding, language_embedding, pose_embedding], dim=1)

        transformer_output = self.transformer(combined_input)

        output_embedding = transformer_output[:, 0, :]

        predicted_pose = self.output_head(output_embedding)
        return predicted_pose

    def training_step(self, batch, batch_idx):
        image = batch['image']
        instruction = batch['instruction']
        current_tool_pose = batch['current_tool_pose']
        target_pose = batch['target_pose']

        predicted_pose = self(image, current_tool_pose, instruction)
        loss = self.criterion(predicted_pose, target_pose)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        instruction = batch['instruction']
        current_tool_pose = batch['current_tool_pose']
        target_pose = batch['target_pose']

        predicted_pose = self(image, current_tool_pose, instruction)
        loss = self.criterion(predicted_pose, target_pose)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

if __name__ == '__main__':
    # Example Usage
    model = Stage1Transformer()
    
    # Dummy inputs
    image = torch.randn(2, 3, 224, 224) # Batch of 2
    tool_pose = torch.randn(2, 7) 
    instruction = ["pick up the tool", "place the block on the red square"]

    # Forward pass
    predicted_pose = model(image, tool_pose, instruction)
    
    print("Predicted Pose:", predicted_pose)
    print("Predicted Pose shape:", predicted_pose.shape)

